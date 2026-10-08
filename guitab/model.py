"""CQT frame/onset guitar tab model.

input  log-CQT [B, T, 288] (36 bins/oct from E1), or [B, T, 288 + 180] with the time-sharp low band (sharp_bins)
  -> harmonic stack: 6 channels (h = 0.5, 1, 2, 3, 4, 5) aligned on the fundamental, 144 bins
     (48 semitones from E2), so a pitch's partials line up vertically for the convolutions;
     with sharp_bins, 6 more channels: the same stack of the sharp band (zero above it)
  -> 2D CNN (time resolution kept, frequency pooled 144 -> 12)
  -> transformer encoder over time (context for fingering, sustains, chords)
  -> heads per frame:
       tab    [B, T, 6, 21]  per string: silent or fret 0..19
       onset  [B, T, 6]      per string attack
       pitch  [B, T, 48]     auxiliary multi-pitch (MIDI 40..87)
Size presets: "mid" ~22M parameters, "large" ~50M.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .guitar import OPEN_PITCH, STRINGS

BASE_BIN = 36  # E2 in a CQT starting at E1 with 36 bins/oct
N_PITCH_BINS = 144
HARMONICS = [0.5, 1, 2, 3, 4, 5]
SIZES = {
    "mid": {"chans": (64, 128, 256), "d_model": 512, "layers": 6, "heads": 8},
    "large": {"chans": (96, 192, 384), "d_model": 768, "layers": 8, "heads": 12},
}


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())


class TabModel(nn.Module):
    def __init__(self, size="mid", n_classes=21, dropout=0.1, max_len=4096, sharp_bins=0):
        super().__init__()
        cfg = SIZES[size]
        c1, c2, c3 = cfg["chans"]
        d = cfg["d_model"]
        self.size = size
        self.sharp_bins = sharp_bins
        self.shifts = [int(round(36 * math.log2(h))) for h in HARMONICS]
        self.cnn = nn.Sequential(
            conv_block(len(HARMONICS) * (2 if sharp_bins else 1), c1), conv_block(c1, c1), nn.MaxPool2d((1, 3)),  # 144 -> 48 (semitones)
            conv_block(c1, c2), conv_block(c2, c2), nn.MaxPool2d((1, 2)),  # -> 24
            conv_block(c2, c3), conv_block(c3, c3), nn.MaxPool2d((1, 2)),  # -> 12
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(c3 * 12, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"])
        self.norm = nn.LayerNorm(d)
        self.tab_head = nn.Linear(d, 6 * n_classes)
        self.onset_head = nn.Linear(d, 6)
        self.pitch_head = nn.Linear(d, N_PITCH_BINS // 3)
        self.n_classes = n_classes

    def harmonic_stack(self, x):
        # x [B, T, 288] -> [B, 6, T, 144]; bins past the top of the CQT are zero
        pad = F.pad(x, (0, max(self.shifts) + BASE_BIN + N_PITCH_BINS - x.shape[-1]))
        return torch.stack([pad[..., BASE_BIN + s: BASE_BIN + s + N_PITCH_BINS] for s in self.shifts], dim=1)

    def forward(self, x):
        b, t, _ = x.shape
        if self.sharp_bins:
            n = x.shape[-1] - self.sharp_bins
            stack = torch.cat([self.harmonic_stack(x[..., :n]), self.harmonic_stack(x[..., n:])], dim=1)
        else:
            stack = self.harmonic_stack(x)
        h = self.cnn(stack)  # [B, C, T, 12]
        h = h.permute(0, 2, 1, 3).reshape(b, t, -1)
        h = self.proj(h) + self.pos[:, :t]
        h = self.norm(self.encoder(h))
        return {
            "tab": self.tab_head(h).view(b, t, 6, self.n_classes),
            "onset": self.onset_head(h),
            "pitch": self.pitch_head(h),
        }


def widen_input(state, model):
    """State dict of a model without the sharp band -> one for `model` (with it): the new input channels of the
    first convolution start at zero, so the widened model computes exactly the same function at step 0."""
    key = "cnn.0.0.weight"
    w = torch.zeros_like(model.state_dict()[key])
    w[:, :state[key].shape[1]] = state[key]
    return {**state, key: w}


def pitch_targets(tab):
    """Multi-pitch frame target [B, T, 48] (MIDI 40..87) from tab labels [B, T, 6]."""
    opens = torch.tensor([OPEN_PITCH[s] for s in STRINGS], device=tab.device)
    active = tab > 0
    pitch = (opens + tab.long() - 1 - 40).clamp(0, 47)
    out = torch.zeros(*tab.shape[:2], 48, device=tab.device)
    out.scatter_add_(2, pitch * active, active.float())
    return (out > 0).float()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
