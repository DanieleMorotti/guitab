"""CQT frame/onset guitar tab model.

input  log-CQT [B, T, 288] (36 bins/oct from E1)
  -> harmonic stack: 6 channels (h = 0.5, 1, 2, 3, 4, 5) aligned on the fundamental, 144 bins
     (48 semitones from E2), so a pitch's partials line up vertically for the convolutions
  -> 2D CNN (time resolution kept, frequency pooled 144 -> 12)
  -> time model (context for fingering, sustains, chords): transformer encoder, bidirectional GRU or dilated convolutions
  -> heads per frame:
       tab    [B, T, 6, 21]  per string: silent or fret 0..19
       onset  [B, T, 6]      per string attack
       pitch  [B, T, 48]     auxiliary multi-pitch (MIDI 40..87)
Size presets: "mid" 24M parameters, "large" 66M (transformer); lighter time models: "small" (4-layer 256-wide
transformer), "gru" (2-layer bidirectional GRU), "tcn" (dilated convolutions). "tiny" is TinyTabModel: a pitch-axis model
of ~0.1M parameters. Build any of them with build_model(size).
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
    "small": {"chans": (64, 128, 256), "d_model": 256, "layers": 4, "heads": 4},
    "gru": {"chans": (64, 128, 256), "d_model": 256, "layers": 2, "time": "gru", "hidden": 256},
    "tcn": {"chans": (64, 128, 256), "d_model": 256, "layers": 8, "time": "tcn"},  # dilations 1..128: +-255 frames (3 s)
    "tiny": {"chans": 32, "layers": 8},  # TinyTabModel
}


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())


class DilatedConvs(nn.Module):
    """Residual stack of dilated 1D convolutions over time (kernel 3, dilation 2**i): [B, T, d] -> [B, T, d]."""

    def __init__(self, d, layers, dropout):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.convs = nn.ModuleList(nn.Conv1d(d, d, 3, padding=2**i, dilation=2**i) for i in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        for norm, conv in zip(self.norms, self.convs):
            h = h + self.drop(conv(F.gelu(norm(h)).transpose(1, 2)).transpose(1, 2))
        return h


class TabModel(nn.Module):
    def __init__(self, size="mid", n_classes=21, dropout=0.1, max_len=4096):
        super().__init__()
        cfg = SIZES[size]
        c1, c2, c3 = cfg["chans"]
        d = cfg["d_model"]
        self.size = size
        self.shifts = [int(round(36 * math.log2(h))) for h in HARMONICS]
        self.cnn = nn.Sequential(
            conv_block(len(HARMONICS), c1), conv_block(c1, c1), nn.MaxPool2d((1, 3)),  # 144 -> 48 (semitones)
            conv_block(c1, c2), conv_block(c2, c2), nn.MaxPool2d((1, 2)),  # -> 24
            conv_block(c2, c3), conv_block(c3, c3), nn.MaxPool2d((1, 2)),  # -> 12
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(c3 * 12, d)
        self.time = cfg.get("time", "transformer")
        if self.time == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, max_len, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, dropout, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg["layers"])
        elif self.time == "gru":
            self.encoder = nn.GRU(d, cfg["hidden"], cfg["layers"], batch_first=True, bidirectional=True, dropout=dropout)
            d = 2 * cfg["hidden"]
        else:
            self.encoder = DilatedConvs(d, cfg["layers"], dropout)
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
        h = self.cnn(self.harmonic_stack(x))  # [B, C, T, 12]
        h = h.permute(0, 2, 1, 3).reshape(b, t, -1)
        h = self.proj(h)
        if self.time == "transformer":
            h = self.encoder(h + self.pos[:, :t])
        elif self.time == "gru":
            h = self.encoder(h)[0]
        else:
            h = self.encoder(h)
        h = self.norm(h)
        return {
            "tab": self.tab_head(h).view(b, t, 6, self.n_classes),
            "onset": self.onset_head(h),
            "pitch": self.pitch_head(h),
        }


class TinyTabModel(nn.Module):
    """Pitch-axis tab model (~0.1M parameters), in the spirit of basic_pitch: every layer is shared across pitches.

    harmonic stack [B, 6, T, 144] -> 2D convolutions (3 bins/semitone, then pooled to 48 semitones), one octave-wide
    -> dilated time convolutions (kernel 3 time x 3 semitones, dilation 2**i in time), still one feature vector per
       semitone, plus a learned per-semitone embedding (absolute pitch)
    -> heads read the pitch axis: pitch = shared linear per semitone; for string s, fret f sits at semitone
       open(s) + f, so each string has a small classifier over its 20-semitone window (fret scores and a silent score)
       and its onset is the onset score at those semitones weighted by the fret probabilities.
    """

    def __init__(self, size="tiny", n_classes=21, dropout=0.1):
        super().__init__()
        cfg = SIZES[size]
        c = cfg["chans"]
        self.size = size
        self.n_classes = n_classes
        self.shifts = [int(round(36 * math.log2(h))) for h in HARMONICS]
        self.front = nn.Sequential(
            nn.Conv2d(len(HARMONICS), c, (3, 5), padding=(1, 2)), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU(), nn.MaxPool2d((1, 3)),  # 144 -> 48 semitones
            nn.Conv2d(c, c, (1, 13), padding=(0, 6)), nn.BatchNorm2d(c), nn.GELU(),  # one octave
        )
        self.pitch_emb = nn.Parameter(torch.zeros(1, c, 1, 48))
        self.time_norms = nn.ModuleList(nn.BatchNorm2d(c) for _ in range(cfg["layers"]))
        self.time_convs = nn.ModuleList(nn.Conv2d(c, c, 3, padding=(2**i, 1), dilation=(2**i, 1))
                                        for i in range(cfg["layers"]))
        self.drop = nn.Dropout(dropout)
        self.pitch_out = nn.Conv2d(c, 2, 1)  # per semitone: pitch and onset scores
        self.string_mlp = nn.ModuleList(nn.Sequential(nn.Linear(c, 16), nn.GELU()) for _ in STRINGS)
        self.fret_score = nn.ModuleList(nn.Linear(16, 1) for _ in STRINGS)
        self.fret_bias = nn.Parameter(torch.zeros(6, n_classes - 1))
        self.silent_score = nn.ModuleList(nn.Linear(32, 1) for _ in STRINGS)
        self.windows = [OPEN_PITCH[s] - 40 for s in STRINGS]

    harmonic_stack = TabModel.harmonic_stack

    def forward(self, x):
        h = self.front(self.harmonic_stack(x)) + self.pitch_emb  # [B, C, T, 48]
        for norm, conv in zip(self.time_norms, self.time_convs):
            h = h + self.drop(conv(F.gelu(norm(h))))
        po = self.pitch_out(h)  # [B, 2, T, 48]
        h = h.permute(0, 2, 3, 1)  # [B, T, 48, C]
        tab, onset = [], []
        nf = self.n_classes - 1
        for i, o in enumerate(self.windows):
            z = self.string_mlp[i](h[:, :, o:o + nf])  # [B, T, 20, 16]
            fret = self.fret_score[i](z)[..., 0] + self.fret_bias[i]
            silent = self.silent_score[i](torch.cat([z.mean(2), z.amax(2)], -1))
            tab.append(torch.cat([silent, fret], -1))
            onset.append((torch.softmax(fret, -1) * po[:, 1, :, o:o + nf]).sum(-1))
        return {"tab": torch.stack(tab, 2), "onset": torch.stack(onset, -1), "pitch": po[:, 0]}


def build_model(size="mid"):
    return TinyTabModel(size) if size == "tiny" else TabModel(size)


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
