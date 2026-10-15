"""Tab language model and joint decoding (A1, opt-in: scripts/transcribe.py --music_lm).

A small causal transformer (3.3M parameters, scripts/train_tablm.py) trained on symbolic tabs only (acoustic / clean /
jazz guitar tracks of the DadaGP-derived SynthTab label archive) predicts the next chord. Decoding is speech-recognition
style: candidate attack groups come from the acoustic model's per-string onset peaks (>= tau, merged within `win` frames);
each string with a peak can take no note or one of its top-2 frets; a beam search picks the chord sequence maximising
    sum of acoustic log-probabilities + lam * LM log P(chord | previous chords) + beta * number of notes.
Notes that start without an onset peak (legato) pass through from the default extraction; then the D3 re-attack merge.
Tokens: PAD 0, BOS 1, EOC 2 (end of chord), note (string s 1..6, fret f 0..19) = 3 + (s - 1) * 20 + f; a chord lists its
notes from the low string to the high string, then EOC. See EXPERIMENTS.md (A1) for the tuning and the results.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn

from .audio import FPS
from .guitar import OPEN_PITCH, STRINGS
from .inference import extract_notes, merge_reattacks

VOCAB, CTX = 3 + 6 * 20, 256
HISTORY = 120  # tokens of chord history the decoder conditions on (~30-40 chords)
# tuned on half of GAPS validation and a third of GuitarSet validation (EXPERIMENTS.md, A1)
DEFAULT_DECODING = {"lam": 0.3, "beta": 1.0, "tau": 0.2, "beam": 4, "win": 4, "reattack": 0.75}


def tok(string, fret):
    return 3 + (string - 1) * 20 + fret


def chord_tokens(notes):
    """Tokens of a chord given as (string, fret, ...) tuples: low string first, then EOC."""
    return [tok(n[0], n[1]) for n in sorted(notes, key=lambda n: -n[0])] + [2]


class TabLM(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.zeros(1, CTX, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, VOCAB)

    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), 1)
        return self.out(self.norm(self.enc(self.emb(x) + self.pos[:, :t], mask=mask, is_causal=True)))


def load_tablm(path, device):
    lm = TabLM()
    lm.load_state_dict(torch.load(path, map_location="cpu"))
    return lm.to(device).eval()


def peaks(o, tau):
    return [t for t in range(1, len(o) - 1) if o[t] >= tau and o[t] >= o[t - 1] and o[t] > o[t + 1]]


def attack_groups(onset, tau, win):
    """[(frame, {column: (peak frame, onset prob)})]: per-string onset peaks merged across strings within win frames."""
    groups = []
    for t, c in sorted((t, col) for col in range(6) for t in peaks(onset[:, col], tau)):
        near = groups and t - groups[-1][0] <= win
        if near and (c not in groups[-1][1] or onset[t, c] > groups[-1][1][c][1]):
            groups[-1][1][c] = (t, float(onset[t, c]))
        else:  # a new attack (a weaker second peak on a string already in the group also starts one)
            groups.append((t, {c: (t, float(onset[t, c]))}))
    return groups


def candidates(tab, group, k_frets=2, k_chords=12):
    """[(acoustic score, ((string, fret, frame), ...))]: per string no note or one of its top frets; best k_chords
    combinations without repeated pitches, always including the empty chord."""
    opts = []
    for c, (t, o) in group.items():
        o = min(max(o, 1e-4), 1 - 1e-4)
        post = tab[t:t + 3, c, 1:].mean(0)
        post = post / max(post.sum(), 1e-9)
        top = [int(f) for f in np.argsort(post)[::-1][:k_frets] if post[f] >= 0.1]
        opts.append([(np.log(1 - o), None)] + [(np.log(o) + np.log(post[f]), (STRINGS[c], f, t)) for f in top])
    out = []
    for combo in itertools.product(*opts):
        notes = tuple(n for _, n in combo if n)
        pitches = [OPEN_PITCH[s] + f for s, f, _ in notes]
        if len(set(pitches)) == len(pitches):
            out.append((sum(a for a, _ in combo), notes))
    out.sort(key=lambda x: -x[0])
    keep = out[:k_chords]
    if all(n for _, n in keep):
        keep += [x for x in out if not x[1]][:1]
    return keep


@torch.no_grad()
def lm_scores(lm, histories, chords, device):
    """[beams, chords] log P(chord tokens | history), one forward pass; histories cut to a common length."""
    seqs = [chord_tokens(ch) for ch in chords]
    L = min(min(len(h) for h in histories), HISTORY)
    S = max(len(q) for q in seqs)
    x = torch.tensor([h[-L:] + q + [0] * (S - len(q)) for h in histories for q in seqs], device=device)
    lp = torch.log_softmax(lm(x).float(), -1)
    idx = torch.tensor([q + [0] * (S - len(q)) for q in seqs], device=device).repeat(len(histories), 1)
    msk = torch.tensor([[1.0] * len(q) + [0.0] * (S - len(q)) for q in seqs], device=device).repeat(len(histories), 1)
    g = lp[:, L - 1:L - 1 + S, :].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return (g * msk).sum(-1).view(len(histories), len(seqs)).cpu().numpy()


def decode_with_lm(tab, onset, lm, device, lam=0.3, beta=1.0, tau=0.2, beam=4, win=4, reattack=0.75):
    """Notes (as extract_notes) from the acoustic model's tab / onset probabilities and the tab LM."""
    beams = [(0.0, [1], [])]  # (score, LM history, chosen (string, fret, frame))
    for _, group in attack_groups(onset, tau, win):
        cands = candidates(tab, group)
        nonempty = [ch for _, ch in cands if ch]
        lms = lm_scores(lm, [b[1] for b in beams], nonempty, device) if lam and nonempty else np.zeros((len(beams), len(nonempty)))
        nxt = []
        for bi, (score, hist, chosen) in enumerate(beams):
            it = iter(lms[bi])
            for ac, ch in cands:
                lm_lp = next(it) if ch else 0.0
                nxt.append((score + ac + lam * lm_lp + beta * len(ch), hist + (chord_tokens(ch) if ch else []), chosen + list(ch)))
        beams = sorted(nxt, key=lambda x: -x[0])[:beam]
    # a note lasts while its string keeps sounding that fret, until the next attack on the string
    active, frets = 1 - tab[:, :, 0], tab[:, :, 1:].argmax(-1)
    starts = {}
    for s, f, t in beams[0][2]:
        starts.setdefault(STRINGS.index(s), []).append((t, f))
    out = []
    for c, lst in starts.items():
        lst.sort()
        for i, (t, f) in enumerate(lst):
            end = lst[i + 1][0] if i + 1 < len(lst) else len(tab)
            e = t + 1
            while e < end and active[e, c] > 0.5 and (frets[e, c] == f or e - t < 3):
                e += 1
            s = STRINGS[c]
            out.append({"onset": t / FPS, "offset": max(e, t + 2) / FPS, "string": s, "fret": f, "pitch": OPEN_PITCH[s] + f})
    # legato notes (no onset peak on their string) from the default extraction pass through
    pk = {c: set(peaks(onset[:, c], tau)) for c in range(6)}
    for n in extract_notes(tab, onset):
        c, t = STRINGS.index(n["string"]), int(round(n["onset"] * FPS))
        if not any(p in pk[c] for p in range(t - 2, t + 3)):
            out.append(n)
    out.sort(key=lambda n: (n["onset"], n["string"]))
    return merge_reattacks(out, onset, reattack) if reattack else out
