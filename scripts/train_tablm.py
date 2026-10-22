"""Build the chord corpus and train the tab language model used by transcribe.py --music_lm (guitab/tablm.py).

    python scripts/train_tablm.py corpus    # tabs -> data/tablm/{train,val}.npy (token arrays)
    python scripts/train_tablm.py train     # -> runs/tablm/tablm.pt (~40 min on the laptop, 8k steps)

Corpus: 6-string standard-tuning acoustic nylon / steel, clean and jazz guitar tracks of the DadaGP-derived SynthTab label
archive (data/synthtab/all_jams_midi.zip; distortion, overdrive and bass tracks are left out). Notes starting at the same
tick form a chord; dead notes and frets > 19 are skipped. The user's benchmark song and the custom-dataset validation titles are
excluded so the LM cannot have memorised a test tab. 2% of tracks are held out (perplexity).
"""

import argparse
import csv
import json
import math
import re
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guitab.device import DEVICES, pick_device  # noqa: E402
from guitab.paths import DATA, TABLM_RUN  # noqa: E402
from guitab.custom_data import SELECTION  # noqa: E402
from guitab.synthtab import LABEL_ZIP  # noqa: E402
from guitab.tablm import CTX, VOCAB, TabLM, tok  # noqa: E402

CORPUS = DATA / "tablm"
STANDARD = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
KEEP = ("Acoustic Nylon Guitar", "Acoustic Steel Guitar", "Electric Clean Guitar", "Electric Jazz Guitar")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def excluded_titles():
    titles = ["remember me"]
    for r in csv.DictReader(open(SELECTION)) if SELECTION.exists() else []:
        if r["split"] == "ss_validation":
            titles.append(re.sub(r"\(.*?\)", "", r["title"]).strip())
    return [norm(t) for t in titles if len(norm(t)) >= 5]


def _init():
    global ZF
    ZF = zipfile.ZipFile(LABEL_ZIP)


def encode(name):
    """Token list of one track (BOS, then chord tokens), or None if not a 6-string standard-tuning track."""
    j = json.loads(ZF.read(name))
    tabs = [a for a in j["annotations"] if a["namespace"] == "note_tab"]
    if len(tabs) != 6 or any(STANDARD.get(a["sandbox"]["string_index"]) != a["sandbox"]["open_tuning"] for a in tabs):
        return None
    chords = {}
    for a in tabs:
        s = a["sandbox"]["string_index"]
        for x in a["data"]:
            v = x["value"]
            if not v.get("dead_note") and 0 <= v["fret"] <= 19:
                chords.setdefault(x["time"], {})[s] = v["fret"]
    toks = [1]
    for t in sorted(chords):
        toks += [tok(s, f) for s, f in sorted(chords[t].items(), reverse=True)] + [2]
    return toks if len(toks) > 50 else None


def cmd_corpus(_):
    CORPUS.mkdir(parents=True, exist_ok=True)
    ex = excluded_titles()
    names = [n for n in zipfile.ZipFile(LABEL_ZIP).namelist() if n.endswith(".jams")]
    keep = [n for n in names if n.split("/")[1].split(" - ")[-1].split("__")[0] in KEEP
            and not any(e in norm(n.split("/")[1]) for e in ex)]
    with Pool(4, initializer=_init) as p:
        seqs = [s for s in p.map(encode, keep, chunksize=64) if s]
    idx = np.random.default_rng(0).permutation(len(seqs))
    nval = max(1, len(seqs) // 50)
    for split, ids in (("val", idx[:nval]), ("train", idx[nval:])):
        arr = np.concatenate([np.array(seqs[i], dtype=np.int16) for i in ids])
        np.save(CORPUS / f"{split}.npy", arr)
        print(f"{split}: {len(ids)} tracks, {len(arr)} tokens")


def batches(arr, bs, rng):
    i = rng.integers(0, len(arr) - CTX - 1, bs)
    x = np.stack([arr[j:j + CTX + 1] for j in i]).astype(np.int64)
    return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


def cmd_train(args):
    tr, va = np.load(CORPUS / "train.npy"), np.load(CORPUS / "val.npy")
    dev = pick_device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TabLM().to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    TABLM_RUN.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1, step / 300) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        x, y = batches(tr, 64, rng)
        loss = F.cross_entropy(model(x.to(dev)).reshape(-1, VOCAB), y.to(dev).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            vr = np.random.default_rng(1)
            with torch.no_grad():
                vl = np.mean([F.cross_entropy(model(a.to(dev)).reshape(-1, VOCAB), b.to(dev).reshape(-1)).item()
                              for a, b in (batches(va, 64, vr) for _ in range(20))])
            model.train()
            print(f"step {step} train {loss.item():.3f} val {vl:.3f} (perplexity {math.exp(vl):.1f}) "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            torch.save(model.state_dict(), TABLM_RUN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=DEVICES, help="auto = cuda, else mps, else cpu")
    parser.add_argument("step", choices=["corpus", "train"])
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    {"corpus": cmd_corpus, "train": cmd_train}[args.step](args)


if __name__ == "__main__":
    main()
