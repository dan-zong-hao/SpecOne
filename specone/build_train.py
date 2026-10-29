"""Training files of SpecOne (paper Sec. IV-B and IV-C) from the JamPivot rows.

    python specone/build_train.py DATA [--out DIR]

Reads DATA/rows_train.jsonl and DATA/rows_pair_train.jsonl and writes (into DATA unless --out is given)
  train_softpairs.jsonl  the SpecOne training set: 39,400 scenes, of which 15,994 are both scenes of the 7,997
                         training contrast pairs (they replace as many random train scenes, so the size is unchanged),
                         utility-soft targets (tau = 0.05)
  dev.jsonl              the last 600 train scenes (never trained on), one-hot targets, for monitoring only
With --ablations also train_soft.jsonl (no pairs), train_onehot.jsonl / train_onehotpairs.jsonl (one-hot targets) and,
with --tau T, train_softpairs_t<T>.jsonl (targets recomputed at another temperature from DATA/scenes_*.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import random

from targets import CHAN_KEYS, MODES, soft_target, to_dist, utility_table


def load(root, name):
    with open(os.path.join(root, f"rows_{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def clean(row, soft):
    r = dict(row)
    if soft:
        r["target"] = r["target_soft"]
    r.pop("target_soft", None)
    return r


def write(root, name, rows, soft):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, name), "w") as f:
        for r in rows:
            f.write(json.dumps(clean(r, soft), ensure_ascii=False, separators=(",", ":")) + "\n")
    print(name, len(rows), "rows", len({r["meta"]["scene"] for r in rows}), "scenes")


def retemper(root, rows, tau):
    """Utility-soft targets at another temperature, recomputed from the oracle utilities of the scene records."""
    util = {}
    for split in ("train", "pair_train"):
        with open(os.path.join(root, f"scenes_{split}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                util[r["scene"]] = utility_table(r)
    out = []
    for r in rows:
        U = util[r["meta"]["scene"]]
        r = dict(r)
        if r["task"] == "chan":
            r["target_soft"] = to_dist(CHAN_KEYS, soft_target(U[:, :4].max(axis=1), tau))
        elif r["task"] == "mode":
            r["target_soft"] = to_dist(MODES, soft_target(U[CHAN_KEYS.index(r["meta"]["chosen"])], tau))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None, help="output folder (default: DATA)")
    a = ap.parse_args()
    root, out = a.data, a.out or a.data
    train = load(root, "train")
    dev_ids = {f"train_{i:06d}" for i in range(39400, 40000)}
    tr = [r for r in train if r["meta"]["scene"] not in dev_ids]
    dev = [r for r in train if r["meta"]["scene"] in dev_ids]
    write(out, "dev.jsonl", dev, soft=False)
    pairs = load(root, "pair_train")
    n_pair_scenes = len({r["meta"]["scene"] for r in pairs})
    scenes = sorted({r["meta"]["scene"] for r in tr})
    keep = set(random.Random(0).sample(scenes, len(scenes) - n_pair_scenes))
    mixed = [r for r in tr if r["meta"]["scene"] in keep] + pairs
    random.Random(1).shuffle(mixed)
    write(out, "train_softpairs.jsonl", mixed, soft=True)
    if a.ablations:
        write(out, "train_soft.jsonl", tr, soft=True)
        write(out, "train_onehot.jsonl", tr, soft=False)
        write(out, "train_onehotpairs.jsonl", mixed, soft=False)
    if a.tau is not None:
        write(out, f"train_softpairs_t{a.tau:g}.jsonl".replace("0.", ""), retemper(root, mixed, a.tau), soft=True)


if __name__ == "__main__":
    main()
