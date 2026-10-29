"""Chained scoring with a SpecOne (System-One) model (paper Sec. IV-A).

    PYTHONPATH=$ONEJEV python specone/score.py --model MODEL --data DATA --split test_id --out preds/X_test_id.jsonl

The probability of each option is the softmax of the logits of the option letters at the last prompt position, so one
forward pass scores all options of a question and the answer is always one of them (OneJev's collect_mm). The
questions are chained: the channel rows are scored first, then each mode row is re-asked with the model's own chosen
channel written into its state ("chosen channel"), so the mode answer is conditioned on the model's decision.
Writes one JSON line per row: id, scene, task, labels, probs, and for mode rows the channel used (chosen_used).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time


def softmax(xs):
    m = max(xs)
    w = [math.exp(x - m) for x in xs]
    s = sum(w)
    return [x / s for x in w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only")
    ap.add_argument("--tasks", default="chan,mode,feas,type")
    ap.add_argument("--no-chain", action="store_true", help="score mode rows on their stored (teacher) channel")
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch
    from train import common, mm
    from train.sft import collect_mm

    mm.set_root(a.data)
    bundle = common.load_bundle(a.model, device=a.device, dtype="bfloat16", multimodal=True)
    bundle.model.eval()
    with open(os.path.join(a.data, f"rows_{a.split}.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    tasks = set(a.tasks.split(","))
    rows = [r for r in rows if r["task"] in tasks]
    if a.limit:
        seen = []
        for r in rows:
            if r["meta"]["scene"] not in seen:
                seen.append(r["meta"]["scene"])
        keep = set(seen[:a.limit])
        rows = [r for r in rows if r["meta"]["scene"] in keep]

    def run(batch):
        t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scored = collect_mm(bundle, batch, 4096, a.batch_tokens, 64, a.device)
        out = [None] * len(batch)
        for s in scored:
            out[s.idx] = {"labels": list(s.labels), "probs": softmax(s.logits)}
        return out, time.time() - t

    first = [r for r in rows if r["task"] != "mode" or a.no_chain]
    res, sec = run(first)
    preds = {}
    for r, s in zip(first, res):
        preds[r["id"]] = s
    mode_rows = [r for r in rows if r["task"] == "mode" and not a.no_chain]
    if mode_rows:
        chained = []
        for r in mode_rows:
            p = preds.get(r["id"].replace("-mode", "-chan"))
            r = json.loads(json.dumps(r))
            if p is not None:
                lab = p["labels"][max(range(len(p["probs"])), key=p["probs"].__getitem__)]
                digit = lab[2:]
                r["state"]["chosen channel"] = f"channel {digit} ({2 * int(digit)} MHz)"
                r["meta"]["chosen_used"] = lab
            chained.append(r)
        res2, sec2 = run(chained)
        sec += sec2
        for r, s in zip(chained, res2):
            if s is not None:
                s["chosen_used"] = r["meta"].get("chosen_used")
            preds[r["id"]] = s
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_none = 0
    with open(a.out, "w") as f:
        for r in rows:
            s = preds.get(r["id"])
            if s is None:
                n_none += 1
                continue
            f.write(json.dumps({"id": r["id"], "scene": r["meta"]["scene"], "task": r["task"], **s}) + "\n")
    print(json.dumps({"model": a.model, "split": a.split, "rows": len(rows), "skipped": n_none,
                      "seconds": round(sec, 1), "rows_per_s": round(len(rows) / max(sec, 1e-9), 1)}))


if __name__ == "__main__":
    main()
