"""Per-decision latency of a System-One model (HF transformers, one request at a time): channel question, then the
mode question with the chosen channel, as in the chained evaluation. Reports p50 / p95 / mean in ms.

    PYTHONPATH=$ONEJEV python specone/latency.py --model MODEL --data DATA --n 200 --out lat.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from train import common, mm
    from train.sft import collect_mm

    mm.set_root(a.data)
    bundle = common.load_bundle(a.model, device="cuda:0", dtype="bfloat16", multimodal=True)
    bundle.model.eval()
    by = {}
    with open(os.path.join(a.data, "rows_test_id.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r["meta"]["scene"], {})[r["task"]] = r
    scenes = list(by)[: a.n + 10]

    def one(row):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = collect_mm(bundle, [row], 4096, 8192, 1, "cuda:0")
        torch.cuda.synchronize()
        return s[0]

    times = []
    for i, sid in enumerate(scenes):
        t = time.perf_counter()
        s = one(by[sid]["chan"])
        lab = s.labels[max(range(len(s.logits)), key=s.logits.__getitem__)]
        m = json.loads(json.dumps(by[sid]["mode"]))
        m["state"]["chosen channel"] = f"channel {lab[2:]} ({2 * int(lab[2:])} MHz)"
        one(m)
        if i >= 10:                                   # the first ten warm up kernels
            times.append(1000 * (time.perf_counter() - t))
    times.sort()
    out = {"model": a.model, "n": len(times), "p50_ms": times[len(times) // 2], "p95_ms": times[int(0.95 * len(times))],
           "mean_ms": sum(times) / len(times), "gpu": torch.cuda.get_device_name(0)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
