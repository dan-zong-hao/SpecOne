"""Selective decisions (paper Sec. IV-D): risk-coverage curves and AURC on the joint decision.

    python specone/selective.py --data DATA --split test_id --preds name1=preds/a.jsonl name2=preds/b.jsonl --out sel.json
Confidence of a decision = kappa = top channel probability x top mode probability (the gate score of paper
Sec. IV-D). At coverage c the c most confident decisions are taken; risk = share of them that are not optimal.
Also the mean utility when deferred slots fall back to muting (U = 0).
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from metrics import MODES, load_scenes


def conf_of(p):
    if "probs" in p:
        i = int(np.argmax(p["probs"]))
        return p["labels"][i], float(p["probs"][i])
    return p.get("choice"), float(p.get("prob", 1.0))


def decisions(data, split, path):
    scenes = load_scenes(data, split)
    preds = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            preds[p["scene"]][p["task"]] = p
    out = []
    for sid, rec in scenes.items():
        P = preds.get(sid, {})
        if "chan" not in P or "mode" not in P:
            continue
        U = rec["U"]
        best = float(np.max(U))
        cl, cc = conf_of(P["chan"])
        ml, mc = conf_of(P["mode"])
        if not (isinstance(cl, str) and cl.startswith("ch") and ml in MODES):
            u = 0.0
        else:
            f, m = int(cl[2:]) - 1, MODES.index(ml)
            u = float(U[f, m]) if np.isfinite(U[f, m]) else -1.0
        out.append({"conf": cc * mc, "correct": u >= best - 1e-6, "u": u, "best": best})
    return out


def curve(dec, points=20):
    d = sorted(dec, key=lambda x: -x["conf"])
    n = len(d)
    cov, risk, util = [], [], []
    corr = np.array([x["correct"] for x in d], float)
    u = np.array([x["u"] for x in d])
    cum = np.cumsum(1 - corr)
    aurc = float(np.mean(cum / np.arange(1, n + 1)))
    for c in np.linspace(1.0 / points, 1.0, points):
        k = max(1, int(round(c * n)))
        cov.append(round(float(c), 3))
        risk.append(round(float(cum[k - 1] / k), 4))
        util.append(round(float(u[:k].sum() / n), 4))      # deferred slots mute (utility 0)
    return {"n": n, "aurc": round(aurc, 4), "coverage": cov, "risk": risk, "utility_mute_fallback": util,
            "utility_all": round(float(u.mean()), 4), "oracle_utility": round(float(np.mean([x["best"] for x in d])), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for item in a.preds:
        name, path = item.split("=", 1)
        res[name] = curve(decisions(a.data, a.split, path))
        print(name, "AURC", res[name]["aurc"], "risk@80%", res[name]["risk"][15], "risk@50%", res[name]["risk"][9])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
