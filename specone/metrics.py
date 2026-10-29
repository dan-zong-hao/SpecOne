"""Metrics of a prediction file against the oracle utilities of a split.

    python specone/metrics.py --data DATA --split test_id --preds preds/X_test_id.jsonl [--json out.json]

Prediction lines: {id, scene, task, labels, probs} as written by score.py (or {id, scene, task, choice} for rules).
Reported (channel / mode / joint decision, chained):
  acc      the chosen option is in the oracle's optimal set (ties count as correct)
  regret   U* - U(chosen channel, chosen mode), in units of the normal rate (mute = 0)
  ece      15-bin expected calibration error of the top probability on the channel question
  feas     accuracy of the yes/no feasibility question; type: accuracy of the auxiliary jammer-type question
  pair     on pair splits: share of pairs whose two scenes both get the optimal joint action
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

MODES = ("normal", "boost", "spread4", "spread16", "mute")


def top(p):
    if "choice" in p:
        return p["choice"], None
    i = int(np.argmax(p["probs"]))
    return p["labels"][i], float(p["probs"][i])


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return 100.0 * e


def load_scenes(data, split):
    out = {}
    with open(os.path.join(data, f"scenes_{split}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            u = np.array([[(-np.inf if x is None else x) for x in row] for row in r["utility"]])
            r["U"] = u
            out[r["scene"]] = r
    return out


def decide(U, Pc, Pm):
    """Channel / mode / joint accuracy and regret of one decision against the utility table U (NCH x M)."""
    best = float(np.max(U))
    cu = U[:, :4].max(axis=1)
    opt_c = set(np.flatnonzero(cu >= cu.max() - 1e-6).tolist())
    lab, c = top(Pc)
    if c is None and "prob" in Pc:
        c = float(Pc["prob"])                                  # generative: probability of the answer tokens
    valid = isinstance(lab, str) and lab.startswith("ch") and lab[2:].isdigit()
    f = int(lab[2:]) - 1 if valid else None
    row = {"chan": f in opt_c if valid else False, "invalid": not valid, "conf": c, "unique": len(opt_c) == 1}
    if Pm is not None:
        mlab, _ = top(Pm)
        if valid and mlab in MODES:
            m = MODES.index(mlab)
            mu = U[f]
            row["mode"] = bool(mu[m] >= np.max(mu) - 1e-6)
            u = float(U[f, m]) if np.isfinite(U[f, m]) else -1.0
        else:                                                    # unparseable answer: counted as not transmitting
            row["mode"] = False
            row["invalid"] = True
            u = 0.0
        row["joint"] = bool(u >= best - 1e-6)
        row["regret"] = best - u
    return row


def evaluate(data, split, preds_path):
    scenes = load_scenes(data, split)
    preds = collections.defaultdict(dict)
    with open(preds_path) as f:
        for line in f:
            p = json.loads(line)
            preds[p["scene"]][p["task"]] = p
    per_scene = {}
    conf, corr, conf_u, corr_u = [], [], [], []
    for sid, rec in scenes.items():
        P = preds.get(sid, {})
        if "chan" not in P:
            continue
        d = decide(rec["U"], P["chan"], P.get("mode"))
        row = {k: d[k] for k in ("chan", "mode", "joint", "regret", "invalid") if k in d}
        row.update(type=rec["jam_type"], family=rec["spec"]["family"])
        if d["conf"] is not None:
            conf.append(d["conf"])
            corr.append(d["chan"])
            if d["unique"]:                                    # calibration where the optimum is unique
                conf_u.append(d["conf"])
                corr_u.append(d["chan"])
        if "feas" in P:
            flab, _ = top(P["feas"])
            row["feas"] = (flab == "yes") == bool(rec["feasible"])
        if "type" in P and rec["jam_type"]:
            tlab, _ = top(P["type"])
            row["type_acc"] = tlab == rec["jam_type"]
        per_scene[sid] = row
    rows = list(per_scene.values())

    def mean(key, subset=None):
        xs = [r[key] for r in (subset if subset is not None else rows) if key in r]
        return round(float(np.mean(xs)) * (1 if key == "regret" else 100), 2 if key != "regret" else 4) if xs else None

    out = {"split": split, "n": len(rows), "chan": mean("chan"), "mode": mean("mode"), "joint": mean("joint"),
           "regret": mean("regret"), "feas": mean("feas"), "type": mean("type_acc"),
           "ece_chan": round(ece(conf, corr), 2) if conf else None,
           "ece_chan_unique": round(ece(conf_u, corr_u), 2) if conf_u else None,
           "n_unique": len(conf_u),
           "invalid": round(100 * float(np.mean([r.get("invalid", False) for r in rows])), 2) if rows else None}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"] or r["family"]].append(r)
    out["by_type"] = {k: {"n": len(v), "joint": mean("joint", v), "regret": mean("regret", v)} for k, v in sorted(by.items())}
    pairs_file = os.path.join(data, f"pairs_{split}.jsonl")
    if os.path.exists(pairs_file):
        kinds = collections.defaultdict(list)
        with open(pairs_file) as f:
            for line in f:
                p = json.loads(line)
                a, b = per_scene.get(p["a"]), per_scene.get(p["b"])
                if a and b and "joint" in a and "joint" in b:
                    kinds[p["kind"]].append(a["joint"] and b["joint"])
        allp = [x for v in kinds.values() for x in v]
        out["pair"] = round(100 * float(np.mean(allp)), 2) if allp else None
        out["pair_by_kind"] = {k: round(100 * float(np.mean(v)), 1) for k, v in kinds.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = evaluate(a.data, a.split, a.preds)
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
