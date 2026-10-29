"""Certified gate for selective decisions (paper Sec. IV-D).

    python specone/gate.py --data DATA --preds preds/X_test_id.jsonl [--alpha 0.05 0.10] [--delta 0.05]
                           [--apply ood_combo=preds/X_ood_combo.jsonl ...] [--out gate.json]

Decision = the chained joint action (channel, then the mode asked on the model's own chosen channel). Confidence
kappa = p(channel answer) x p(mode answer | chosen channel). Correct = the joint action is in the oracle's optimal set
(ties count). The link acts when kappa >= g and defers otherwise.

g is set by selection with guaranteed risk (SGR; Geifman & El-Yaniv, NeurIPS 2017, Alg. 1): a binary search over the
sorted calibration confidences for the largest coverage whose one-sided Clopper-Pearson upper bound on the error of the
accepted joint actions is <= alpha, each test at delta / ceil(log2(m + 1)), so that the bound holds with probability
1 - delta. The split of --preds is halved by scene number (even = calibration, odd = evaluation; pair scenes by pair
number); the threshold fitted on the calibration half is applied unchanged to the evaluation half and to every
--apply split.
"""
from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np
from scipy.stats import beta

from metrics import MODES, load_scenes


def conf_of(p):
    i = int(np.argmax(p["probs"]))
    return p["labels"][i], float(p["probs"][i])


def joint_u(U, cl, ml):
    f = int(cl[2:]) - 1 if isinstance(cl, str) and cl.startswith("ch") and cl[2:].isdigit() else None
    if f is None or ml not in MODES or not 0 <= f < U.shape[0]:
        return 0.0                                     # unparseable answer: not transmitting (as in metrics.py)
    m = MODES.index(ml)
    return float(U[f, m]) if np.isfinite(U[f, m]) else -1.0   # an excluded action counts as -1 (as in metrics.py)


def scene_num(sid):
    """Scene number for the parity split; pair scenes (..._<k>_a / _b) get their pair number."""
    for tok in reversed(sid.split("_")):
        if tok.isdigit():
            return int(tok)
    return -1


def decisions(data, split, path):
    scenes = load_scenes(data, split)
    P = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            P[p["scene"]][p["task"]] = p
    num, kappa, correct = [], [], []
    for sid, rec in scenes.items():
        p = P.get(sid, {})
        if "chan" not in p or "mode" not in p:
            continue
        U = rec["U"]
        cl, cc = conf_of(p["chan"])
        ml, mc = conf_of(p["mode"])
        num.append(scene_num(sid))
        kappa.append(cc * mc)
        correct.append(joint_u(U, cl, ml) >= float(np.max(U)) - 1e-6)
    return {"num": np.array(num), "kappa": np.array(kappa), "correct": np.array(correct, bool)}


def cp_upper(e, n, delta):
    """One-sided Clopper-Pearson upper bound on a binomial rate (e errors in n)."""
    if n == 0 or e >= n:
        return 1.0
    return float(beta.ppf(1 - delta, e + 1, n - e))


def cp_two_sided(e, n, level=0.95):
    if n == 0:
        return [None, None]
    a = 1 - level
    lo = 0.0 if e == 0 else float(beta.ppf(a / 2, e, n - e + 1))
    hi = 1.0 if e == n else float(beta.ppf(1 - a / 2, e + 1, n - e))
    return [round(100 * lo, 2), round(100 * hi, 2)]


def stats_at(kc, wrong, theta):
    acc = kc >= theta
    return int(acc.sum()), int(wrong[acc].sum())


def sgr(kc, yc, alpha, delta):
    """Threshold theta (accept kappa >= theta) certified on the calibration decisions (kc, yc)."""
    ks = np.sort(kc)[::-1]
    m, wrong = len(ks), ~yc
    K = max(1, math.ceil(math.log2(m + 1)))
    lo, hi, theta, best_n, bound = 0, m - 1, math.inf, -1, None
    tests = 0
    while lo <= hi and tests < K:
        z = (lo + hi) // 2
        n, e = stats_at(kc, wrong, ks[z])
        b = cp_upper(e, n, delta / K)
        tests += 1
        if b <= alpha:
            if n > best_n:
                theta, best_n, bound = float(ks[z]), n, b
            lo = z + 1
        else:
            hi = z - 1
    return theta, bound


def report(D, theta):
    acc = D["kappa"] >= theta
    n, na = len(acc), int(acc.sum())
    e = int((~D["correct"][acc]).sum())
    return {"n": n, "coverage": round(100 * na / n, 2) if n else None, "accepted": na,
            "err": round(100 * e / na, 2) if na else None, "err_ci95": cp_two_sided(e, na)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True, help="score.py output on the split to calibrate on")
    ap.add_argument("--split", default="test_id", help="split of --preds")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--apply", nargs="*", default=[], help="split=preds.jsonl, gated with the calibrated threshold")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = decisions(a.data, a.split, a.preds)
    cal_mask = D["num"] % 2 == 0
    cal = {k: v[cal_mask] for k, v in D.items()}
    sets = {"eval": {k: v[~cal_mask] for k, v in D.items()}}
    for item in a.apply:
        s, path = item.split("=", 1)
        sets[s] = decisions(a.data, s, path)
    res = {"preds": a.preds, "delta": a.delta, "n_cal": int(cal_mask.sum()), "gates": {}}
    for alpha in a.alpha:
        theta, bound = sgr(cal["kappa"], cal["correct"], alpha, a.delta)
        g = res["gates"][str(alpha)] = {"theta": theta, "cal": report(cal, theta),
                                        "cal_bound": round(100 * bound, 2) if bound is not None else None}
        for s, Ds in sets.items():
            g[s] = report(Ds, theta)
        print(f"alpha={alpha:g} theta={theta:.4f} " + " ".join(
            f"{s}: coverage {g[s]['coverage']}% err {g[s]['err']}%" for s in sets))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
