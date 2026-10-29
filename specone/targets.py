"""Utility-soft targets (paper Sec. IV-B).

    pi(o) ∝ exp(-(U* - U(o)) / tau),   pi(o) = 0 for excluded actions,

where U(o) is the expected utility of option o (for the channel question, that of the best transmitting mode on the
channel), U* the best of them, and tau = 0.05 of the normal rate. tau -> 0 gives the uniform distribution over the
optimal set (one-hot when the optimum is unique), which is the `target` field of every row; `target_soft` holds the
tau = 0.05 distribution.

A scene's utility table U has one row per channel (ch1 ... ch9) and one column per mode (MODES); excluded actions
are stored as null in the scene records and read as -inf here.
"""
from __future__ import annotations

import numpy as np

MODES = ("normal", "boost", "spread4", "spread16", "mute")
CHAN_KEYS = [f"ch{i}" for i in range(1, 10)]
TAU = 0.05


def utility_table(scene_record: dict) -> np.ndarray:
    """U (9 x 5) of a scene record; null (excluded action) -> -inf."""
    return np.array([[(-np.inf if x is None else x) for x in row] for row in scene_record["utility"]])


def optimal_set(values: np.ndarray, tol: float = 1e-6) -> list[int]:
    v = np.where(np.isfinite(values), values, -np.inf)
    best = v.max()
    return [int(i) for i in np.flatnonzero(v >= best - tol)]


def soft_target(values: np.ndarray, tau: float) -> np.ndarray:
    """q(a) ∝ exp(-(U* - U(a)) / tau); not allowed -> 0. tau = 0 gives the uniform distribution over the optimal set."""
    v = np.where(np.isfinite(values), values, -np.inf)
    if tau <= 0:
        q = np.zeros_like(v)
        q[optimal_set(v)] = 1.0
    else:
        q = np.exp((v - v.max()) / tau)
        q[~np.isfinite(v)] = 0.0
    return q / q.sum()


def to_dist(keys, q) -> dict:
    """Sparse target as stored in the rows: option key -> probability, entries <= 1e-6 dropped."""
    return {k: round(float(v), 6) for k, v in zip(keys, q) if v > 1e-6}


def channel_values(U: np.ndarray) -> np.ndarray:
    """Value of each channel for the channel question: its best transmitting mode (mute excluded)."""
    return U[:, :4].max(axis=1)


def chan_target(U: np.ndarray, tau: float = TAU) -> dict:
    return to_dist(CHAN_KEYS, soft_target(channel_values(U), tau))


def mode_target(U: np.ndarray, chosen: str, tau: float = TAU) -> dict:
    """Target of the mode question asked on channel `chosen` (e.g. "ch6")."""
    return to_dist(MODES, soft_target(U[CHAN_KEYS.index(chosen)], tau))


def training_channel(U: np.ndarray, f_prev: int) -> str:
    """Channel written into a training mode row (teacher forcing): the current channel when it is optimal, else the
    first optimal channel. At test time the mode question is asked on the model's own choice (score.py)."""
    opt = optimal_set(channel_values(U))
    return CHAN_KEYS[f_prev if f_prev in opt else opt[0]]
