"""Bộ giải Greedy / MILP dùng chung cho router (bản sao đúng logic của 07_toi_uu.py; khu vực và tham số đọc từ decision_bundle)."""
import numpy as np
import joblib
from scipy.optimize import milp, LinearConstraint, Bounds
from config import *

_db = joblib.load(DATA / "models" / "decision_bundle.joblib")
zone, PARAMS = _db["zone"], _db["params"]
N_ZONES, n = int(zone.max()) + 1, len(zone)
Zmat = np.zeros((n, N_ZONES)); Zmat[np.arange(n), zone] = 1
S_MIN, F_MIN, B_MIN, K_DEF, B_FULL = PARAMS["S"], PARAMS["F"], PARAMS["B"], PARAMS["K"], PARAMS["b_full"]


def cost_min(z, s=S_MIN, F=F_MIN):
    return s * z.sum() + F * (Zmat[z.astype(bool)].sum(0) > 0).sum()


def solve_milp(v, K=K_DEF, B=B_MIN, F=F_MIN, s=S_MIN, gap=0.0):
    nv = n + N_ZONES
    c = np.concatenate([-v, np.zeros(N_ZONES)])
    A = np.array([np.concatenate([np.ones(n), np.zeros(N_ZONES)]), np.concatenate([np.full(n, s), np.full(N_ZONES, F)])])
    link = np.hstack([np.eye(n), -Zmat])
    cons = [LinearConstraint(A, -np.inf, [K, B]), LinearConstraint(link, -np.inf, 0)]
    r = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1), options={"mip_rel_gap": gap})
    z = np.zeros(n, bool)
    if r.x is not None: z = r.x[:n] > 0.5
    return z & (v > 0)


def solve_greedy(v, K=K_DEF, B=B_MIN, F=F_MIN, s=S_MIN):
    z = np.zeros(n, bool); used = np.zeros(N_ZONES, bool); mins = 0
    for i in np.argsort(-v):
        if v[i] <= 0 or z.sum() >= K: break
        add = s + (0 if used[zone[i]] else F)
        if mins + add <= B:
            z[i] = True; used[zone[i]] = True; mins += add
    return z
