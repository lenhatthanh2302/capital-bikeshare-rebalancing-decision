"""BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare. Toàn bộ quy trình trong MỘT file.

Quy trình: dữ liệu -> EDA/chọn trạm -> đặc trưng + nhãn -> dự báo + hiệu chỉnh -> PFI/SHAP -> nhân quả (pilot)
           -> tối ưu (SPO+/MILP/Greedy) -> Action Router -> ĐÓNG BĂNG -> locked 1 lần + OOT.

Cách chạy (trong thư mục Code):
    py -3 pipeline.py 1 2 3 4 5 6 7 8     # các bước phát triển (có thể chạy từng bước: py -3 pipeline.py 4)
    py -3 pipeline.py dry                 # diễn tập bước 10 trên policy_val (không mở locked)
    py -3 pipeline.py freeze              # bước 9: công bố tiêu chí chấp nhận + băm SHA-256
    py -3 pipeline.py real                # bước 10: mở locked + OOT, CHỈ MỘT LẦN
Khi Results/locked_access.lock đã tồn tại, mọi bước ghi đè kết quả đã đóng băng (1–9) bị từ chối.
Mỗi bước là một hàm buoc_NN_*; dấu #%% giữ nguyên để chạy từng cell trong Spyder/VS Code.
"""
import sys

import pandas as pd
import numpy as np
import joblib
import duckdb
import time
import hashlib
import os
import tempfile
import warnings
import shap
import platform
import sklearn
import scipy
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy import stats
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



#%% ======================================================================
# 0. CẤU HÌNH DÙNG CHUNG
# ========================================================================
SEED = 42
CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
DATA, RESULTS = BASE / "Data", BASE / "Results"

N_STATIONS = 80
K_PER_SHIFT = 8                       # số trạm xe tải ghé được mỗi ca (giả định, có sensitivity ở Bước 6)
SHIFTS = {"sang": (6, 10), "chieu": (16, 20)}   # [giờ bắt đầu, giờ kết thúc)
TAU_QUANTILE = 0.90                   # ngưỡng sự kiện = phân vị 90 theo (trạm, ca), học trên train

# 5 vai trò dữ liệu + out-of-time (ngày D của đơn vị phân tích, gồm cả hai đầu)
BLOCKS = {
    "train":       ("2024-01-01", "2025-10-31"),
    "validation":  ("2025-11-01", "2025-12-31"),
    "calibration": ("2026-01-01", "2026-02-28"),
    "policy_val":  ("2026-03-01", "2026-04-30"),
    "locked":      ("2026-05-01", "2026-06-30"),
    "oot":         ("2026-07-01", "2026-09-30"),
}
DEV_END = BLOCKS["policy_val"][1]     # Bước 3–5 chỉ dựng dữ liệu đến hết policy_val; locked/oot mở ở bước đóng băng

# Ngày lễ liên bang Mỹ rơi vào ngày thường (lịch công bố trước nên biết được tại t0)
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
])


def block_of(d: pd.Series) -> pd.Series:
    out = pd.Series("khac", index=d.index, dtype=object)
    for k, (a, b) in BLOCKS.items():
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = k
    return out



#%% ======================================================================
# THƯ VIỆN DÙNG CHUNG (bộ giải, đặc trưng, router, đánh giá) – các bước 8, 9, 10 gọi lại
# ========================================================================

# --- bộ giải Greedy / MILP (bản sao đúng logic của bước 7) ---
S_MIN, F_MIN, B_MIN, K_DEF, B_FULL = 10, 20, 140, K_PER_SHIFT, 0.7     # = tham số lưu trong decision_bundle (bước 7 ghi, nap_bo_giai kiểm tra)
zone = PARAMS = Zmat = N_ZONES = n = None


def nap_bo_giai():
    """Nạp khu vực của 80 trạm từ decision_bundle (có sau bước 7)."""
    global zone, PARAMS, N_ZONES, n, Zmat
    _db = joblib.load(DATA / "models" / "decision_bundle.joblib")
    zone, PARAMS = _db["zone"], _db["params"]
    N_ZONES, n = int(zone.max()) + 1, len(zone)
    Zmat = np.zeros((n, N_ZONES)); Zmat[np.arange(n), zone] = 1
    assert (PARAMS["S"], PARAMS["F"], PARAMS["B"], PARAMS["K"], PARAMS["b_full"]) == (S_MIN, F_MIN, B_MIN, K_DEF, B_FULL)


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


# --- dựng đặc trưng đến một ngày bất kỳ (bản sao đúng logic của bước 3; replay test ở bước 10 kiểm tính đúng đắn) ---
"""Dựng bảng đặc trưng đến một ngày bất kỳ (bản sao đúng logic của 03_dac_trung.py, dùng τ đã đóng băng).
Tính đúng đắn của bản sao được kiểm bằng replay test ở 10_locked_oot.py: các dòng dev dựng lại phải trùng features_dev.parquet."""

LAGS, WINS, VARS, WARMUP_DAYS = [1, 2, 3, 4, 5], [5, 10, 20], ["net_out", "net_in", "n_out", "n_in"], 20
sel_fe = stations_fe = None


def nap_bo_dac_trung():
    global sel_fe, stations_fe
    sel_fe = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str})
    stations_fe = sel_fe.station_id.tolist()


def load_grid(end_date):
    """Lưới (trạm, ngày thường, ca) và tổng chuyến toàn hệ thống, chỉ đọc dữ liệu đến end_date."""
    sid = "(" + ",".join("'" + s + "'" for s in stations_fe) + ")"
    con = duckdb.connect(); SH = (DATA / "station_hour.parquet").as_posix()
    case_shift = " ".join(f"WHEN hour(hour_ts) >= {a} AND hour(hour_ts) < {b} THEN '{s}'" for s, (a, b) in SHIFTS.items())
    agg = con.execute(f"""
    SELECT station_id, CAST(hour_ts AS DATE) AS ngay, CASE {case_shift} END AS ca, sum(n_out) AS n_out, sum(n_in) AS n_in
    FROM read_parquet('{SH}') WHERE station_id IN {sid} AND isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{end_date}'
    GROUP BY 1, 2, 3 HAVING ca IS NOT NULL""").df()
    agg["ngay"] = pd.to_datetime(agg["ngay"])
    days = pd.bdate_range(BLOCKS["train"][0], end_date)
    idx = pd.MultiIndex.from_product([stations_fe, days, list(SHIFTS)], names=["station_id", "ngay", "ca"])
    base = agg.set_index(["station_id", "ngay", "ca"]).reindex(idx, fill_value=0).reset_index()
    base[["n_out", "n_in"]] = base[["n_out", "n_in"]].astype(int)
    base["net_out"] = base.n_out - base.n_in
    base["net_in"] = -base.net_out
    city = con.execute(f"""SELECT CAST(hour_ts AS DATE) AS ngay, sum(n_out) AS city_out FROM read_parquet('{SH}')
    WHERE isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{end_date}' GROUP BY 1 ORDER BY 1""").df()
    city["ngay"] = pd.to_datetime(city["ngay"])
    city = city.set_index("ngay").reindex(days, fill_value=0)
    return base, city, days


def add_tau_labels(base):
    tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})     # τ đóng băng, học trên train ở bước 3
    base = base.merge(tau, on=["station_id", "ca"], how="left")
    base["Y_short"] = ((base.net_out >= base.tau_out) & (base.net_out > 0)).astype(int)
    base["Y_full"] = ((base.net_in >= base.tau_in) & (base.net_in > 0)).astype(int)
    base["khoi"] = block_of(base.ngay)
    return base


def build_features(b, c):
    b = b.sort_values(["station_id", "ca", "ngay"]).reset_index(drop=True)
    f = b[["station_id", "ngay", "ca"]].copy()
    g = b.groupby(["station_id", "ca"], sort=False)
    for v in VARS:
        for k in LAGS:
            f[f"{v}_lag{k}"] = g[v].shift(k)
        s1 = g[v].shift(1)
        for w in WINS:
            f[f"{v}_m{w}"] = s1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(w, min_periods=w).mean())
        f[f"{v}_m60"] = s1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(60, min_periods=20).mean())
    for v, t in (("net_out", "tau_out"), ("net_in", "tau_in")):
        den = b[t].clip(lower=1)
        for k in (1, 2, 5):
            f[f"{v}_lag{k}_over_tau"] = f[f"{v}_lag{k}"] / den
        for w in WINS + [60]:
            f[f"{v}_m{w}_over_tau"] = f[f"{v}_m{w}"] / den
        f[t] = b[t]
    for v, t, nm in (("net_out", "tau_out", "ev_short"), ("net_in", "tau_in", "ev_full")):
        ev = ((b[v] >= b[t]) & (b[v] > 0)).astype(float)
        e1 = ev.groupby([b.station_id, b.ca], sort=False).shift(1)
        f[f"{nm}_lag1"] = e1
        for w in (10, 20, 60):
            f[f"{nm}_r{w}"] = e1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(w, min_periods=5).mean())
    opp = b[["station_id", "ngay", "ca", "net_out", "n_out", "n_in"]].copy()
    opp["ca"] = opp.ca.map({"sang": "chieu", "chieu": "sang"})
    nxt = {d: n for d, n in zip(c.index[:-1], c.index[1:])}
    opp["ngay"] = opp.ngay.map(nxt)
    opp = opp.dropna(subset=["ngay"]).rename(columns={"net_out": "opp_net_out_lag1", "n_out": "opp_n_out_lag1", "n_in": "opp_n_in_lag1"})
    f = f.merge(opp, on=["station_id", "ngay", "ca"], how="left")
    cs = c.city_out.shift(1)
    cf = pd.DataFrame({"city_out_lag1": cs, "city_out_m20": cs.rolling(20, min_periods=20).mean()})
    cf["city_ratio"] = cf.city_out_lag1 / cf.city_out_m20
    f = f.merge(cf.reset_index().rename(columns={"index": "ngay"}), on="ngay", how="left")
    f["dow"] = f.ngay.dt.dayofweek
    f["thang"] = f.ngay.dt.month
    f["thang_sin"] = np.sin(2 * np.pi * f.thang / 12)
    f["thang_cos"] = np.cos(2 * np.pi * f.thang / 12)
    f["la_ca_sang"] = (f.ca == "sang").astype(int)
    f["la_ngay_le"] = f.ngay.isin(HOLIDAYS).astype(int)
    prev = {n: d for d, n in nxt.items()}
    f["hom_qua_la_le"] = f.ngay.map(prev).isin(HOLIDAYS).astype(int)
    f = f.merge(sel_fe.set_index("station_id")[["lat", "lng"]], left_on="station_id", right_index=True, how="left")
    return f


def build_dataset(end_date, features):
    """Bảng (đặc trưng + nhãn + khối) đến end_date, đã bỏ 20 ngày khởi động; trả thêm (base, city) để chạy thí nghiệm rò rỉ."""
    base, city, days = load_grid(end_date)
    base = add_tau_labels(base)
    feat = build_features(base, city)
    lbl = base[["station_id", "ngay", "ca", "net_out", "net_in", "Y_short", "Y_full", "khoi"]]
    df = feat.merge(lbl, on=["station_id", "ngay", "ca"], how="left")
    df = df[df.ngay >= days[WARMUP_DAYS]].dropna(subset=features)
    return df.reset_index(drop=True), base, city


def leakage_check(df, base, city, features, days_to_test):
    """Ghi đè mọi kết quả từ ngày D trở đi bằng 99999, dựng lại, so với bảng gốc: chênh lệch phải bằng 0."""
    rows = []
    for d in days_to_test:
        d = pd.Timestamp(d)
        b2, c2 = base.copy(), city.copy()
        for v in VARS: b2.loc[b2.ngay >= d, v] = 99999
        c2.loc[c2.index >= d, "city_out"] = 99999
        f2 = build_features(b2, c2)
        a = f2[f2.ngay == d].set_index(["station_id", "ca"])[features].sort_index()
        o = df[df.ngay == d].set_index(["station_id", "ca"])[features].sort_index()
        diff = float(np.nanmax(np.abs(a.values.astype(float) - o.values.astype(float))))
        rows.append({"ngay_kiem_tra": d.date(), "khoi": df[df.ngay == d].khoi.iloc[0], "chenh_lech_lon_nhat": diff,
                     "ket_qua": "Đạt" if diff == 0 else "KHÔNG ĐẠT"})
    return pd.DataFrame(rows)


# --- router (bản sao đúng logic của bước 8; replay test so với đầu ra của bước 8) ---
POLICY_ID, CONTROL_SALT, CONTROL_PCT = "bike-rebalance-v1", "ctl-v1", 10
V_MIN, MAX_VISITS_5D, PSI_WARN, PSI_STOP = 0.10, 3, 0.10, 0.25
CA_ORDER = {"sang": 0, "chieu": 1}
sel = stations = tau = None
N = TAU_OUT_MED = TAU_IN_MED = MODEL_VERSION = None


def nap_bo_router():
    """Nạp danh sách trạm, ngưỡng τ đóng băng và phiên bản mô hình (cần đã có bước 2, 3, 4, 7)."""
    global sel, stations, N, tau, TAU_OUT_MED, TAU_IN_MED, MODEL_VERSION
    sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str}).sort_values("station_id").reset_index(drop=True)
    stations = sel.station_id.tolist(); N = len(stations)
    tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})
    TAU_OUT_MED, TAU_IN_MED = tau.tau_out.median(), tau.tau_in.median()
    MODEL_VERSION = file_hash(DATA / "models" / "forecast_bundle.joblib") + "+" + file_hash(DATA / "models" / "decision_bundle.joblib")


LAGC = [f"n_out_lag{k}" for k in range(1, 6)] + [f"n_in_lag{k}" for k in range(1, 6)]


def file_hash(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def prep_cells(pred, feat):
    """pred: xác suất + nhãn theo (trạm, ngày, ca); feat: bảng đặc trưng cùng khoá. Trả về bảng đã sắp theo (ngày, ca, trạm)."""
    ft = feat[["station_id", "ngay", "ca", "net_out_lag5", "net_in_lag5"] + LAGC]
    d = pred.merge(ft, on=["station_id", "ngay", "ca"], how="left")
    d["ca_ord"] = d.ca.map(CA_ORDER)
    d = d.sort_values(["ngay", "ca_ord", "station_id"]).reset_index(drop=True)
    assert (d.groupby(["ngay", "ca"]).size() == N).all()
    d["hop_le"] = d[LAGC].sum(axis=1) > 0
    d["v"] = d.p_Y_short_final + B_FULL * d.p_Y_full_final
    d["v_du_phong"] = d.net_out_lag5.clip(lower=0) / TAU_OUT_MED + B_FULL * d.net_in_lag5.clip(lower=0) / TAU_IN_MED
    return d


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref); b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


def psi_daily(d):
    """PSI điểm dự báo, tham chiếu = validation, cửa sổ 5 ngày làm việc đến hết ngày D."""
    ref = {y: d[d.khoi == "validation"][f"p_{y}_final"].values for y in ("Y_short", "Y_full")}
    days_all = np.sort(d.ngay.unique()); rows = []
    for i, day in enumerate(days_all):
        if len(days_all[max(0, i - 4): i + 1]) < 5: continue
        w = d[d.ngay.isin(days_all[max(0, i - 4): i + 1])]
        rows.append({"ngay": day, "khoi": d[d.ngay == day].khoi.iloc[0],
                     "psi_short": psi(ref["Y_short"], w.p_Y_short_final.values), "psi_full": psi(ref["Y_full"], w.p_Y_full_final.values)})
    out = pd.DataFrame(rows); out["psi_max"] = out[["psi_short", "psi_full"]].max(axis=1)
    return out


def cell_perf(d):
    rows = []
    for (day, co), g in d.groupby(["ngay", "ca_ord"], sort=True):
        yy = (g.Y_short + B_FULL * g.Y_full).values
        rows.append({"ngay": day, "m": yy[np.argsort(-g.v.values)[:K_PER_SHIFT]].sum(), "f": yy[np.argsort(-g.v_du_phong.values)[:K_PER_SHIFT]].sum()})
    return pd.DataFrame(rows)


def perf_by_day(cellperf, days):
    """Hiệu quả thực 20 ô gần nhất (đến hết ngày D-1) của mô hình và của quy tắc dự phòng."""
    out = {}
    for day in days:
        prev = cellperf[cellperf.ngay < day].tail(20)
        out[day] = (prev.m.mean(), prev.f.mean())
    return out


def control_bucket(station, day, shift):
    h = hashlib.sha256(f"{CONTROL_SALT}|{station}|{pd.Timestamp(day).date()}|{shift}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def route(df, psi_by_day, perf_by_day=None, psi_override=None, shuffle=False, perf_override=None, use_soft=True, use_l3=True):
    plans, logs, hist = [], [], {}
    perf_by_day = perf_by_day or {}
    cells = list(df.groupby(["ngay", "ca_ord"], sort=True))
    day_index = {dd: i for i, dd in enumerate(sorted({c[0][0] for c in cells}))}
    for (day, co), g in cells:
        t1 = time.perf_counter()
        g = g.sample(frac=1, random_state=1) if shuffle else g
        g = g.sort_values("station_id") if not shuffle else g
        shift = g.ca.iloc[0]
        psi_v = psi_by_day.get(day, np.nan) if psi_override is None else psi_override
        pm, pfb = (perf_by_day.get(day, (np.nan, np.nan)) if perf_override is None else perf_override)
        fallback = bool(use_l3 and psi_v >= PSI_STOP and pm < pfb)
        warn = bool(psi_v >= PSI_WARN)
        v = (g.v_du_phong if fallback else g.v).values.copy()
        n_invalid = int((~g.hop_le.values).sum()); v[~g.hop_le.values] = 0
        order = np.argsort(g.station_id.values) if shuffle else np.arange(len(g))
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        solver = solve_greedy if fallback else solve_milp
        n_low = n_block = 0
        if use_soft:
            z0 = solver(v[order])[inv]
            low = (v < V_MIN) & (not fallback)
            di = day_index[day]
            blocked = np.array([sum(1 for dj in hist.get(s, []) if di - 4 <= dj <= di) >= MAX_VISITS_5D for s in g.station_id])
            n_low, n_block = int((z0 & low).sum()), int((z0 & blocked & ~low).sum())
            v[low] = 0; v[blocked] = 0
        zs = solver(v[order])
        z = zs[inv]
        di = day_index[day]
        for i in np.where(z)[0]:
            r = g.iloc[i]; sid = r.station_id
            bucket = control_bucket(sid, day, shift)
            arm = "CONTROL" if bucket < CONTROL_PCT else "TREATMENT"
            if arm == "TREATMENT": hist.setdefault(sid, []).append(di)
            direction = "SHORTAGE" if r.p_Y_short_final >= B_FULL * r.p_Y_full_final else "OVERFLOW"
            plans.append({"idempotency_key": f"{POLICY_ID}|{sid}|{pd.Timestamp(day).date()}|{shift}", "policy_id": POLICY_ID,
                          "model_version": MODEL_VERSION, "t0": (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=22)).isoformat(),
                          "ngay": day, "ca": shift, "station_id": sid, "arm": arm, "action": "VISIT" if arm == "TREATMENT" else "HOLD_CONTROL",
                          "reason_code": direction, "p_short": r.p_Y_short_final, "p_full": r.p_Y_full_final, "v": r.v,
                          "mode": "FALLBACK_RULE" if fallback else "MODEL", "bucket": bucket, "Y_short": r.Y_short, "Y_full": r.Y_full})
        logs.append({"ngay": day, "ca": shift, "psi": psi_v, "mode": "FALLBACK_RULE" if fallback else "MODEL", "canh_bao_psi": warn,
                     "bi_loai_khong_hop_le": n_invalid, "bi_loai_diem_thap": n_low, "bi_loai_gioi_han_lan_ghe": n_block,
                     "so_tram_chon": int(z.sum()), "phut": float(cost_min(zs)), "ms": (time.perf_counter() - t1) * 1000})
    return pd.DataFrame(plans), pd.DataFrame(logs)


def apply_override(cell_plan, cell_stations_valid, action, station, reason):
    """Quy tắc cứng không bao giờ bị ghi đè; mọi ghi đè cần lý do."""
    if not reason: return False, "thiếu lý do"
    cur = cell_plan[cell_plan.station_id == station]
    if action == "REMOVE":
        return (True, "quy tắc mềm") if len(cur) else (False, "trạm không có trong kế hoạch")
    if action == "ADD":
        if len(cur): return False, "DUP_STATION (lớp 1)"
        if station not in cell_stations_valid: return False, "STATION_INVALID (lớp 1)"
        sel_ids = cell_plan.station_id.tolist() + [station]
        zz = np.zeros(N, bool); zz[[stations.index(s) for s in sel_ids]] = True
        if zz.sum() > K_PER_SHIFT: return False, "OVER_K (lớp 1)"
        if cost_min(zz) > B_MIN: return False, "OVER_BUDGET (lớp 1)"
        return True, "quy tắc mềm"
    if action == "CONTROL_TO_TREATMENT": return False, "CONTROL_INTEGRITY (lớp 1)"
    return False, "hành động không hỗ trợ"


def router_run_and_test(dp, psi_by_day, perf_by_day, seed=SEED):
    """Chạy router trên một khối và chạy bộ kiểm tra tự động. Trả về dict: plan, log, bal, aa, tests."""
    rng = np.random.default_rng(seed)
    plan, log = route(dp, psi_by_day, perf_by_day)
    share = (plan.arm == "CONTROL").mean()
    rows = []
    for c in ["p_short", "p_full", "v"]:
        a, b = plan[plan.arm == "TREATMENT"][c], plan[plan.arm == "CONTROL"][c]
        rows.append({"bien": c, "TB_can_thiep": a.mean(), "TB_control": b.mean(), "SMD": (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)})
    plan["Y_any"] = plan.Y_short * 1.0 + plan.Y_full * B_FULL
    g_ = plan.groupby(["ngay", "arm"]).Y_any.agg(["sum", "size"]).unstack().fillna(0)
    udays = plan.ngay.unique(); bs = []
    for _ in range(1000):
        s_ = g_.loc[rng.choice(udays, len(udays))].sum()
        bs.append(s_[("sum", "TREATMENT")] / s_[("size", "TREATMENT")] - s_[("sum", "CONTROL")] / s_[("size", "CONTROL")])
    bal = pd.DataFrame(rows)
    smd_max = bal.SMD.abs().max()
    tests = []
    def T(name, ok, detail=""): tests.append({"kiem_tra": name, "ket_qua": "Đạt" if ok else "KHÔNG ĐẠT", "chi_tiet": detail})
    T("Idempotency: không có khoá trùng", plan.idempotency_key.is_unique, f"({len(plan)} khoá)")
    plan2, _ = route(dp, psi_by_day, perf_by_day)
    cols = ["idempotency_key", "arm", "reason_code", "mode"]
    T("Chạy lại cho kết quả y hệt (replay)", plan[cols].equals(plan2[cols]) and np.allclose(plan.v, plan2.v))
    plan3, _ = route(dp.sample(frac=1, random_state=7), psi_by_day, perf_by_day, shuffle=True)
    k1, k2 = set(zip(plan.idempotency_key, plan.arm)), set(zip(plan3.idempotency_key, plan3.arm))
    T("Đảo thứ tự dữ liệu đầu vào không đổi lệnh và CONTROL", k1 == k2, f"(khác biệt {len(k1 ^ k2)})")
    T("Lớp 1: mỗi ca <= K trạm", (log.so_tram_chon <= K_PER_SHIFT).all(), f"(tối đa {log.so_tram_chon.max()})")
    T("Lớp 1: mỗi ca <= B phút", (log.phut <= B_MIN).all(), f"(tối đa {log.phut.max():.0f})")
    T("Lớp 1: mỗi trạm một lần mỗi ca", not plan.duplicated(["station_id", "ngay", "ca"]).any())
    T("Lớp 1: không chọn trạm không hoạt động", bool(plan.merge(dp[["station_id", "ngay", "ca", "hop_le"]], on=["station_id", "ngay", "ca"]).hop_le.all()))
    T(f"CONTROL trong 8–12% (mục tiêu {CONTROL_PCT}%)", 0.08 <= share <= 0.12, f"({share:.1%})")
    T("CONTROL cân bằng điểm dự báo (|SMD| < 0,3)", smd_max < 0.3, f"(max {smd_max:.2f})")
    T("A/A: kết cục không khác giữa hai nhóm (CI chứa 0)", np.percentile(bs, 2.5) <= 0 <= np.percentile(bs, 97.5))
    h = dp.iloc[:N * 4]
    _, a1 = route(h, psi_by_day, psi_override=0.40, perf_override=(1.0, 2.0)); T("Lớp 3: PSI cao + mô hình kém hơn dự phòng -> dự phòng", (a1["mode"] == "FALLBACK_RULE").all())
    _, a2 = route(h, psi_by_day, psi_override=0.40, perf_override=(2.0, 1.0)); T("Lớp 3: PSI cao nhưng mô hình tốt hơn -> giữ mô hình, cảnh báo", (a2["mode"] == "MODEL").all() and a2.canh_bao_psi.all())
    _, a3 = route(h, psi_by_day, psi_override=0.02, perf_override=(1.0, 2.0)); T("Lớp 3: PSI thấp giữ mô hình", (a3["mode"] == "MODEL").all())
    ex = plan[(plan.ngay == plan.ngay.iloc[0]) & (plan.ca == plan.ca.iloc[0])]
    gcell = dp[(dp.ngay == plan.ngay.iloc[0]) & (dp.ca == plan.ca.iloc[0])]
    valid_ids = set(gcell[gcell.hop_le].station_id)
    other = [s for s in stations if s not in set(ex.station_id) and s in valid_ids]
    T("HITL: gỡ trạm kèm lý do được chấp nhận", apply_override(ex, valid_ids, "REMOVE", ex.station_id.iloc[0], "xe tải hỏng")[0])
    T("HITL: ghi đè không có lý do bị từ chối", not apply_override(ex, valid_ids, "REMOVE", ex.station_id.iloc[0], "")[0])
    if len(ex) >= K_PER_SHIFT: T("HITL: thêm trạm khi ca đã đủ K bị từ chối", not apply_override(ex, valid_ids, "ADD", other[0], "sự kiện")[0])
    T("HITL: chuyển CONTROL sang can thiệp bị từ chối", not apply_override(ex, valid_ids, "CONTROL_TO_TREATMENT", ex.station_id.iloc[0], "quản lý")[0])
    T("HITL: thêm trạm đã có bị từ chối", not apply_override(ex, valid_ids, "ADD", ex.station_id.iloc[0], "trùng")[0])
    return {"plan": plan, "log": log, "bal": bal, "aa": (float(np.mean(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "share": float(share), "smd_max": float(smd_max), "tests": pd.DataFrame(tests)}


# --- đánh giá dự báo + quyết định + kiểm tra tiêu chí chấp nhận ---
LABELS = ["Y_short", "Y_full"]
LAG = {"Y_short": "net_out_lag5", "Y_full": "net_in_lag5"}
N_BOOT, N_RAND = 1000, 10


def ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((t.n * (t.y - t.p).abs()).sum() / t.n.sum())


def precision_at_k(d, score, ycol, rng, k=K_PER_SHIFT):
    t = d[["ngay", "ca", ycol]].copy()
    t["s"] = np.asarray(score, float) + rng.uniform(0, 1e-9, len(t))
    t["r"] = t.groupby(["ngay", "ca"]).s.rank(ascending=False, method="first")
    return t[t.r <= k].groupby(["ngay", "ca"])[ycol].mean()


def evaluate_forecast(d, rng):
    rows = []
    for y in LABELS:
        rate = d[y].mean()
        for nm, s in (("GBM", d[f"p_{y}_final"].values), ("Cùng ca tuần trước", d[LAG[y]].values.astype(float)), ("Tỷ lệ lịch sử", d[f"rate_{y}"].values)):
            pk = precision_at_k(d, s, y, rng).mean()
            r = {"label": y, "model": nm, "PR_AUC": average_precision_score(d[y], s), "P@K": pk, "Lift@K": pk / rate, "ty_le_su_kien": rate}
            if nm == "GBM":
                p = np.clip(s, 1e-6, 1 - 1e-6)
                r.update({"ECE": ece(d[y].values, p), "TB_xac_suat": p.mean(), "ti_so_du_bao_thuc": p.mean() / rate})
            rows.append(r)
    return pd.DataFrame(rows)


def _to_mat(d, col):
    return d.sort_values(["ngay", "ca_ord", "station_id"])[col].values.reshape(-1, d.station_id.nunique())


def evaluate_decision(d, rng):
    ds = d.sort_values(["ngay", "ca_ord", "station_id"])
    days = ds.groupby(["ngay", "ca_ord"]).ngay.first().values
    R = _to_mat(ds, "Y_short") * 1.0 + B_FULL * _to_mat(ds, "Y_full")
    V, VL = _to_mat(ds, "v"), _to_mat(ds, "v_du_phong")
    Z = {"Oracle": np.array([solve_milp(r) for r in R]), "GBM + MILP": np.array([solve_milp(v) for v in V]),
         "GBM + Greedy": np.array([solve_greedy(v) for v in V]), "Cùng ca tuần trước + MILP": np.array([solve_milp(v) for v in VL])}
    val = {k: (z * R).sum(1) for k, z in Z.items()}
    val["Ngẫu nhiên + MILP"] = np.mean([(np.array([solve_milp(r) for r in rng.random(R.shape)]) * R).sum(1) for _ in range(N_RAND)], axis=0)
    pred_gap = (Z["GBM + MILP"] * V).sum(1) - (Z["GBM + Greedy"] * V).sum(1)
    udays = pd.Series(days).unique()
    def ci(a, b):
        per_day = pd.Series(val[a] - val[b]).groupby(days).mean()
        bs = [per_day.reindex(rng.choice(udays, len(udays))).mean() for _ in range(N_BOOT)]
        return float((val[a] - val[b]).mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    pairs = [("GBM + MILP", "Ngẫu nhiên + MILP"), ("GBM + MILP", "Cùng ca tuần trước + MILP"), ("GBM + MILP", "GBM + Greedy")]
    boot = pd.DataFrame([{"A": a, "B": b, **dict(zip(["chenh_TB", "CI95_duoi", "CI95_tren"], ci(a, b)))} for a, b in pairs])
    tab = pd.DataFrame([{"chinh_sach": k, "gia_tri_thuc_TB_moi_o": v.mean(), "ti_le_so_oracle": v.mean() / val["Oracle"].mean()} for k, v in val.items()])
    return tab, boot, {"so_o": len(R), "so_ngay": len(udays), "milp_ge_greedy_du_bao": float((pred_gap >= -1e-9).mean())}


def evaluate(d, seed=SEED):
    """Trả về (bảng dự báo, bảng giá trị quyết định, bảng bootstrap, dict chỉ số phẳng dùng cho tiêu chí chấp nhận)."""
    rng = np.random.default_rng(seed)
    fc = evaluate_forecast(d, rng)
    tab, boot, extra = evaluate_decision(d, rng)
    m = {}
    for y, k in (("Y_short", "short"), ("Y_full", "full")):
        g = fc[(fc.label == y) & (fc.model == "GBM")].iloc[0]; l5 = fc[(fc.label == y) & (fc.model == "Cùng ca tuần trước")].iloc[0]
        m[f"lift_{k}"], m[f"ece_{k}"], m[f"ti_so_{k}"] = g["Lift@K"], g["ECE"], g["ti_so_du_bao_thuc"]
        m[f"prauc_hon_lag5_{k}"] = g["PR_AUC"] - l5["PR_AUC"]
    v = tab.set_index("chinh_sach")
    m["gia_tri_gbm_milp"] = v.loc["GBM + MILP", "gia_tri_thuc_TB_moi_o"]
    m["ti_le_so_oracle"] = v.loc["GBM + MILP", "ti_le_so_oracle"]
    m["gbm_tru_ngau_nhien_TB"] = boot.iloc[0].chenh_TB; m["gbm_tru_ngau_nhien_CI_duoi"] = boot.iloc[0].CI95_duoi
    m["gbm_tru_lag5_TB"] = boot.iloc[1].chenh_TB
    m["milp_ge_greedy_du_bao"] = extra["milp_ge_greedy_du_bao"]
    return fc, tab, boot, m, extra


def full_block_metrics(d_all, psi_df, cperf, blk):
    """Toàn bộ chỉ số của một khối: dự báo, quyết định, router (kèm bộ kiểm tra tự động). d_all phải bắt đầu từ validation (PSI cần tham chiếu)."""
    dp = d_all[d_all.khoi == blk]
    pbd = perf_by_day(cperf, sorted(dp.ngay.unique()))
    fc, tab, boot, m, extra = evaluate(dp)
    rr = router_run_and_test(dp, psi_df.set_index("ngay").psi_max.to_dict(), pbd)
    log, plan = rr["log"], rr["plan"]
    m["router_tests_fail"] = int((rr["tests"].ket_qua != "Đạt").sum())
    m["fallback_share"] = float((log["mode"] == "FALLBACK_RULE").mean())
    m["latency_p95_ms"] = float(log.ms.quantile(.95))
    m["guardrail_mat_gia_tri"] = float(tab.set_index("chinh_sach").loc["GBM + MILP", "gia_tri_thuc_TB_moi_o"] - plan.Y_any.sum() / len(log))
    m["control_share"], m["smd_max"] = rr["share"], rr["smd_max"]
    return {"fc": fc, "tab": tab, "boot": boot, "m": m, "extra": extra, "rr": rr}


def predict_frozen(df, bundle):
    """Xác suất từ mô hình và bộ hiệu chỉnh đã đóng băng (cùng công thức với bước 4)."""
    F = bundle["features"]
    out = df[["station_id", "ngay", "ca", "khoi", "Y_short", "Y_full", "net_out", "net_in"]].copy()
    for y in LABELS:
        raw = bundle["models"][(y, "GBM")].predict_proba(df[F])[:, 1]
        c = bundle["calibrators"][(y, "GBM")]
        pc = np.clip(raw, 1e-6, 1 - 1e-6)
        platt = c["Platt"].predict_proba(np.log(pc / (1 - pc)).reshape(-1, 1))[:, 1]
        out[f"p_{y}_raw"], out[f"p_{y}_platt"] = raw, platt
        out[f"p_{y}_final"] = platt if c["chosen"] == "Platt" else raw
        r = bundle["rate"][y]
        out[f"rate_{y}"] = df[["station_id", "ca"]].merge(r, on=["station_id", "ca"], how="left").r.fillna(r.r.mean()).values
    return out


def check_criteria(crit, mets):
    """crit: bảng tiêu chí (khoi, chi_so, toan_tu, nguong); mets: {khoi: {chi_so: giá trị}}. Trả về bảng kết quả."""
    rows = []
    for r in crit.itertuples():
        val = float(mets[r.khoi][r.chi_so]); th = str(r.nguong)
        if r.toan_tu == "trong": lo, hi = map(float, th.split(";")); ok = lo <= val <= hi
        else: ok = {">=": val >= float(th), ">": val > float(th), "<=": val <= float(th), "==": val == float(th)}[r.toan_tu]
        rows.append({"ma": r.ma, "khoi": r.khoi, "noi_dung": r.noi_dung, "chi_so": r.chi_so, "toan_tu": r.toan_tu, "nguong": th,
                     "gia_tri_do_duoc": round(val, 4), "ket_qua": "Đạt" if ok else "KHÔNG ĐẠT"})
    return pd.DataFrame(rows)

#%% ======================================================================
def buoc_01_gop_du_lieu():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare (từ dự báo nhu cầu đến điều xe)
    # BƯỚC 1: Gộp 33 file CSV -> parquet, làm sạch, đối chiếu nhất quán
    #
    # Mục tiêu: biến 33 file CSV (~3,2 GB, ~17,6 triệu chuyến) thành các bảng nhẹ để
    # mọi bước sau chỉ đọc parquet, đồng thời để lại bằng chứng tái lập (SHA-256 từng
    # file gốc, số dòng bị loại theo từng lý do, các phép đối chiếu Đạt/Không).
    # Chưa chia train/test và chưa tính bất kỳ ngưỡng nào ở bước này, nên chưa có rò rỉ.
    # ========================================================================


    SEED = 42
    MIN_DUR_S = 60            # chuyến dưới 60 giây coi là lỗi/ghi nhầm (trang nguồn cũng nói đã loại)
    MAX_DUR_S = 24 * 3600     # chuyến trên 24 giờ coi là xe không được trả đúng cách
    TONG_DONG_GOC_DA_BIET = 17_605_262   # số dòng thu được ở lần khảo sát trước, dùng để đối chiếu

    SRC = BASE / "Datasets"
    DATA = BASE / "Data"
    RESULTS = BASE / "Results"
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    files = sorted(SRC.glob("*-capitalbikeshare-tripdata.csv"))
    assert len(files) == 33, f"Cần 33 file, đang có {len(files)}"
    t_start = time.perf_counter()
    print(f"[0] Tìm thấy {len(files)} file CSV: {files[0].name} ... {files[-1].name}")


    #%% ======================================================================
    # 1. SHA-256 từng file gốc
    # Mục tiêu: ghi dấu vân tay dữ liệu trước khi đụng vào, để Chương 8 chứng minh
    # được dữ liệu đầu vào không bị sửa giữa lúc làm bài và lúc nộp.
    # ========================================================================
    def sha256_of(path, chunk=8 * 1024 * 1024):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:       # hashlib nhả GIL nên chạy luồng là đủ nhanh
        hashes = list(ex.map(sha256_of, files))
    hash_df = pd.DataFrame({
        "file_month": [f.name[:6] for f in files],
        "file_name": [f.name for f in files],
        "bytes": [f.stat().st_size for f in files],
        "sha256": hashes,
    })
    print(f"[1/6] SHA-256 xong {len(files)} file trong {time.perf_counter() - t0:.1f}s")


    #%% ======================================================================
    # 2. Nạp CSV vào DuckDB (dạng chuỗi rồi ép kiểu)
    # Mục tiêu: đọc cả 33 file một lần. Đọc thành chuỗi trước rồi TRY_CAST để dòng lỗi
    # timestamp không làm hỏng cả file (từ 202406 timestamp có thêm mili-giây, trước đó thì không).
    # Ô trống ở cột trạm là NULL thật: đây là các chuyến e-bike không đỗ tại trạm.
    # ========================================================================
    work_db = Path(tempfile.gettempdir()) / "bikeshare_work.duckdb"   # DB tạm nằm ngoài OneDrive để khỏi đồng bộ file lớn
    if work_db.exists():
        work_db.unlink()
    con = duckdb.connect(str(work_db))
    con.execute("SET memory_limit='5GB'")
    con.execute("SET threads=8")
    con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_spill').as_posix()}'")

    paths_sql = "[" + ",".join("'" + f.as_posix().replace("'", "''") + "'" for f in files) + "]"
    t0 = time.perf_counter()
    con.execute(f"""
    CREATE TABLE raw AS
    SELECT
      ride_id,
      rideable_type,
      TRY_CAST(started_at AS TIMESTAMP) AS started_at,
      TRY_CAST(ended_at   AS TIMESTAMP) AS ended_at,
      NULLIF(trim(start_station_name), '') AS start_station_name,
      NULLIF(trim(start_station_id),   '') AS start_station_id,
      NULLIF(trim(end_station_name),   '') AS end_station_name,
      NULLIF(trim(end_station_id),     '') AS end_station_id,
      TRY_CAST(start_lat AS DOUBLE) AS start_lat,
      TRY_CAST(start_lng AS DOUBLE) AS start_lng,
      TRY_CAST(end_lat   AS DOUBLE) AS end_lat,
      TRY_CAST(end_lng   AS DOUBLE) AS end_lng,
      member_casual,
      regexp_extract(filename, '(\\d{{6}})-capitalbikeshare', 1) AS file_month
    FROM read_csv({paths_sql}, all_varchar=true, header=true, filename=true)
    """)
    n_raw = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    print(f"[2/6] Nạp xong {n_raw:,} dòng trong {time.perf_counter() - t0:.1f}s")


    #%% ======================================================================
    # 3. Gắn nhãn lý do loại (mỗi dòng chỉ mang MỘT lý do, theo thứ tự ưu tiên)
    # Mục tiêu: biết chính xác bao nhiêu dòng bị loại và vì sao; quy tắc áp dụng giống
    # nhau cho mọi tháng, nên sau này chia tập xong không phải làm sạch lại.
    # Trùng ride_id chỉ xét giữa các dòng đã hợp lệ (nếu bản đầu bị loại vì lỗi thì
    # bản sau không bị coi là trùng). Giữ bản ở file sớm hơn.
    # ========================================================================
    t0 = time.perf_counter()
    con.execute(f"""
    CREATE TABLE flagged AS
    WITH a AS (
      SELECT *,
        CASE
          WHEN started_at IS NULL OR ended_at IS NULL                    THEN '1_timestamp_loi'
          WHEN ended_at < started_at                                     THEN '2_thoi_luong_am'
          WHEN date_diff('second', started_at, ended_at) < {MIN_DUR_S}   THEN '3_duoi_60s'
          WHEN date_diff('second', started_at, ended_at) > {MAX_DUR_S}   THEN '4_tren_24h'
        END AS reason1
      FROM raw
    ), b AS (
      SELECT *,
        row_number() OVER (PARTITION BY ride_id, (reason1 IS NULL)
                           ORDER BY file_month, started_at) AS rn
      FROM a
    )
    SELECT * EXCLUDE (reason1, rn),
           COALESCE(reason1, CASE WHEN rn > 1 THEN '5_ride_id_trung' END) AS reason
    FROM b
    """)
    con.execute("DROP TABLE raw")

    reason_overall = con.execute("""
    SELECT COALESCE(reason, '0_giu_lai') AS ly_do, count(*) AS so_dong
    FROM flagged GROUP BY 1 ORDER BY 1
    """).df()
    reason_overall["ty_le_%"] = (reason_overall["so_dong"] / n_raw * 100).round(3)
    print(reason_overall.to_string(index=False))

    per_file = con.execute("""
    SELECT file_month,
           count(*)                                         AS dong_goc,
           count(*) FILTER (WHERE reason = '1_timestamp_loi') AS timestamp_loi,
           count(*) FILTER (WHERE reason = '2_thoi_luong_am') AS thoi_luong_am,
           count(*) FILTER (WHERE reason = '3_duoi_60s')      AS duoi_60s,
           count(*) FILTER (WHERE reason = '4_tren_24h')      AS tren_24h,
           count(*) FILTER (WHERE reason = '5_ride_id_trung') AS ride_id_trung,
           count(*) FILTER (WHERE reason IS NULL)             AS dong_giu_lai
    FROM flagged GROUP BY 1 ORDER BY 1
    """).df()
    audit = hash_df.merge(per_file, on="file_month", how="left")
    audit.to_csv(RESULTS / "data_audit.csv", index=False, encoding="utf-8-sig")
    reason_overall.to_csv(RESULTS / "data_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[3/6] Gắn nhãn xong trong {time.perf_counter() - t0:.1f}s")


    #%% ======================================================================
    # 4. Xuất các bảng parquet
    # Mục tiêu: chỉ giữ cột cần. trips_clean để tính biến theo mốc t0 sau này (ví dụ
    # chỉ dùng chuyến có ended_at <= t0); station_hour là bảng thưa trạm x giờ (ô
    # bằng 0 sẽ được thêm ở bước lập lưới, từ ngày trạm xuất hiện lần đầu);
    # nostation_hour giữ riêng dòng chuyến không có mã trạm vì không gán được vào trạm.
    # ========================================================================
    t0 = time.perf_counter()
    con.execute("CREATE TABLE clean AS SELECT * EXCLUDE (reason, start_station_name, end_station_name) "
                "FROM flagged WHERE reason IS NULL")
    con.execute("CREATE TABLE names AS SELECT start_station_id, start_station_name, end_station_id, end_station_name "
                "FROM flagged WHERE reason IS NULL")
    con.execute("DROP TABLE flagged")

    p_trips = (DATA / "trips_clean.parquet").as_posix()
    con.execute(f"""
    COPY (SELECT ride_id, rideable_type, member_casual, started_at, ended_at,
                 start_station_id, end_station_id, start_lat, start_lng, end_lat, end_lng, file_month
          FROM clean ORDER BY started_at)
    TO '{p_trips}' (FORMAT parquet, COMPRESSION zstd)
    """)

    p_sh = (DATA / "station_hour.parquet").as_posix()
    con.execute(f"""
    COPY (
      WITH o AS (SELECT start_station_id AS station_id, date_trunc('hour', started_at) AS hour_ts, count(*) AS n_out
                 FROM clean WHERE start_station_id IS NOT NULL GROUP BY 1, 2),
           i AS (SELECT end_station_id AS station_id, date_trunc('hour', ended_at) AS hour_ts, count(*) AS n_in
                 FROM clean WHERE end_station_id IS NOT NULL GROUP BY 1, 2)
      SELECT COALESCE(o.station_id, i.station_id) AS station_id,
             COALESCE(o.hour_ts,    i.hour_ts)    AS hour_ts,
             COALESCE(n_out, 0) AS n_out,
             COALESCE(n_in,  0) AS n_in
      FROM o FULL JOIN i ON o.station_id = i.station_id AND o.hour_ts = i.hour_ts
      ORDER BY station_id, hour_ts
    ) TO '{p_sh}' (FORMAT parquet, COMPRESSION zstd)
    """)

    p_ns = (DATA / "nostation_hour.parquet").as_posix()
    con.execute(f"""
    COPY (
      SELECT 'out' AS huong, date_trunc('hour', started_at) AS hour_ts, rideable_type, count(*) AS n
      FROM clean WHERE start_station_id IS NULL GROUP BY 2, 3
      UNION ALL
      SELECT 'in',  date_trunc('hour', ended_at),   rideable_type, count(*)
      FROM clean WHERE end_station_id IS NULL GROUP BY 2, 3
      ORDER BY 1, 2, 3
    ) TO '{p_ns}' (FORMAT parquet, COMPRESSION zstd)
    """)

    # Một id có thể mang nhiều tên theo thời gian -> lấy tên xuất hiện nhiều nhất; toạ độ lấy trung vị.
    p_st = (DATA / "stations.parquet").as_posix()
    con.execute(f"""
    COPY (
      WITH pts AS (
        SELECT c.start_station_id AS station_id, c.start_lat AS lat, c.start_lng AS lng, c.started_at AS ts
        FROM clean c WHERE c.start_station_id IS NOT NULL
        UNION ALL
        SELECT c.end_station_id, c.end_lat, c.end_lng, c.ended_at
        FROM clean c WHERE c.end_station_id IS NOT NULL
      ), nm AS (
        SELECT station_id, mode(name) AS station_name FROM (
          SELECT start_station_id AS station_id, start_station_name AS name FROM names WHERE start_station_id IS NOT NULL
          UNION ALL
          SELECT end_station_id, end_station_name FROM names WHERE end_station_id IS NOT NULL
        ) GROUP BY 1
      )
      SELECT p.station_id, nm.station_name,
             median(p.lat) AS lat, median(p.lng) AS lng,
             min(p.ts) AS first_seen, max(p.ts) AS last_seen, count(*) AS n_su_kien
      FROM pts p LEFT JOIN nm USING (station_id)
      GROUP BY p.station_id, nm.station_name ORDER BY p.station_id
    ) TO '{p_st}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"[4/6] Xuất parquet xong trong {time.perf_counter() - t0:.1f}s")
    for p in [p_trips, p_sh, p_ns, p_st]:
        print(f"      {Path(p).name:24s} {os.path.getsize(p) / 1e6:8.1f} MB")


    #%% ======================================================================
    # 5. Phép đối chiếu nhất quán giữa các bảng
    # Mục tiêu: lỗi khi gộp 17,6 triệu dòng rất khó thấy bằng mắt, nên mỗi bảng phải
    # khớp lại với bảng gốc bằng một phép kiểm có kết quả Đạt/Không.
    # ========================================================================
    checks = []
    def add_check(ten, ky_vong, thuc_te, dung=None):
        ok = (ky_vong == thuc_te) if dung is None else dung
        checks.append({"phep_kiem": ten, "ky_vong": ky_vong, "thuc_te": thuc_te, "ket_qua": "Đạt" if ok else "KHÔNG ĐẠT"})

    q = lambda sql: con.execute(sql).fetchone()[0]
    reason_sum = int(reason_overall.loc[reason_overall["ly_do"] != "0_giu_lai", "so_dong"].sum())
    n_clean = q("SELECT count(*) FROM clean")

    add_check("Tổng dòng 33 file = số dòng đã biết từ khảo sát trước", TONG_DONG_GOC_DA_BIET, n_raw)
    add_check("Tổng dòng theo từng file cộng lại = tổng dòng nạp", n_raw, int(audit["dong_goc"].sum()))
    add_check("Dòng gốc = dòng giữ lại + dòng bị loại", n_raw, n_clean + reason_sum)
    add_check("trips_clean.parquet có đúng số dòng giữ lại", n_clean,
              q(f"SELECT count(*) FROM read_parquet('{p_trips}')"))
    add_check("ride_id duy nhất trong dữ liệu sạch", n_clean,
              q(f"SELECT count(DISTINCT ride_id) FROM read_parquet('{p_trips}')"))
    add_check("Thời lượng nhỏ nhất trong dữ liệu sạch >= 60s",
              f">= {MIN_DUR_S}", q(f"SELECT min(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')"),
              dung=q(f"SELECT min(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')") >= MIN_DUR_S)
    add_check("Thời lượng lớn nhất trong dữ liệu sạch <= 24h",
              f"<= {MAX_DUR_S}", q(f"SELECT max(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')"),
              dung=q(f"SELECT max(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')") <= MAX_DUR_S)
    add_check("Tổng n_out (station_hour) = số chuyến sạch có mã trạm đi",
              q("SELECT count(*) FROM clean WHERE start_station_id IS NOT NULL"),
              q(f"SELECT sum(n_out) FROM read_parquet('{p_sh}')"))
    add_check("Tổng n_in (station_hour) = số chuyến sạch có mã trạm đến",
              q("SELECT count(*) FROM clean WHERE end_station_id IS NOT NULL"),
              q(f"SELECT sum(n_in) FROM read_parquet('{p_sh}')"))
    add_check("nostation_hour (out) = số chuyến sạch thiếu mã trạm đi",
              q("SELECT count(*) FROM clean WHERE start_station_id IS NULL"),
              q(f"SELECT coalesce(sum(n),0) FROM read_parquet('{p_ns}') WHERE huong='out'"))
    add_check("nostation_hour (in) = số chuyến sạch thiếu mã trạm đến",
              q("SELECT count(*) FROM clean WHERE end_station_id IS NULL"),
              q(f"SELECT coalesce(sum(n),0) FROM read_parquet('{p_ns}') WHERE huong='in'"))
    add_check("Chuyến thiếu mã trạm đi mà KHÔNG phải e-bike (kỳ vọng 0)", 0,
              q("SELECT count(*) FROM clean WHERE start_station_id IS NULL AND rideable_type <> 'electric_bike'"))
    add_check("Mỗi (trạm, giờ) xuất hiện đúng một dòng trong station_hour",
              q(f"SELECT count(*) FROM read_parquet('{p_sh}')"),
              q(f"SELECT count(*) FROM (SELECT DISTINCT station_id, hour_ts FROM read_parquet('{p_sh}'))"))
    add_check("Mọi trạm trong station_hour đều có trong stations.parquet", 0,
              q(f"SELECT count(*) FROM (SELECT DISTINCT station_id FROM read_parquet('{p_sh}')) s "
                f"WHERE s.station_id NOT IN (SELECT station_id FROM read_parquet('{p_st}'))"))
    add_check("Đủ 33 tháng trong dữ liệu sạch", 33,
              q(f"SELECT count(DISTINCT file_month) FROM read_parquet('{p_trips}')"))

    chk = pd.DataFrame(checks)
    chk.to_csv(RESULTS / "consistency_checks.csv", index=False, encoding="utf-8-sig")
    print(chk.to_string(index=False))
    n_fail = int((chk["ket_qua"] != "Đạt").sum())
    print(f"[5/6] {len(chk) - n_fail}/{len(chk)} phép kiểm Đạt")


    #%% ======================================================================
    # 6. Tóm tắt nhanh để dùng trong báo cáo
    # ========================================================================
    n_st = q(f"SELECT count(*) FROM read_parquet('{p_st}')")
    rng = con.execute(f"SELECT min(started_at), max(started_at) FROM read_parquet('{p_trips}')").fetchone()
    print(f"      Chuyến sạch: {n_clean:,} / {n_raw:,} ({n_clean / n_raw * 100:.2f}%)  | số trạm: {n_st}")
    print(f"      Khoảng thời gian: {rng[0]} -> {rng[1]}")
    con.close()
    work_db.unlink(missing_ok=True)
    print(f"[✓] BƯỚC 1 hoàn tất sau {time.perf_counter() - t_start:.1f}s"
          + ("" if n_fail == 0 else f"  – CÓ {n_fail} PHÉP KIỂM KHÔNG ĐẠT, cần xem lại!"))

#%% ======================================================================
def buoc_02_eda_chon_tram():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 2: Khám phá dữ liệu (EDA) và chọn tập trạm cho bài toán tối ưu
    #
    # Mục tiêu: (a) vẽ các hình mô tả dữ liệu cho báo cáo; (b) chọn 80 trạm chỉ dựa
    # vào giai đoạn TRAIN (202401–202510) để không rò rỉ thông tin từ validation/test;
    # (c) xem phân phối dòng xe ròng theo ca để biết việc chọn ngưỡng τ có khả thi không.
    # Mọi thống kê dùng để chọn trạm/ngưỡng chỉ tính trên train.
    # ========================================================================

    SEED = 42
    TRAIN_START, TRAIN_END = "2024-01-01", "2025-11-01"     # train = 202401–202510 (nửa mở)
    N_STATIONS = 80
    SHIFTS = {"sang": (6, 10), "chieu": (16, 20)}            # giờ bắt đầu (gồm) -> kết thúc (không gồm)
    COLOR = {"main": "#1F3A5F", "alt": "#D9822B", "grey": "#8A8A8A"}

    DATA, RESULTS = BASE / "Data", BASE / "Results"
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    con = duckdb.connect()
    T = (DATA / "trips_clean.parquet").as_posix()
    SH = (DATA / "station_hour.parquet").as_posix()
    NS = (DATA / "nostation_hour.parquet").as_posix()
    ST = (DATA / "stations.parquet").as_posix()
    t_start = time.perf_counter()


    #%% ======================================================================
    # 1. Hình 1 – Số chuyến theo tháng và tỷ lệ e-bike thiếu mã trạm (mốc gãy 08/2025)
    # ========================================================================
    m = con.execute(f"""
    SELECT strftime(started_at, '%Y-%m') AS thang, count(*) AS chuyen,
           count(*) FILTER (WHERE rideable_type='electric_bike') AS ebike,
           count(*) FILTER (WHERE rideable_type='electric_bike' AND start_station_id IS NULL) AS ebike_thieu_tram,
           count(*) FILTER (WHERE rideable_type='electric_bike' AND start_station_id IS NOT NULL) AS ebike_co_tram
    FROM read_parquet('{T}') GROUP BY 1 ORDER BY 1
    """).df()
    m["ty_le_ebike_thieu_tram_%"] = (m.ebike_thieu_tram / m.ebike * 100).round(1)
    m.to_csv(RESULTS / "eda_theo_thang.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(m.thang, m.chuyen / 1e3, color=COLOR["main"])
    ax[0].set_title("Hình 1a. Số chuyến sạch theo tháng (nghìn)")
    ax[0].set_xticks(range(0, len(m), 3)); ax[0].set_xticklabels(m.thang[::3], rotation=45)
    ax[1].plot(m.thang, m["ty_le_ebike_thieu_tram_%"], marker="o", color=COLOR["alt"], label="% e-bike thiếu mã trạm đi")
    ax[1].axvline("2025-08", color="red", ls="--", lw=1); ax[1].text("2025-08", 5, " 08/2025", color="red")
    ax[1].set_title("Hình 1b. Tỷ lệ e-bike không có mã trạm đi"); ax[1].set_ylim(0, 60)
    ax[1].set_xticks(range(0, len(m), 3)); ax[1].set_xticklabels(m.thang[::3], rotation=45); ax[1].legend()
    plt.tight_layout(); plt.savefig(RESULTS / "eda_hinh1_theo_thang.png"); plt.close()
    print("[1/6] Hình 1 xong")


    #%% ======================================================================
    # 2. Hình 2 – Phân phối thời lượng chuyến (log) theo loại xe và loại khách
    # Mục tiêu: nhìn thấy đuôi dài và khác biệt casual/member; là cơ sở giải thích vì sao
    # phải cắt <60s và >24h (quy tắc đã áp dụng ở Bước 1).
    # ========================================================================
    d = con.execute(f"""
    SELECT rideable_type, member_casual,
           date_diff('second', started_at, ended_at) / 60.0 AS phut
    FROM read_parquet('{T}') USING SAMPLE 400000 ROWS (reservoir, {SEED})
    """).df()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(0, np.log10(1440), 60)
    for k, g in d.groupby("member_casual"):
        ax[0].hist(g.phut, bins=bins, alpha=.6, label=k, density=True)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Thời lượng (phút, trục log)")
    ax[0].set_title("Hình 2a. Phân phối thời lượng theo loại khách"); ax[0].legend()
    order = sorted(d.rideable_type.unique())
    ax[1].boxplot([np.log10(d.phut[d.rideable_type == r]) for r in order], tick_labels=order, showfliers=False)
    ax[1].set_ylabel("log10(phút)"); ax[1].set_title("Hình 2b. Thời lượng theo loại xe (không vẽ ngoại lai)")
    plt.tight_layout(); plt.savefig(RESULTS / "eda_hinh2_thoi_luong.png"); plt.close()
    dur_stats = d.groupby("member_casual").phut.describe(percentiles=[.5, .9, .99]).round(1)
    dur_stats.to_csv(RESULTS / "eda_thoi_luong_thong_ke.csv", encoding="utf-8-sig")
    print(dur_stats.to_string())
    print("[2/6] Hình 2 xong")


    #%% ======================================================================
    # 3. Hình 3 – Heatmap nhu cầu theo giờ x thứ (chỉ chuyến có mã trạm đi, giai đoạn train)
    # Mục tiêu: lộ ra hai đỉnh sáng/chiều ngày thường, là lý do chọn 2 ca 06–10h và 16–20h.
    # ========================================================================
    h = con.execute(f"""
    SELECT isodow(hour_ts) AS thu, hour(hour_ts) AS gio, sum(n_out) AS chuyen
    FROM read_parquet('{SH}') WHERE hour_ts >= '{TRAIN_START}' AND hour_ts < '{TRAIN_END}'
    GROUP BY 1, 2
    """).df().pivot(index="thu", columns="gio", values="chuyen")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(h.values / 1e3, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(7)); ax.set_yticklabels(["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24)); ax.set_xlabel("Giờ trong ngày")
    ax.grid(False); fig.colorbar(im, label="Nghìn chuyến (train)")
    for s, (a, b) in SHIFTS.items():
        ax.add_patch(plt.Rectangle((a - .5, -.5), b - a, 7, fill=False, ec="red", lw=1.5))
    ax.set_title("Hình 3. Chuyến đi theo giờ và thứ (khung đỏ = hai ca điều xe)")
    plt.tight_layout(); plt.savefig(RESULTS / "eda_hinh3_heatmap_gio_thu.png"); plt.close()
    print("[3/6] Hình 3 xong")


    #%% ======================================================================
    # 4. Chọn 80 trạm: CHỈ dùng train
    # Quy tắc (cố định trước khi xem bất kỳ kết quả mô hình nào):
    #   - có chuyến đi ở đủ 22 tháng của train (trạm hoạt động liên tục, không phải trạm mới/tạm);
    #   - xếp theo tổng chuyến đi + đến trong train, lấy 80 trạm đầu.
    # Sau đó (chỉ để báo cáo) kiểm tra các trạm này còn hoạt động ở các giai đoạn sau không.
    # ========================================================================
    cand = con.execute(f"""
    WITH s AS (
      SELECT station_id, strftime(hour_ts,'%Y-%m') AS thang, sum(n_out) AS o, sum(n_in) AS i
      FROM read_parquet('{SH}') WHERE hour_ts >= '{TRAIN_START}' AND hour_ts < '{TRAIN_END}'
      GROUP BY 1, 2
    )
    SELECT station_id, count(*) FILTER (WHERE o > 0) AS thang_co_chuyen, sum(o) AS out_train, sum(i) AS in_train,
           sum(o + i) AS tong_train
    FROM s GROUP BY 1
    """).df()
    n_thang_train = 22
    ok = cand[cand.thang_co_chuyen == n_thang_train].sort_values("tong_train", ascending=False)
    chosen = ok.head(N_STATIONS).copy()
    names = con.execute(f"SELECT station_id, station_name, lat, lng FROM read_parquet('{ST}')").df()
    chosen = chosen.merge(names, on="station_id", how="left")
    chosen["xep_hang"] = range(1, len(chosen) + 1)
    chosen.to_csv(RESULTS / "stations_selected.csv", index=False, encoding="utf-8-sig")
    print(f"      Trạm có chuyến đủ {n_thang_train} tháng train: {len(ok)} / {len(cand)} trạm có hoạt động trong train")
    print(f"      80 trạm chọn chiếm {chosen.tong_train.sum() / cand.tong_train.sum() * 100:.1f}% tổng chuyến (đi+đến) có mã trạm trong train")

    sid = "(" + ",".join("'" + s + "'" for s in chosen.station_id) + ")"
    after = con.execute(f"""
    SELECT strftime(hour_ts,'%Y-%m') AS thang, count(DISTINCT station_id) FILTER (WHERE n_out > 0) AS tram_con_hoat_dong, sum(n_out) AS out
    FROM read_parquet('{SH}') WHERE station_id IN {sid} AND hour_ts >= '2025-11-01' GROUP BY 1 ORDER BY 1
    """).df()
    print("      Số trạm (trong 80) còn có chuyến đi, các tháng sau train:")
    print(after.to_string(index=False))
    after.to_csv(RESULTS / "stations_selected_con_hoat_dong.csv", index=False, encoding="utf-8-sig")
    print("[4/6] Chọn trạm xong")


    #%% ======================================================================
    # 5. Dòng xe ròng theo (trạm, ngày, ca) của 80 trạm – chỉ giai đoạn train
    # Net outflow = số xe rời trạm trừ số xe đến trạm trong ca. Dương lớn = trạm bị rút xe.
    # Lưu ý: đây là PROXY của nguy cơ thiếu xe (không có số xe thật tại trạm).
    # Hình 4 dùng để xem ngưỡng τ (phân vị 90 theo từng trạm) có khả thi không.
    # ========================================================================
    parts = []
    for s, (a, b) in SHIFTS.items():
        parts.append(con.execute(f"""
        SELECT station_id, CAST(hour_ts AS DATE) AS ngay, '{s}' AS ca,
               sum(n_out) AS n_out, sum(n_in) AS n_in, sum(n_out) - sum(n_in) AS net_out
        FROM read_parquet('{SH}')
        WHERE station_id IN {sid} AND hour(hour_ts) >= {a} AND hour(hour_ts) < {b}
          AND hour_ts >= '{TRAIN_START}' AND hour_ts < '{TRAIN_END}' AND isodow(hour_ts) <= 5
        GROUP BY 1, 2""").df())
    net = pd.concat(parts)
    # ngày thường: thêm các ô vắng mặt (không có chuyến) = 0 để không thổi phồng phân phối
    days = pd.bdate_range("2024-01-01", "2025-10-31")
    idx = pd.MultiIndex.from_product([chosen.station_id, days.date, list(SHIFTS)], names=["station_id", "ngay", "ca"])
    net = net.set_index(["station_id", "ngay", "ca"]).reindex(idx, fill_value=0).reset_index()
    tau = net.groupby(["station_id", "ca"]).net_out.quantile(.90).rename("tau_p90").reset_index()
    net = net.merge(tau, on=["station_id", "ca"])
    net["Y"] = (net.net_out >= net.tau_p90) & (net.net_out > 0)
    rate = net.groupby("ca").Y.mean().round(3)
    print("      Tỷ lệ Y=1 theo ca (train, ngày thường):"); print(rate.to_string())
    print("      Thống kê net_out theo ca:"); print(net.groupby("ca").net_out.describe(percentiles=[.1, .5, .9, .99]).round(2).to_string())
    net.groupby("ca").net_out.describe(percentiles=[.1, .5, .9, .99]).round(2).to_csv(RESULTS / "eda_net_out_theo_ca.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for s, c in zip(SHIFTS, [COLOR["main"], COLOR["alt"]]):
        ax[0].hist(net.net_out[net.ca == s].clip(-30, 30), bins=61, alpha=.6, label=f"ca {s}", color=c)
    ax[0].set_xlabel("Net outflow trong ca (xe đi − xe đến, cắt ±30)"); ax[0].set_title("Hình 4a. Phân phối dòng xe ròng theo ca"); ax[0].legend()
    top = chosen.head(10).station_id.tolist()
    bx = [net.net_out[(net.station_id == t) & (net.ca == "sang")] for t in top]
    ax[1].boxplot(bx, tick_labels=[f"#{i + 1}" for i in range(10)], showfliers=False)
    ax[1].set_title("Hình 4b. Net outflow ca sáng của 10 trạm đông nhất"); ax[1].set_xlabel("Xếp hạng trạm")
    plt.tight_layout(); plt.savefig(RESULTS / "eda_hinh4_net_outflow.png"); plt.close()
    print("[5/6] Hình 4 xong")


    #%% ======================================================================
    # 6. Tỷ lệ chuyến bị mất khi chỉ dùng chuyến có mã trạm (độ lệch của dữ liệu)
    # Mục tiêu: nói thẳng bao nhiêu % chuyến e-bike không gán được vào trạm, theo từng khối thời gian của kế hoạch chia tập.
    # ========================================================================
    blocks = {"train 202401–202510": ("2024-01-01", "2025-11-01"), "validation 202511–12": ("2025-11-01", "2026-01-01"),
              "calibration 202601–02": ("2026-01-01", "2026-03-01"), "policy val 202603–04": ("2026-03-01", "2026-05-01"),
              "locked 202605–06": ("2026-05-01", "2026-07-01"), "OOT 202607–09": ("2026-07-01", "2026-10-01")}
    rows = []
    for k, (a, b) in blocks.items():
        r = con.execute(f"""
        SELECT count(*) AS chuyen, count(*) FILTER (WHERE start_station_id IS NULL) AS thieu_tram_di
        FROM read_parquet('{T}') WHERE started_at >= '{a}' AND started_at < '{b}'""").fetchone()
        rows.append({"khoi": k, "chuyen": r[0], "thieu_ma_tram_di": r[1], "ty_le_%": round(r[1] / r[0] * 100, 1)})
    blk = pd.DataFrame(rows)
    blk.to_csv(RESULTS / "eda_khoi_thoi_gian.csv", index=False, encoding="utf-8-sig")
    print(blk.to_string(index=False))
    con.close()
    print(f"[6/6] [✓] BƯỚC 2 hoàn tất sau {time.perf_counter() - t_start:.1f}s")

#%% ======================================================================
def buoc_03_dac_trung():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 3: Lập lưới (trạm, ngày thường, ca), nhãn Y và đặc trưng tại t0
    #
    # Mục tiêu: tạo bảng dữ liệu cho mô hình, trong đó mọi đặc trưng của ngày D chỉ dùng
    # thông tin có trước t0 = 22:00 ngày D−1. Kỹ thuật: mọi đặc trưng đều là shift(1) theo
    # ngày làm việc (lag/rolling kết thúc ở ngày làm việc liền trước), nên không thể chứa
    # kết quả của ngày D. Cuối file có thí nghiệm kiểm tra rò rỉ để chứng minh điều đó.
    # Chỉ dựng đến hết policy_val (DEV_END); locked/oot chưa được đụng tới.
    # ========================================================================

    t_start = time.perf_counter()
    LAGS = [1, 2, 3, 4, 5]
    WINS = [5, 10, 20]
    VARS = ["net_out", "net_in", "n_out", "n_in"]
    WARMUP_DAYS = 20          # 20 ngày làm việc đầu chưa đủ lịch sử cho rolling 20 -> bỏ khỏi bảng mô hình

    sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str})
    stations = sel.station_id.tolist()
    assert len(stations) == N_STATIONS
    sid = "(" + ",".join("'" + s + "'" for s in stations) + ")"
    con = duckdb.connect()
    SH = (DATA / "station_hour.parquet").as_posix()


    #%% ======================================================================
    # 1. Lưới (trạm, ngày thường, ca): đếm xe đi/đến; ô không có chuyến = 0
    # Mục tiêu: nhu cầu bằng 0 là thông tin thật, không được để thiếu ô.
    # ========================================================================
    case_shift = " ".join(f"WHEN hour(hour_ts) >= {a} AND hour(hour_ts) < {b} THEN '{s}'" for s, (a, b) in SHIFTS.items())
    agg = con.execute(f"""
    SELECT station_id, CAST(hour_ts AS DATE) AS ngay, CASE {case_shift} END AS ca,
           sum(n_out) AS n_out, sum(n_in) AS n_in
    FROM read_parquet('{SH}')
    WHERE station_id IN {sid} AND isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{DEV_END}'
    GROUP BY 1, 2, 3 HAVING ca IS NOT NULL
    """).df()
    agg["ngay"] = pd.to_datetime(agg["ngay"])

    days = pd.bdate_range(BLOCKS["train"][0], DEV_END)
    idx = pd.MultiIndex.from_product([stations, days, list(SHIFTS)], names=["station_id", "ngay", "ca"])
    base = agg.set_index(["station_id", "ngay", "ca"]).reindex(idx, fill_value=0).reset_index()
    base[["n_out", "n_in"]] = base[["n_out", "n_in"]].astype(int)
    base["net_out"] = base.n_out - base.n_in
    base["net_in"] = -base.net_out
    print(f"[1/5] Lưới: {len(base):,} dòng = {N_STATIONS} trạm x {len(days)} ngày thường x {len(SHIFTS)} ca")

    # số chuyến toàn hệ thống mỗi ngày thường (dùng làm đặc trưng "mức cầu hôm qua")
    city = con.execute(f"""
    SELECT CAST(hour_ts AS DATE) AS ngay, sum(n_out) AS city_out FROM read_parquet('{SH}')
    WHERE isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{DEV_END}' GROUP BY 1 ORDER BY 1
    """).df()
    city["ngay"] = pd.to_datetime(city["ngay"])
    city = city.set_index("ngay").reindex(days, fill_value=0)


    #%% ======================================================================
    # 2. Ngưỡng τ học trên TRAIN và nhãn
    # Y_short: xe ròng đi ra >= τ_out của đúng trạm đó trong đúng ca đó (và > 0).
    # Y_full : xe ròng đi vào >= τ_in (và > 0). Cả hai chỉ là proxy, không có số xe thật tại trạm.
    # τ tính một lần trên train rồi dùng cố định cho mọi khối sau (không tính lại trên khối khác).
    # ========================================================================
    tr = base[(base.ngay >= BLOCKS["train"][0]) & (base.ngay <= BLOCKS["train"][1])]
    tau = tr.groupby(["station_id", "ca"]).agg(
        tau_out=("net_out", lambda s: s.quantile(TAU_QUANTILE)),
        tau_in=("net_in", lambda s: s.quantile(TAU_QUANTILE))).reset_index()
    tau.to_csv(RESULTS / "tau_thresholds.csv", index=False, encoding="utf-8-sig")
    base = base.merge(tau, on=["station_id", "ca"], how="left")
    base["Y_short"] = ((base.net_out >= base.tau_out) & (base.net_out > 0)).astype(int)
    base["Y_full"] = ((base.net_in >= base.tau_in) & (base.net_in > 0)).astype(int)
    base["khoi"] = block_of(base.ngay)
    print("[2/5] τ xong. Tỷ lệ sự kiện theo khối (cùng τ của train):")
    print(base.groupby("khoi")[["Y_short", "Y_full"]].mean().round(3).to_string())


    #%% ======================================================================
    # 3. Đặc trưng tại t0 (hàm dùng lại được cho thí nghiệm rò rỉ ở mục 5)
    # ========================================================================
    def build_features(b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        """b: lưới (đã có n_out, n_in, net_out, net_in); c: tổng chuyến toàn hệ thống theo ngày."""
        f = b[["station_id", "ngay", "ca"]].copy()
        b = b.sort_values(["station_id", "ca", "ngay"]).reset_index(drop=True)
        f = b[["station_id", "ngay", "ca"]].copy()
        g = b.groupby(["station_id", "ca"], sort=False)
        for v in VARS:
            for k in LAGS:
                f[f"{v}_lag{k}"] = g[v].shift(k)
            s1 = g[v].shift(1)
            for w in WINS:
                f[f"{v}_m{w}"] = s1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(w, min_periods=w).mean())
            f[f"{v}_m60"] = s1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(60, min_periods=20).mean())
        # Đặc trưng tương đối so với ngưỡng τ của chính trạm-ca: cho mô hình biết "bao nhiêu thì là bất thường ở trạm này".
        # τ là tham số cố định học trên train (không đổi theo ngày), còn sự kiện được tính lại từ số liệu đã shift(1).
        for v, t in (("net_out", "tau_out"), ("net_in", "tau_in")):
            den = b[t].clip(lower=1)
            for k in (1, 2, 5):
                f[f"{v}_lag{k}_over_tau"] = f[f"{v}_lag{k}"] / den
            for w in WINS + [60]:
                f[f"{v}_m{w}_over_tau"] = f[f"{v}_m{w}"] / den
            f[t] = b[t]
        for v, t, nm in (("net_out", "tau_out", "ev_short"), ("net_in", "tau_in", "ev_full")):
            ev = ((b[v] >= b[t]) & (b[v] > 0)).astype(float)
            e1 = ev.groupby([b.station_id, b.ca], sort=False).shift(1)
            f[f"{nm}_lag1"] = e1
            for w in (10, 20, 60):
                f[f"{nm}_r{w}"] = e1.groupby([b.station_id, b.ca], sort=False).transform(lambda s: s.rolling(w, min_periods=5).mean())
        # ca đối diện của ngày làm việc liền trước (đã kết thúc trước t0)
        opp = b[["station_id", "ngay", "ca", "net_out", "n_out", "n_in"]].copy()
        opp["ca"] = opp.ca.map({"sang": "chieu", "chieu": "sang"})
        nxt = {d: n for d, n in zip(c.index[:-1], c.index[1:])}
        opp["ngay"] = opp.ngay.map(nxt)                      # ngày làm việc kế tiếp của dòng đó = ngày D dùng nó
        opp = opp.dropna(subset=["ngay"]).rename(columns={"net_out": "opp_net_out_lag1", "n_out": "opp_n_out_lag1", "n_in": "opp_n_in_lag1"})
        f = f.merge(opp, on=["station_id", "ngay", "ca"], how="left")
        # mức cầu toàn hệ thống: hôm qua (ngày làm việc trước) so với trung bình 20 ngày trước đó
        cs = c.city_out.shift(1)
        cf = pd.DataFrame({"city_out_lag1": cs, "city_out_m20": cs.rolling(20, min_periods=20).mean()})
        cf["city_ratio"] = cf.city_out_lag1 / cf.city_out_m20
        f = f.merge(cf.reset_index().rename(columns={"index": "ngay"}), on="ngay", how="left")
        # lịch (biết trước)
        f["dow"] = f.ngay.dt.dayofweek
        f["thang"] = f.ngay.dt.month
        f["thang_sin"] = np.sin(2 * np.pi * f.thang / 12)
        f["thang_cos"] = np.cos(2 * np.pi * f.thang / 12)
        f["la_ca_sang"] = (f.ca == "sang").astype(int)
        f["la_ngay_le"] = f.ngay.isin(HOLIDAYS).astype(int)
        prev = {n: d for d, n in nxt.items()}
        f["hom_qua_la_le"] = f.ngay.map(prev).isin(HOLIDAYS).astype(int)
        st = sel.set_index("station_id")[["lat", "lng"]]
        f = f.merge(st, left_on="station_id", right_index=True, how="left")
        return f

    feat = build_features(base, city)
    lbl = base[["station_id", "ngay", "ca", "net_out", "net_in", "Y_short", "Y_full", "khoi"]]
    df = feat.merge(lbl, on=["station_id", "ngay", "ca"], how="left")
    FEATURES = [c for c in feat.columns if c not in ("station_id", "ngay", "ca")]

    first_ok = days[WARMUP_DAYS]
    n0 = len(df)
    df = df[df.ngay >= first_ok].dropna(subset=FEATURES).reset_index(drop=True)
    print(f"[3/5] {len(FEATURES)} đặc trưng; bỏ {n0 - len(df):,} dòng khởi động (trước {first_ok.date()}); còn {len(df):,} dòng")


    #%% ======================================================================
    # 4. Lưu bảng + thống kê cho báo cáo
    # ========================================================================
    df.to_parquet(DATA / "features_dev.parquet", index=False)
    pd.Series(FEATURES, name="feature").to_csv(RESULTS / "feature_list.csv", index=False)
    summ = df.groupby("khoi").agg(dong=("Y_short", "size"), ngay=("ngay", "nunique"),
                                  ty_le_Y_short=("Y_short", "mean"), ty_le_Y_full=("Y_full", "mean")).round(3)
    summ.to_csv(RESULTS / "dataset_dev_summary.csv", encoding="utf-8-sig")
    print(summ.to_string())
    print(f"[4/5] Đã lưu features_dev.parquet ({(DATA / 'features_dev.parquet').stat().st_size / 1e6:.1f} MB)")


    #%% ======================================================================
    # 5. Thí nghiệm kiểm tra rò rỉ
    # Cách làm: với một số ngày D ngẫu nhiên, GHI ĐÈ mọi kết quả từ ngày D trở đi bằng số vô nghĩa
    # (99999), dựng lại đặc trưng, và so với bảng đã lưu. Nếu đặc trưng của ngày D đổi thì nó
    # đang nhìn thấy tương lai. Kết quả mong đợi: không có khác biệt nào.
    # ========================================================================
    rng = np.random.default_rng(SEED)
    test_days = rng.choice(df.ngay.unique(), size=4, replace=False)
    rows = []
    for d in sorted(test_days):
        d = pd.Timestamp(d)
        b2 = base.copy()
        m = b2.ngay >= d
        for v in VARS:
            b2.loc[m, v] = 99999
        c2 = city.copy(); c2.loc[c2.index >= d, "city_out"] = 99999
        f2 = build_features(b2, c2)
        a = f2[f2.ngay == d].set_index(["station_id", "ca"])[FEATURES].sort_index()
        o = df[df.ngay == d].set_index(["station_id", "ca"])[FEATURES].sort_index()
        diff = np.nanmax(np.abs(a.values.astype(float) - o.values.astype(float)))
        rows.append({"ngay_kiem_tra": d.date(), "so_dong": len(o), "chenh_lech_lon_nhat": float(diff),
                     "ket_qua": "Đạt" if diff == 0 else "KHÔNG ĐẠT"})
    leak = pd.DataFrame(rows)
    leak.to_csv(RESULTS / "leakage_experiment.csv", index=False, encoding="utf-8-sig")
    print(leak.to_string(index=False))
    print(f"[5/5] [✓] BƯỚC 3 hoàn tất sau {time.perf_counter() - t_start:.1f}s")

#%% ======================================================================
def buoc_04_du_bao_hieu_chinh():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 4: Dự báo xác suất rủi ro (thiếu xe / đầy xe) và hiệu chỉnh (calibration)
    #
    # Mục tiêu: huấn luyện trên TRAIN, chọn mô hình trên VALIDATION, học bộ hiệu chỉnh trên
    # CALIBRATION (tập riêng), kiểm tra hiệu chỉnh trên POLICY_VAL. Không đụng locked/oot.
    # So với hai mốc đơn giản: "cùng ca tuần trước" và "tỷ lệ lịch sử của trạm-ca (train)".
    # Chỉ số: PR-AUC, Precision@K (K = số trạm xe tải ghé được mỗi ca), Brier, log loss, ECE.
    # Độ ổn định: 5 seed + bootstrap theo khối ngày (không bootstrap từng dòng vì các ô cùng ngày tương quan).
    # ========================================================================

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    N_BOOT = 500
    SEEDS = [0, 1, 2, 3, 4]
    LABELS = {"Y_short": "thiếu xe", "Y_full": "đầy xe"}
    BASELINE_LAG = {"Y_short": "net_out_lag5", "Y_full": "net_in_lag5"}   # lag5 ngày làm việc = cùng ca, cùng thứ, tuần trước
    COLOR = {"gbm": "#1F3A5F", "logit": "#2E8B57", "lag5": "#8A8A8A", "rate": "#D9822B"}
    plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    (DATA / "models").mkdir(exist_ok=True)

    df = pd.read_parquet(DATA / "features_dev.parquet")
    FEATURES = pd.read_csv(RESULTS / "feature_list.csv").feature.tolist()
    blk = {k: df[df.khoi == k].reset_index(drop=True) for k in ["train", "validation", "calibration", "policy_val"]}
    assert df.khoi.isin(["train", "validation", "calibration", "policy_val"]).all()
    print("[0] " + ", ".join(f"{k}={len(v):,}" for k, v in blk.items()))


    #%% ======================================================================
    # 1. Hàm đánh giá
    # ========================================================================
    rng_tie = np.random.default_rng(SEED)

    def ece(y, p, bins=10):
        q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
        t = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
        return float((t.n * (t.y - t.p).abs()).sum() / t.n.sum())

    def precision_at_k(d, score, ycol, k=K_PER_SHIFT):
        """Với mỗi (ngày, ca) xếp 80 trạm theo score, lấy K trạm đầu, tính tỷ lệ đúng. Hoà điểm phá ngẫu nhiên cố định."""
        t = d[["ngay", "ca", ycol]].copy()
        t["s"] = np.asarray(score, float) + rng_tie.uniform(0, 1e-9, len(t))
        t["r"] = t.groupby(["ngay", "ca"]).s.rank(ascending=False, method="first")
        g = t[t.r <= k].groupby(["ngay", "ca"])[ycol].mean()
        return g

    def metrics(d, ycol, score, is_prob=True):
        y = d[ycol].values
        out = {"PR_AUC": average_precision_score(y, score), "P@K": precision_at_k(d, score, ycol).mean(),
               "Lift@K": precision_at_k(d, score, ycol).mean() / y.mean()}
        if is_prob:
            p = np.clip(score, 1e-6, 1 - 1e-6)
            out.update({"Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p), "ECE": ece(y, p)})
        return out


    #%% ======================================================================
    # 2. Hai mốc đơn giản + chọn siêu tham số trên VALIDATION
    # ========================================================================
    tr, va = blk["train"], blk["validation"]
    rate = {y: tr.groupby(["station_id", "ca"])[y].mean().rename("r").reset_index() for y in LABELS}

    def rate_score(d, y):
        return d.merge(rate[y], on=["station_id", "ca"], how="left").r.fillna(tr[y].mean()).values

    def fit_gbm(y, params, seed):
        m = HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=50,
                                           max_features=0.7, random_state=seed, **params)
        return m.fit(tr[FEATURES], tr[y])

    def fit_logit(y, C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(tr[FEATURES], tr[y])

    grid_gbm = [{"max_depth": d, "max_iter": n} for d in (3, 5) for n in (150, 300)]
    grid_logit = [0.05, 0.5, 5]
    best = {}
    sel_rows = []
    for y in LABELS:
        for i, p in enumerate(grid_gbm):
            ll = log_loss(va[y], np.clip(fit_gbm(y, p, SEED).predict_proba(va[FEATURES])[:, 1], 1e-6, 1 - 1e-6))
            sel_rows.append({"label": y, "model": "GBM", "tham_so": str(p), "val_logloss": ll})
        for C in grid_logit:
            ll = log_loss(va[y], np.clip(fit_logit(y, C).predict_proba(va[FEATURES])[:, 1], 1e-6, 1 - 1e-6))
            sel_rows.append({"label": y, "model": "Logit", "tham_so": f"C={C}", "val_logloss": ll})
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(RESULTS / "model_selection_validation.csv", index=False, encoding="utf-8-sig")
    for y in LABELS:
        s = sel_df[sel_df.label == y]
        best[(y, "GBM")] = grid_gbm[int(s[s.model == "GBM"].reset_index().val_logloss.idxmin())]
        best[(y, "Logit")] = grid_logit[int(s[s.model == "Logit"].reset_index().val_logloss.idxmin())]
    print("[2/6] Siêu tham số chọn theo log loss trên validation:", best)


    #%% ======================================================================
    # 3. Huấn luyện mô hình cuối, so sánh trên VALIDATION, 5 seed, bootstrap theo ngày
    # ========================================================================
    models, pred_va = {}, {}
    for y in LABELS:
        models[(y, "GBM")] = fit_gbm(y, best[(y, "GBM")], SEED)
        models[(y, "Logit")] = fit_logit(y, best[(y, "Logit")])
        pred_va[(y, "GBM")] = models[(y, "GBM")].predict_proba(va[FEATURES])[:, 1]
        pred_va[(y, "Logit")] = models[(y, "Logit")].predict_proba(va[FEATURES])[:, 1]
        pred_va[(y, "Lag5")] = va[BASELINE_LAG[y]].values.astype(float)
        pred_va[(y, "Rate")] = rate_score(va, y)

    rows = []
    for (y, m), s in pred_va.items():
        rows.append({"label": y, "model": m, **metrics(va, y, s, is_prob=m in ("GBM", "Logit", "Rate"))})
    cmp_va = pd.DataFrame(rows).round(4)
    cmp_va.to_csv(RESULTS / "model_comparison_validation.csv", index=False, encoding="utf-8-sig")
    print(cmp_va.to_string(index=False))
    print("      Tỷ lệ sự kiện validation:", {y: round(va[y].mean(), 3) for y in LABELS},
          "| P@K ngẫu nhiên kỳ vọng = tỷ lệ sự kiện")

    seed_rows = []
    for y in LABELS:
        for sd in SEEDS:
            p = fit_gbm(y, best[(y, "GBM")], sd).predict_proba(va[FEATURES])[:, 1]
            seed_rows.append({"label": y, "seed": sd, "PR_AUC": average_precision_score(va[y], p),
                              "P@K": precision_at_k(va, p, y).mean()})
    seed_df = pd.DataFrame(seed_rows)
    seed_sum = seed_df.groupby("label")[["PR_AUC", "P@K"]].agg(["mean", "std"]).round(4)
    seed_df.to_csv(RESULTS / "gbm_seeds_validation.csv", index=False, encoding="utf-8-sig")
    print("      GBM qua 5 seed (validation):"); print(seed_sum.to_string())

    # bootstrap theo ngày: chênh lệch GBM so với từng mốc
    dates = va.ngay.unique()
    day_idx = {d: np.where(va.ngay.values == d)[0] for d in dates}
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for y in LABELS:
        yv = va[y].values
        pk = {m: precision_at_k(va, pred_va[(y, m)], y).reset_index() for m in ["GBM", "Logit", "Lag5", "Rate"]}
        daymean = {m: pk[m].assign(ngay=pd.to_datetime(pk[m]["ngay"])).groupby("ngay")[y].mean() for m in pk}
        res = {(m): {"ap": [], "pk": []} for m in ["GBM", "Logit", "Lag5", "Rate"]}
        for _ in range(N_BOOT):
            sd = rng.choice(dates, size=len(dates), replace=True)
            ii = np.concatenate([day_idx[d] for d in sd])
            w = pd.Series(sd).value_counts()
            for m in res:
                res[m]["ap"].append(average_precision_score(yv[ii], pred_va[(y, m)][ii]))
                res[m]["pk"].append(float((daymean[m].reindex(w.index) * w).sum() / w.sum()))
        for m in ["Logit", "Lag5", "Rate"]:
            for k, nm in [("ap", "PR_AUC"), ("pk", "P@K")]:
                diff = np.array(res["GBM"][k]) - np.array(res[m][k])
                boot_rows.append({"label": y, "so_sanh": f"GBM − {m}", "chi_so": nm,
                                  "chenh_lech_TB": diff.mean(), "CI95_duoi": np.percentile(diff, 2.5), "CI95_tren": np.percentile(diff, 97.5),
                                  "CI_khong_chua_0": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0)})
    boot = pd.DataFrame(boot_rows).round(4)
    boot.to_csv(RESULTS / "bootstrap_gbm_vs_baselines.csv", index=False, encoding="utf-8-sig")
    print("      Bootstrap theo ngày (500 lần), chênh lệch GBM trừ mốc:"); print(boot.to_string(index=False))
    print("[3/6] So sánh mô hình xong")


    #%% ======================================================================
    # 4. Hiệu chỉnh: học trên CALIBRATION, kiểm tra trên POLICY_VAL
    # Platt có ít tham số (hợp với dưới 1.000 mẫu dương); isotonic chỉ báo để đối chiếu. Việc chọn dùng Platt
    # hay giữ xác suất thô theo quy tắc ngoài mẫu ở dưới; quy tắc này chốt sau khi đã xem policy_val (sẽ nêu trong báo cáo).
    # ========================================================================
    ca_, po = blk["calibration"], blk["policy_val"]
    cal_rows, calibrators, pred_po, rule_rows = [], {}, {}, []
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    # Quy tắc chọn (đặt trước khi chạy): chia calibration theo tháng, học Platt trên một tháng, đo ECE trên tháng còn lại
    # (và ngược lại). Chỉ dùng Platt nếu ECE trung bình ngoài mẫu thấp hơn xác suất thô; không thì giữ xác suất thô.
    # Không đo ECE trên chính tập đã học Platt vì khi đó Platt gần như luôn thắng.
    ca_month = ca_.ngay.dt.strftime("%Y-%m").values
    months = sorted(set(ca_month))
    assert len(months) == 2
    for y in LABELS:
        for m in ["GBM", "Logit"]:
            raw_ca = models[(y, m)].predict_proba(ca_[FEATURES])[:, 1]
            raw_po = models[(y, m)].predict_proba(po[FEATURES])[:, 1]
            e_raw, e_platt = [], []
            for a, b in [(months[0], months[1]), (months[1], months[0])]:
                ia, ib = ca_month == a, ca_month == b
                pl = LogisticRegression(C=1e6, max_iter=2000).fit(lg(raw_ca[ia]), ca_[y].values[ia])
                e_raw.append(ece(ca_[y].values[ib], np.clip(raw_ca[ib], 1e-6, 1 - 1e-6)))
                e_platt.append(ece(ca_[y].values[ib], np.clip(pl.predict_proba(lg(raw_ca[ib]))[:, 1], 1e-6, 1 - 1e-6)))
            chosen = "Platt" if np.mean(e_platt) < np.mean(e_raw) else "Chưa hiệu chỉnh"
            rule_rows.append({"label": y, "model": m, "ECE_tho_ngoai_mau": np.mean(e_raw), "ECE_platt_ngoai_mau": np.mean(e_platt), "da_chon": chosen})
            platt = LogisticRegression(C=1e6, max_iter=2000).fit(lg(raw_ca), ca_[y])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_ca, ca_[y])
            calibrators[(y, m)] = {"Platt": platt, "Isotonic": iso, "chosen": chosen}
            versions = {"Chưa hiệu chỉnh": raw_po, "Platt": platt.predict_proba(lg(raw_po))[:, 1], "Isotonic": iso.predict(raw_po)}
            for v, p in versions.items():
                pp = np.clip(p, 1e-6, 1 - 1e-6)
                cal_rows.append({"label": y, "model": m, "phien_ban": v + (" (đã chọn)" if v == chosen else ""),
                                 "Brier": brier_score_loss(po[y], pp), "LogLoss": log_loss(po[y], pp), "ECE": ece(po[y].values, pp),
                                 "TB_xac_suat": pp.mean(), "ty_le_su_kien_thuc": po[y].mean()})
            pred_po[(y, m)] = versions
    cal_df = pd.DataFrame(cal_rows).round(4)
    cal_df.to_csv(RESULTS / "calibration_policy_val.csv", index=False, encoding="utf-8-sig")
    rule_df = pd.DataFrame(rule_rows).round(4)
    rule_df.to_csv(RESULTS / "calibration_rule_choice.csv", index=False, encoding="utf-8-sig")
    print("      Quy tắc chọn bộ hiệu chỉnh (ECE ngoài mẫu, 2 tháng của calibration):"); print(rule_df.to_string(index=False))
    print(f"      Mẫu dương trên calibration: " + str({y: int(ca_[y].sum()) for y in LABELS}))
    print(cal_df.to_string(index=False))
    print("[4/6] Hiệu chỉnh xong")


    #%% ======================================================================
    # 5. Ổn định theo tháng và theo nhóm trạm (dùng chéo validation + calibration + policy_val)
    # ========================================================================
    chk = pd.concat([va, ca_, po]).reset_index(drop=True)
    busy = set(pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str}).head(20).station_id)
    chk["nhom_tram"] = np.where(chk.station_id.isin(busy), "20 trạm đông nhất", "60 trạm còn lại")
    chk["thang_ym"] = chk.ngay.dt.strftime("%Y-%m")
    mrows = []
    for y in LABELS:
        chk[f"{y}_gbm"] = models[(y, "GBM")].predict_proba(chk[FEATURES])[:, 1]
        chk[f"{y}_lag5"] = chk[BASELINE_LAG[y]].values.astype(float)
        chk[f"{y}_rate"] = rate_score(chk, y)
        for tm, g in chk.groupby("thang_ym"):
            for m in ["gbm", "lag5", "rate"]:
                mrows.append({"label": y, "thang": tm, "model": m, "P@K": precision_at_k(g, g[f"{y}_{m}"], y).mean(),
                              "ty_le_su_kien": g[y].mean()})
    mon = pd.DataFrame(mrows).round(4)
    mon.to_csv(RESULTS / "stability_by_month.csv", index=False, encoding="utf-8-sig")
    grp = []
    for y in LABELS:
        for nm, g in chk.groupby("nhom_tram"):
            grp.append({"label": y, "nhom": nm, "ty_le_su_kien": g[y].mean(),
                        "PR_AUC_GBM": average_precision_score(g[y], g[f"{y}_gbm"]),
                        "TyLe_bo_sot_top8": 1 - precision_at_k(g, g[f"{y}_gbm"], y).mean()})
    pd.DataFrame(grp).round(4).to_csv(RESULTS / "error_by_station_group.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(grp).round(3).to_string(index=False))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for j, y in enumerate(LABELS):
        a = ax[0, j]
        for m, nm, c in [("GBM", "GBM", COLOR["gbm"]), ("Logit", "Logistic", COLOR["logit"]), ("Lag5", "Cùng ca tuần trước", COLOR["lag5"]), ("Rate", "Tỷ lệ lịch sử trạm-ca", COLOR["rate"])]:
            pr, rc, _ = precision_recall_curve(va[y], pred_va[(y, m)])
            a.plot(rc, pr, label=nm, color=c)
        a.axhline(va[y].mean(), color="k", ls=":", lw=1, label="Ngẫu nhiên")
        a.set_title(f"Hình 5{'ab'[j]}. Đường PR – {LABELS[y]} (validation)"); a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.legend(fontsize=8)
        b = ax[1, j]
        for m, nm, c in [("gbm", "GBM", COLOR["gbm"]), ("lag5", "Cùng ca tuần trước", COLOR["lag5"]), ("rate", "Tỷ lệ lịch sử", COLOR["rate"])]:
            t = mon[(mon.label == y) & (mon.model == m)]
            b.plot(t.thang, t["P@K"], marker="o", label=nm, color=c)
        t = mon[(mon.label == y) & (mon.model == "gbm")]
        b.plot(t.thang, t.ty_le_su_kien, ls=":", color="k", label="Tỷ lệ sự kiện (ngẫu nhiên)")
        b.set_title(f"Hình 5{'cd'[j]}. Precision@{K_PER_SHIFT} theo tháng – {LABELS[y]}"); b.legend(fontsize=8)
        b.set_xticks(range(len(t))); b.set_xticklabels(t.thang, rotation=45)
    plt.tight_layout(); plt.savefig(RESULTS / "model_hinh5_pr_va_on_dinh.png"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, y in enumerate(LABELS):
        a = ax[j]
        for v, c in [("Chưa hiệu chỉnh", "#8A8A8A"), ("Platt", "#1F3A5F"), ("Isotonic", "#D9822B")]:
            p = pred_po[(y, "GBM")][v]
            q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
            t = pd.DataFrame({"y": po[y].values, "p": p, "b": q}).groupby("b").agg(y=("y", "mean"), p=("p", "mean"))
            a.plot(t.p, t.y, marker="o", label=v, color=c)
        a.plot([0, .5], [0, .5], "k:", lw=1)
        a.set_title(f"Hình 6{'ab'[j]}. Độ tin cậy – {LABELS[y]} (policy_val, GBM)"); a.set_xlabel("Xác suất dự báo"); a.set_ylabel("Tỷ lệ thực"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "model_hinh6_reliability.png"); plt.close()
    print("[5/6] Hình + ổn định xong")


    #%% ======================================================================
    # 6. Lưu mô hình, bộ hiệu chỉnh và dự báo (dùng cho SPO+, MILP, router ở các bước sau)
    # ========================================================================
    joblib.dump({"models": models, "calibrators": calibrators, "features": FEATURES, "best": best,
                 "rate": rate, "k": K_PER_SHIFT}, DATA / "models" / "forecast_bundle.joblib")
    out = chk[["station_id", "ngay", "ca", "khoi", "Y_short", "Y_full", "net_out", "net_in"]].copy()
    for y in LABELS:
        raw = models[(y, "GBM")].predict_proba(chk[FEATURES])[:, 1]
        lg = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        out[f"p_{y}_raw"] = raw
        out[f"p_{y}_platt"] = calibrators[(y, "GBM")]["Platt"].predict_proba(lg)[:, 1]
        out[f"p_{y}_final"] = out[f"p_{y}_platt"] if calibrators[(y, "GBM")]["chosen"] == "Platt" else raw
    out.to_parquet(DATA / "pred_dev.parquet", index=False)
    print(f"[6/6] [✓] BƯỚC 4 hoàn tất sau {time.perf_counter() - t_start:.1f}s")

#%% ======================================================================
def buoc_05_pfi_shap():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 5: Giải thích mô hình bằng PFI và SHAP
    #
    # Mục tiêu: trả lời "mô hình dựa vào cái gì để nói trạm này hôm nay có rủi ro?" và kiểm tra lời giải thích
    # có ổn định theo thời gian không. Đây là giải thích về MÔ HÌNH, không phải nhân quả (không nói
    # "tăng đặc trưng X thì sự kiện tăng").
    # - PFI: xáo trộn một đặc trưng (hoặc một nhóm) trên validation, đo PR-AUC giảm bao nhiêu. Lặp 10 lần.
    #   Các đặc trưng lag / trung bình trượt tương quan mạnh nên PFI từng cột bị chia nhỏ -> làm thêm PFI THEO NHÓM.
    # - Ổn định: so thứ hạng PFI giữa validation và policy_val (Spearman).
    # - SHAP: hướng tác động của từng đặc trưng (mẫu 2.000 dòng validation để chạy nhanh).
    # ========================================================================

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    N_REP = 10
    LABELS = {"Y_short": "thiếu xe", "Y_full": "đầy xe"}
    plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
    FEATURES, models = bundle["features"], bundle["models"]
    df = pd.read_parquet(DATA / "features_dev.parquet")
    va, po = df[df.khoi == "validation"].reset_index(drop=True), df[df.khoi == "policy_val"].reset_index(drop=True)


    def family(f: str) -> str:
        if "_over_tau" in f or f in ("tau_out", "tau_in"): return "Tương đối so với ngưỡng τ"
        if f.startswith("ev_"): return "Tần suất sự kiện gần đây"
        if f.startswith("opp_"): return "Ca đối diện hôm qua"
        if f.startswith("city_"): return "Mức cầu toàn hệ thống"
        if "_lag" in f: return "Lag 1–5 ngày"
        if "_m" in f: return "Trung bình trượt 5–60 ngày"
        if f in ("lat", "lng"): return "Vị trí trạm"
        return "Lịch (thứ, tháng, lễ, ca)"


    fam = {f: family(f) for f in FEATURES}
    groups = {}
    for f, g in fam.items():
        groups.setdefault(g, []).append(f)
    pd.Series(fam, name="nhom").rename_axis("feature").to_csv(RESULTS / "feature_groups.csv", encoding="utf-8-sig")
    print("[0] Nhóm đặc trưng:", {g: len(v) for g, v in groups.items()})


    #%% ======================================================================
    # 1. PFI từng cột và theo nhóm (validation và policy_val)
    # ========================================================================
    def pfi(model, d, y, cols_list, rng):
        base = average_precision_score(d[y], model.predict_proba(d[FEATURES])[:, 1])
        out = []
        for cols in cols_list:
            drops = []
            for _ in range(N_REP):
                x = d[FEATURES].copy()
                perm = rng.permutation(len(x))
                x[cols] = d[cols].values[perm]
                drops.append(base - average_precision_score(d[y], model.predict_proba(x)[:, 1]))
            out.append((np.mean(drops), np.std(drops)))
        return base, out


    rng = np.random.default_rng(SEED)
    rows_i, rows_g = [], []
    for y in LABELS:
        m = models[(y, "GBM")]
        for nm, d in (("validation", va), ("policy_val", po)):
            base, res = pfi(m, d, y, [[f] for f in FEATURES], rng)
            for f, (mu, sd) in zip(FEATURES, res):
                rows_i.append({"label": y, "khoi": nm, "feature": f, "nhom": fam[f], "giam_PR_AUC": mu, "sd": sd, "PR_AUC_goc": base})
            _, resg = pfi(m, d, y, list(groups.values()), rng)
            for g, (mu, sd) in zip(groups, resg):
                rows_g.append({"label": y, "khoi": nm, "nhom": g, "so_cot": len(groups[g]), "giam_PR_AUC": mu, "sd": sd})
    pi = pd.DataFrame(rows_i).round(5)
    pg = pd.DataFrame(rows_g).round(5)
    pi.to_csv(RESULTS / "pfi_tung_cot.csv", index=False, encoding="utf-8-sig")
    pg.to_csv(RESULTS / "pfi_theo_nhom.csv", index=False, encoding="utf-8-sig")
    for y in LABELS:
        print(f"\n  PFI theo nhóm – {LABELS[y]} (validation, PR-AUC giảm khi xáo cả nhóm):")
        print(pg[(pg.label == y) & (pg.khoi == "validation")].sort_values("giam_PR_AUC", ascending=False)
              [["nhom", "so_cot", "giam_PR_AUC", "sd"]].to_string(index=False))
        print(f"  Top 8 cột – {LABELS[y]}:")
        print(pi[(pi.label == y) & (pi.khoi == "validation")].sort_values("giam_PR_AUC", ascending=False).head(8)
              [["feature", "giam_PR_AUC", "sd"]].to_string(index=False))
    print("[1/4] PFI xong")


    #%% ======================================================================
    # 2. Ổn định giữa validation và policy_val
    # ========================================================================
    st = []
    for y in LABELS:
        for lvl, t, key in (("cột", pi, "feature"), ("nhóm", pg, "nhom")):
            a = t[(t.label == y) & (t.khoi == "validation")].set_index(key).giam_PR_AUC
            b = t[(t.label == y) & (t.khoi == "policy_val")].set_index(key).giam_PR_AUC.reindex(a.index)
            top_a, top_b = set(a.nlargest(10 if lvl == "cột" else 3).index), set(b.nlargest(10 if lvl == "cột" else 3).index)
            st.append({"label": y, "muc": lvl, "spearman": spearmanr(a, b)[0], "giao_top": len(top_a & top_b),
                       "so_top": 10 if lvl == "cột" else 3})
    st = pd.DataFrame(st).round(3)
    st.to_csv(RESULTS / "pfi_on_dinh.csv", index=False, encoding="utf-8-sig")
    print("\n  Ổn định thứ hạng validation vs policy_val:"); print(st.to_string(index=False))
    print("[2/4] Ổn định xong")


    #%% ======================================================================
    # 3. SHAP (TreeExplainer trên mẫu validation) – log-odds, hướng tác động
    # ========================================================================
    sample = va.sample(2000, random_state=SEED)
    shap_tab = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, y in enumerate(LABELS):
        ex = shap.TreeExplainer(models[(y, "GBM")])
        sv = ex.shap_values(sample[FEATURES])
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 3: sv = sv[:, :, 1]
        mean_abs = pd.Series(np.abs(sv).mean(0), index=FEATURES)
        corr = pd.Series({f: np.corrcoef(sample[f].values, sv[:, i])[0, 1] for i, f in enumerate(FEATURES)})
        t = pd.DataFrame({"label": y, "feature": FEATURES, "nhom": [fam[f] for f in FEATURES],
                          "mean_abs_shap": mean_abs.values, "huong_tuong_quan": corr.values}).sort_values("mean_abs_shap", ascending=False)
        shap_tab.append(t)
        top = t.head(12).iloc[::-1]
        ax[j].barh(top.feature, top.mean_abs_shap, color="#1F3A5F")
        ax[j].set_title(f"Hình 7{'ab'[j]}. SHAP – {LABELS[y]} (12 đặc trưng mạnh nhất)")
        ax[j].set_xlabel("Trung bình |SHAP| (log-odds)")
    plt.tight_layout(); plt.savefig(RESULTS / "pfi_shap_hinh7_shap.png"); plt.close()
    pd.concat(shap_tab).round(4).to_csv(RESULTS / "shap_tong_hop.csv", index=False, encoding="utf-8-sig")
    for t in shap_tab:
        print(f"\n  SHAP top 8 – {LABELS[t.label.iloc[0]]}:"); print(t.head(8)[["feature", "mean_abs_shap", "huong_tuong_quan"]].round(3).to_string(index=False))

    # hình PFI nhóm
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, y in enumerate(LABELS):
        t = pg[pg.label == y].pivot(index="nhom", columns="khoi", values="giam_PR_AUC").sort_values("validation")
        yy = np.arange(len(t))
        ax[j].barh(yy - .2, t.validation, .4, label="validation", color="#1F3A5F")
        ax[j].barh(yy + .2, t.policy_val, .4, label="policy_val", color="#D9822B")
        ax[j].set_yticks(yy); ax[j].set_yticklabels(t.index, fontsize=8)
        ax[j].set_title(f"Hình 8{'ab'[j]}. PFI theo nhóm – {LABELS[y]}"); ax[j].set_xlabel("PR-AUC giảm khi xáo nhóm"); ax[j].legend()
    plt.tight_layout(); plt.savefig(RESULTS / "pfi_shap_hinh8_pfi_nhom.png"); plt.close()
    print("[3/4] SHAP và hình xong")
    print(f"[4/4] [✓] BƯỚC 5 hoàn tất sau {time.perf_counter() - t_start:.1f}s")

#%% ======================================================================
def buoc_06_nhan_qua_pilot():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 6: Chương nhân quả trung thực – hợp đồng estimand và thiết kế pilot ngẫu nhiên
    #
    # Mục tiêu: dữ liệu KHÔNG ghi chuyến xe tải nào ghé trạm nào (không có nhóm can thiệp), nên không ước lượng được
    # hiệu ứng nhân quả của việc cân bằng xe từ dữ liệu này. Thay vì ngụy tạo, ta (1) viết hợp đồng estimand,
    # (2) thiết kế pilot ngẫu nhiên và (3) tính cần bao nhiêu ngày pilot để phát hiện được một hiệu ứng cho trước,
    # dựa trên xác suất dự báo thật của bước 4 (policy_val). Hiệu ứng giả định chỉ là kịch bản, không phải kết quả.
    # ========================================================================

    t_start = time.perf_counter()
    SEEDV = SEED
    N_SIM = 2000
    CAND = 2 * K_PER_SHIFT                       # ứng viên = 2K trạm xếp hạng cao nhất; ngẫu nhiên K trạm được ghé
    LABELS = {"Y_short": "thiếu xe", "Y_full": "đầy xe"}
    EFFECTS = [0.2, 0.3, 0.5]                    # giảm tương đối xác suất sự kiện (KỊCH BẢN giả định)
    DAYS_GRID = [20, 40, 60, 90, 120, 180]
    plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    pr = pd.read_parquet(DATA / "pred_dev.parquet")
    po = pr[pr.khoi == "policy_val"].copy()


    #%% ======================================================================
    # 1. Hợp đồng estimand (bảng cho báo cáo)
    # ========================================================================
    contract = pd.DataFrame([
        ("Đơn vị phân tích", "Một (trạm, ngày thường, ca) nằm trong tập 2K = 16 ứng viên rủi ro cao nhất của ca đó"),
        ("Can thiệp", "Xe tải ghé trạm trong ca để cân bằng xe (nhận xe nếu thiếu, lấy xe nếu đầy)"),
        ("Đối chứng", "Không ghé trạm trong ca đó (vẫn theo dõi như bình thường)"),
        ("Cách gán", "Ngẫu nhiên hoá trong từng (ngày, ca): K trong 2K ứng viên được ghé, K còn lại là đối chứng; hạt giống ghi lại"),
        ("Estimand", "ATE trên nhóm ứng viên: chênh lệch xác suất xảy ra sự kiện hết/đầy xe giữa được ghé và không được ghé"),
        ("Kết cục đo đúng", "Số phút trạm hết xe (hoặc đầy chỗ) từ nguồn tình trạng trạm theo thời gian thực (cần xác minh có thu thập được)"),
        ("Vì sao KHÔNG dùng Y hiện tại làm kết cục pilot", "Y dựa trên chuyến đã diễn ra (nhu cầu bị cắt cụt): trạm được thêm xe sẽ có thêm chuyến, "
                                                          "làm net_out tăng chứ không giảm, nên Y bị méo theo chiều ngược"),
        ("Giả định cần có", "Không nhiễu lẫn nhờ ngẫu nhiên hoá; không tràn tác động (xe lấy từ trạm này làm trạm khác thiếu) -> "
                           "theo dõi cả trạm cho xe; xe tải thực sự đến đúng trạm; đo kết cục độc lập với việc gán"),
        ("Rủi ro đạo đức/vận hành", "Không dữ liệu cá nhân của người thuê; đối chứng bị bỏ qua chủ yếu là trạm biên (hạng 9–16), không phải trạm rủi ro cao nhất; "
                                  "dừng pilot nếu chỉ số dịch vụ toàn hệ thống xấu đi quá ngưỡng"),
        ("Sau pilot", "Hiệu ứng ước lượng thay cho giả định b (lợi ích mỗi lần ghé) ở bước tối ưu; xem dị biệt theo nhóm trạm; "
                     "giữ nhóm đối chứng cố định 8–10% khi vận hành thật"),
    ], columns=["Mục", "Nội dung"])
    contract.to_csv(RESULTS / "estimand_contract.csv", index=False, encoding="utf-8-sig")
    print("[1/4] Hợp đồng estimand xong"); print(contract.to_string(index=False, max_colwidth=90))


    #%% ======================================================================
    # 2. Tỷ lệ nền của nhóm ứng viên và chi phí cơ hội của việc ngẫu nhiên hoá (policy_val)
    # ========================================================================
    def groups(y):
        """Trả về ma trận (số ô ngày-ca) x CAND: giá trị Y của 2K ứng viên cao nhất, xếp theo xác suất cuối cùng."""
        t = po[["ngay", "ca", y, f"p_{y}_final"]].copy()
        t["r"] = t.groupby(["ngay", "ca"])[f"p_{y}_final"].rank(ascending=False, method="first")
        t = t[t.r <= CAND].sort_values(["ngay", "ca", "r"])
        return t.groupby(["ngay", "ca"])[y].apply(lambda s: s.values).pipe(lambda s: np.vstack(s.values))

    base_rows, mats = [], {}
    for y in LABELS:
        M = groups(y); mats[y] = M
        top_k, top_2k = M[:, :K_PER_SHIFT].mean(), M.mean()
        base_rows.append({"label": y, "so_o_ngay_ca": len(M), "ty_le_su_kien_toan_bo": po[y].mean(),
                          "ty_le_top_K": top_k, "ty_le_top_2K": top_2k,
                          "chi_phi_co_hoi_su_kien_moi_ca": (top_k - top_2k) * K_PER_SHIFT})
    base = pd.DataFrame(base_rows).round(4)
    base.to_csv(RESULTS / "pilot_ty_le_nen.csv", index=False, encoding="utf-8-sig")
    print("\n[2/4] Tỷ lệ nền và chi phí cơ hội của ngẫu nhiên hoá (policy_val):"); print(base.to_string(index=False))


    #%% ======================================================================
    # 3. Mô phỏng sức mạnh kiểm định: cần bao nhiêu ngày pilot?
    # Mỗi lần mô phỏng: lấy mẫu có hoàn lại các ô (ngày, ca) thật của policy_val -> ngẫu nhiên gán K/2K ứng viên được ghé ->
    # với ứng viên được ghé, xoá sự kiện với xác suất = hiệu ứng tương đối -> tính chênh lệch từng ô (cụm theo ô),
    # kiểm định t một mẫu trên các ô (không coi từng trạm là độc lập).
    # ========================================================================
    rng = np.random.default_rng(SEEDV)

    def power(M, n_days, eff):
        n_cells = n_days * 2
        rej = 0
        for _ in range(N_SIM):
            ii = rng.integers(0, len(M), n_cells)
            Y = M[ii].copy()
            order = np.argsort(rng.random(Y.shape), axis=1)
            treat = np.zeros_like(Y, dtype=bool)
            np.put_along_axis(treat, order[:, :K_PER_SHIFT], True, axis=1)
            flip = (rng.random(Y.shape) < eff) & (Y == 1) & treat
            Y[flip] = 0
            d = (Y * treat).sum(1) / K_PER_SHIFT - (Y * ~treat).sum(1) / K_PER_SHIFT
            t, p = stats.ttest_1samp(d, 0.0)
            rej += (p < 0.05) and (d.mean() < 0)
        return rej / N_SIM

    rows = []
    for y in LABELS:
        for eff in EFFECTS:
            for nd in DAYS_GRID:
                rows.append({"label": y, "hieu_ung_tuong_doi": eff, "so_ngay": nd, "suc_manh": power(mats[y], nd, eff)})
    pw = pd.DataFrame(rows).round(3)
    pw.to_csv(RESULTS / "pilot_suc_manh.csv", index=False, encoding="utf-8-sig")
    need = (pw[pw.suc_manh >= 0.8].groupby(["label", "hieu_ung_tuong_doi"]).so_ngay.min().unstack().reindex(columns=EFFECTS))
    need.to_csv(RESULTS / "pilot_so_ngay_can.csv", encoding="utf-8-sig")
    print("\n[3/4] Số ngày làm việc tối thiểu để đạt sức mạnh 80% (NaN = chưa đạt trong 180 ngày):"); print(need.to_string())

    # tỷ lệ I (sai dương): hiệu ứng = 0 -> phải xấp xỉ 5% một phía
    fp = {y: power(mats[y], 60, 0.0) for y in LABELS}
    print("      Kiểm tra: tỷ lệ bác bỏ khi KHÔNG có hiệu ứng (60 ngày, kỳ vọng ~0,025 vì một phía):", {k: round(v, 3) for k, v in fp.items()})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, y in enumerate(LABELS):
        for eff, c in zip(EFFECTS, ["#8A8A8A", "#2E8B57", "#1F3A5F"]):
            t = pw[(pw.label == y) & (pw.hieu_ung_tuong_doi == eff)]
            ax[j].plot(t.so_ngay, t.suc_manh, marker="o", color=c, label=f"giảm {int(eff*100)}% sự kiện")
        ax[j].axhline(0.8, color="k", ls=":", lw=1)
        ax[j].set_title(f"Hình 9{'ab'[j]}. Sức mạnh pilot theo số ngày – {LABELS[y]}")
        ax[j].set_xlabel("Số ngày làm việc pilot"); ax[j].set_ylabel("Sức mạnh kiểm định"); ax[j].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "nhanqua_hinh9_suc_manh_pilot.png"); plt.close()
    print(f"[4/4] [✓] BƯỚC 6 (nhân quả) hoàn tất sau {time.perf_counter() - t_start:.1f}s")

#%% ======================================================================
def buoc_07_toi_uu():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 7: Từ xác suất đến quyết định – SPO+ , Greedy, MILP, độ nhạy, ROI
    #
    # Bài toán mỗi (ngày, ca): chọn tối đa K trong 80 trạm để xe tải ghé, tối đa hoá giá trị kỳ vọng
    #   v_i = p_thiếu_i · b_thiếu + p_đầy_i · b_đầy   (mỗi trạm tối đa một lần/ca).
    # Ràng buộc thực tế: 80 trạm chia 6 khu vực; mỗi trạm tốn s phút, mỗi khu vực có trạm được ghé tốn thêm F phút vào khu;
    # tổng thời gian ca <= B. (Chỉ có ràng buộc "tối đa K" thì Greedy đã tối ưu; ràng buộc khu vực mới làm MILP có việc để làm.)
    # Mọi hằng số vận hành là GIẢ ĐỊNH và có phân tích độ nhạy.
    # Vai trò dữ liệu: SPO+ học trên TRAIN (xác suất out-of-fold), chọn checkpoint trên VALIDATION,
    # so sánh chính sách trên POLICY_VAL. Không đụng locked/oot.
    # ========================================================================

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    N_ZONES = 6
    S_MIN, F_MIN, B_MIN = 10, 20, 140          # phút/trạm, phút vào khu, ngân sách phút mỗi ca (giả định)
    B_FULL = 0.7                               # giá trị một sự kiện đầy xe so với một sự kiện thiếu xe (giả định)
    SCEN_K, SCEN_W = [6, 8, 10], [0.25, 0.5, 0.25]   # kịch bản năng lực xe tải để chọn checkpoint
    EFF, COST_MIN, RIDE_VALUE = 0.3, 1.5, 3.0  # hiệu ứng ghé (khớp kịch bản pilot), $/phút xe tải, $/chuyến mất (giả định)
    N_BOOT = 1000
    COLOR = {"Greedy": "#8A8A8A", "MILP": "#1F3A5F", "Ridge": "#2E8B57", "SPO+": "#D9822B", "Random": "#C0C0C0"}
    plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str}).sort_values("station_id").reset_index(drop=True)
    stations = sel.station_id.tolist()
    zone = KMeans(N_ZONES, n_init=10, random_state=SEED).fit_predict(sel[["lat", "lng"]].values)
    print("[0] Số trạm mỗi khu vực:", np.bincount(zone).tolist())


    #%% ======================================================================
    # 1. Dữ liệu theo ô (ngày, ca) x 80 trạm. Xác suất train lấy out-of-fold (5 khối thời gian) để SPO+ không học
    #    trên xác suất đã "thuộc bài".
    # ========================================================================
    bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
    FEATURES, best = bundle["features"], bundle["best"]
    feat = pd.read_parquet(DATA / "features_dev.parquet")
    pred = pd.read_parquet(DATA / "pred_dev.parquet")
    LAB = ["Y_short", "Y_full"]

    tr = feat[feat.khoi == "train"].sort_values(["ngay", "ca", "station_id"]).reset_index(drop=True)
    days_tr = np.sort(tr.ngay.unique())
    fold_of_day = dict(zip(days_tr, np.array_split(np.arange(len(days_tr)), 5)))
    fold_id = np.zeros(len(tr), int)
    for k, idx in enumerate(np.array_split(np.arange(len(days_tr)), 5)):
        fold_id[tr.ngay.isin(days_tr[idx]).values] = k
    oof = {y: np.zeros(len(tr)) for y in LAB}
    for y in LAB:
        for k in range(5):
            m = HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=50, max_features=0.7,
                                               random_state=SEED, **best[(y, "GBM")]).fit(tr.loc[fold_id != k, FEATURES], tr.loc[fold_id != k, y])
            oof[y][fold_id == k] = m.predict_proba(tr.loc[fold_id == k, FEATURES])[:, 1]
    tr["p_Y_short_raw"], tr["p_Y_full_raw"] = oof["Y_short"], oof["Y_full"]
    print("[1a] Xác suất out-of-fold cho train xong")


    def to_cells(d):
        d = d.sort_values(["ngay", "ca", "station_id"]).reset_index(drop=True)
        assert (d.groupby(["ngay", "ca"]).size() == len(stations)).all() and (d.station_id.to_numpy(dtype=object).reshape(-1, len(stations)) == np.array(stations)).all()
        sh = (-1, len(stations))
        out = {"ngay": d.ngay.values.reshape(sh)[:, 0], "ca": d.ca.to_numpy(dtype=object).reshape(sh)[:, 0]}
        for c in ["Y_short", "Y_full", "p_Y_short_raw", "p_Y_full_raw"] + [c for c in d.columns if c.startswith("p_Y_") and c.endswith(("_final", "_platt"))] + ["net_out", "net_in"]:
            if c in d.columns: out[c] = d[c].values.reshape(sh)
        return out


    C = {"train": to_cells(tr[["ngay", "ca", "station_id", "Y_short", "Y_full", "net_out", "net_in", "p_Y_short_raw", "p_Y_full_raw"]]),
         "validation": to_cells(pred[pred.khoi == "validation"]),
         "policy_val": to_cells(pred[pred.khoi == "policy_val"])}
    print("[1b] Số ô (ngày-ca):", {k: len(v["ngay"]) for k, v in C.items()})
    tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})
    print(f"      Bối cảnh: τ_out trung vị {tau.tau_out.median():.0f} xe, τ_in trung vị {tau.tau_in.median():.0f} xe (một sự kiện là ròng đi/đến >= τ).")


    #%% ======================================================================
    # 2. Bộ giải: Greedy và MILP (scipy.milp / HiGHS)
    # ========================================================================
    n = len(stations)
    Zmat = np.zeros((n, N_ZONES)); Zmat[np.arange(n), zone] = 1


    def cost_min(z, s=S_MIN, F=F_MIN):
        return s * z.sum() + F * (Zmat[z.astype(bool)].sum(0) > 0).sum()


    def solve_milp(v, K=K_PER_SHIFT, B=B_MIN, F=F_MIN, s=S_MIN, gap=0.0):
        nv = n + N_ZONES
        c = np.concatenate([-v, np.zeros(N_ZONES)])
        A = [np.concatenate([np.ones(n), np.zeros(N_ZONES)]), np.concatenate([np.full(n, s), np.full(N_ZONES, F)])]
        ub = [K, B]
        link = np.hstack([np.eye(n), -Zmat])               # x_i - y_zone(i) <= 0
        cons = [LinearConstraint(np.array(A), -np.inf, ub), LinearConstraint(link, -np.inf, 0)]
        r = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1), options={"mip_rel_gap": gap})
        z = np.zeros(n, bool)
        if r.x is not None: z = r.x[:n] > 0.5
        return z & (v > 0)


    def solve_greedy(v, K=K_PER_SHIFT, B=B_MIN, F=F_MIN, s=S_MIN):
        z = np.zeros(n, bool); used = np.zeros(N_ZONES, bool); mins = 0
        for i in np.argsort(-v):
            if v[i] <= 0 or z.sum() >= K: break
            add = s + (0 if used[zone[i]] else F)
            if mins + add <= B:
                z[i] = True; used[zone[i]] = True; mins += add
        return z


    def solve_all(V, solver, **kw):
        return np.array([solver(V[i], **kw) for i in range(len(V))])


    def value_matrix(cells, ps, pf, bf=B_FULL):
        return cells[ps] * 1.0 + cells[pf] * bf


    def realized(cells, bf=B_FULL):
        return cells["Y_short"] * 1.0 + cells["Y_full"] * bf


    def regret_per_cell(Z, R, Zor):
        return 1 - (Z * R).sum(1) / np.maximum((Zor * R).sum(1), 1e-9)


    #%% ======================================================================
    # 3. Mô hình giá trị: Ridge (hai bước) và SPO+ (học trực tiếp để quyết định đúng)
    # Đặc trưng mỗi trạm-ô: p_thiếu, p_đầy (GBM thô), độ lệch so với trung bình của ô, khu vực (one-hot).
    # SPO+: gradient = 2·(z_đối_nghịch − z_thật) với z_đối_nghịch = argmax (2ĉ − c)·z (giải bằng chính MILP).
    # ========================================================================
    def design(cells):
        ps, pf = cells["p_Y_short_raw"], cells["p_Y_full_raw"]
        X = np.stack([ps, pf, ps - ps.mean(1, keepdims=True), pf - pf.mean(1, keepdims=True)] +
                     [np.broadcast_to(Zmat[:, k], ps.shape) for k in range(N_ZONES - 1)], axis=-1)
        return X


    Xtr, Xva = design(C["train"]), design(C["validation"])
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-9
    nz = lambda X: (X - mu) / sd
    Rtr, Rva = realized(C["train"]), realized(C["validation"])
    ORtr = solve_all(Rtr, solve_milp)                  # nghiệm oracle trên train (K, B mặc định)
    ORva = {K: solve_all(Rva, solve_milp, K=K) for K in SCEN_K}


    def val_regret(theta, X=None):
        """Regret trung bình có trọng số 25/50/25 trên validation qua ba kịch bản năng lực."""
        X = Xva if X is None else X
        V = nz(X) @ theta[:-1] + theta[-1]
        tot = 0.0
        for K, w in zip(SCEN_K, SCEN_W):
            tot += w * regret_per_cell(solve_all(V, solve_milp, K=K), Rva, ORva[K]).mean()
        return tot


    t0 = time.perf_counter()
    Xf, yf = nz(Xtr).reshape(-1, Xtr.shape[-1]), Rtr.reshape(-1)
    ridge_rows, ridge_models = [], {}
    for a in (1, 10, 100, 1000, 10000):
        rm = Ridge(alpha=a).fit(Xf, yf)
        th = np.concatenate([rm.coef_, [rm.intercept_]]); ridge_models[a] = th
        ridge_rows.append({"alpha": a, "regret_validation": val_regret(th)})
    ridge_df = pd.DataFrame(ridge_rows)
    a_best = int(ridge_df.loc[ridge_df.regret_validation.idxmin(), "alpha"]); theta_ridge = ridge_models[a_best]
    print("[3a] Ridge: regret validation theo alpha"); print(ridge_df.round(4).to_string(index=False), f"-> chọn alpha={a_best}")

    theta = theta_ridge.copy(); m_, v_ = np.zeros_like(theta), np.zeros_like(theta)
    STEPS, BATCH, LR, CK = 120, 64, 0.02, 10
    ck = [(0, val_regret(theta), theta.copy())]
    Xn = nz(Xtr)
    for step in range(1, STEPS + 1):
        ib = rng.choice(len(Xn), BATCH, replace=False)
        chat = Xn[ib] @ theta[:-1] + theta[-1]
        Zadv = solve_all(2 * chat - Rtr[ib], solve_milp)
        g_c = 2 * (Zadv.astype(float) - ORtr[ib].astype(float))          # gradient theo ĉ (bài toán cực đại hoá)
        g = np.concatenate([np.einsum("bn,bnd->d", g_c, Xn[ib]), [g_c.sum()]]) / (BATCH * n)
        m_ = 0.9 * m_ + 0.1 * g; v_ = 0.999 * v_ + 0.001 * g * g
        theta = theta - LR * (m_ / (1 - 0.9 ** step)) / (np.sqrt(v_ / (1 - 0.999 ** step)) + 1e-8)
        if step % CK == 0: ck.append((step, val_regret(theta), theta.copy()))
    ck_df = pd.DataFrame([(s, r) for s, r, _ in ck], columns=["buoc", "regret_validation"])
    ck_df.to_csv(RESULTS / "spo_checkpoint_validation.csv", index=False, encoding="utf-8-sig")
    step_best, _, theta_spo = min(ck, key=lambda t: t[1])
    print("[3b] SPO+ checkpoint (regret validation 25/50/25):"); print(ck_df.round(4).to_string(index=False), f"-> chọn bước {step_best}")
    print(f"      Thời gian huấn luyện SPO+/Ridge: {time.perf_counter() - t0:.0f}s")


    #%% ======================================================================
    # 4. So sánh chính sách trên POLICY_VAL (K, B mặc định)
    # ========================================================================
    P = C["policy_val"]
    Rpo = realized(P)
    ZOR = solve_all(Rpo, solve_milp)
    Vdir_final = value_matrix(P, "p_Y_short_final", "p_Y_full_final")
    Vdir_raw = value_matrix(P, "p_Y_short_raw", "p_Y_full_raw")
    Xpo = nz(design(P))
    V_ridge, V_spo = Xpo @ theta_ridge[:-1] + theta_ridge[-1], Xpo @ theta_spo[:-1] + theta_spo[-1]
    V_rand = rng.random(Vdir_raw.shape)
    V_lag5 = P["net_out"] * 0                                       # chỗ giữ: Lag5 tính bên dưới
    feat_po = feat[feat.khoi == "policy_val"].sort_values(["ngay", "ca", "station_id"]).reset_index(drop=True)
    V_lag5 = (feat_po.Y_short.values * 0 + feat_po.net_out_lag5.clip(lower=0).values / tau.tau_out.median()).reshape(-1, n) \
             + B_FULL * (feat_po.net_in_lag5.clip(lower=0).values / tau.tau_in.median()).reshape(-1, n)

    policies = {"Cùng ca tuần trước + MILP": (V_lag5, solve_milp), "Ngẫu nhiên + MILP": (V_rand, solve_milp),
                "GBM + Greedy": (Vdir_final, solve_greedy), "GBM + MILP": (Vdir_final, solve_milp),
                "Ridge + MILP": (V_ridge, solve_milp), "SPO+ + MILP": (V_spo, solve_milp)}
    res, Zs, tms = [], {}, {}
    for name, (V, fn) in policies.items():
        t1 = time.perf_counter(); Z = solve_all(V, fn); tms[name] = (time.perf_counter() - t1) / len(V) * 1000
        Zs[name] = Z
        rg = regret_per_cell(Z, Rpo, ZOR)
        res.append({"chinh_sach": name, "regret_TB_%": 100 * rg.mean(), "gia_tri_thuc_TB_moi_o": (Z * Rpo).sum(1).mean(),
                    "so_tram_TB": Z.sum(1).mean(), "so_khu_TB": np.mean([(Zmat[z].sum(0) > 0).sum() for z in Z]),
                    "phut_TB": np.mean([cost_min(z) for z in Z]), "ms_moi_ca_giai": tms[name]})
    res.append({"chinh_sach": "Oracle (biết kết quả)", "regret_TB_%": 0.0, "gia_tri_thuc_TB_moi_o": (ZOR * Rpo).sum(1).mean(),
                "so_tram_TB": ZOR.sum(1).mean(), "so_khu_TB": np.mean([(Zmat[z].sum(0) > 0).sum() for z in ZOR]),
                "phut_TB": np.mean([cost_min(z) for z in ZOR]), "ms_moi_ca_giai": np.nan})
    cmp_df = pd.DataFrame(res).round(3)
    cmp_df.to_csv(RESULTS / "toi_uu_so_sanh_policy_val.csv", index=False, encoding="utf-8-sig")
    print("\n[4a] So sánh chính sách (policy_val, K=%d, B=%d phút):" % (K_PER_SHIFT, B_MIN)); print(cmp_df.to_string(index=False))

    # bootstrap theo ngày: chênh lệch giá trị thực giữa các cặp chính sách
    day = pd.Series(P["ngay"]); udays = day.unique(); di = {d: np.where(P["ngay"] == d)[0] for d in udays}
    val = {k: (Z * Rpo).sum(1) for k, Z in Zs.items()}
    pairs = [("GBM + MILP", "GBM + Greedy"), ("SPO+ + MILP", "Ridge + MILP"), ("SPO+ + MILP", "GBM + MILP"),
             ("Ridge + MILP", "GBM + MILP"), ("GBM + MILP", "Ngẫu nhiên + MILP"), ("GBM + MILP", "Cùng ca tuần trước + MILP")]
    brow = []
    for a, b in pairs:
        d = val[a] - val[b]; per_day = pd.Series(d).groupby(P["ngay"]).mean()
        bs = [per_day.reindex(rng.choice(udays, len(udays))).mean() for _ in range(N_BOOT)]
        brow.append({"A": a, "B": b, "chenh_gia_tri_TB": d.mean(), "CI95_duoi": np.percentile(bs, 2.5), "CI95_tren": np.percentile(bs, 97.5),
                     "CI_khong_chua_0": bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0)})
    boot = pd.DataFrame(brow).round(4)
    boot.to_csv(RESULTS / "toi_uu_bootstrap.csv", index=False, encoding="utf-8-sig")
    print("\n[4b] Bootstrap theo ngày (A trừ B, giá trị thực mỗi ô):"); print(boot.to_string(index=False))

    # MILP so với Greedy trên giá trị DỰ BÁO (MILP phải >= Greedy) và mip_gap
    pg_rows = []
    for gap in (0.0, 0.01, 0.05):
        t1 = time.perf_counter(); Zg = solve_all(Vdir_final, solve_milp, gap=gap); dt = (time.perf_counter() - t1) / len(Zg) * 1000
        pg_rows.append({"mip_gap": gap, "gia_tri_du_bao_TB": (Zg * Vdir_final).sum(1).mean(), "gia_tri_thuc_TB": (Zg * Rpo).sum(1).mean(), "ms_moi_ca": dt})
    Zgr = Zs["GBM + Greedy"]
    pg_rows.append({"mip_gap": "Greedy", "gia_tri_du_bao_TB": (Zgr * Vdir_final).sum(1).mean(), "gia_tri_thuc_TB": (Zgr * Rpo).sum(1).mean(), "ms_moi_ca": tms["GBM + Greedy"]})
    gapdf = pd.DataFrame(pg_rows).round(4); gapdf.to_csv(RESULTS / "toi_uu_mip_gap.csv", index=False, encoding="utf-8-sig")
    diff_cells = (Zs["GBM + MILP"] != Zgr).any(1).mean()
    print("\n[4c] mip_gap và Greedy:"); print(gapdf.to_string(index=False)); print(f"      Tỷ lệ ô mà MILP chọn khác Greedy: {diff_cells:.1%}")


    #%% ======================================================================
    # 5. Độ nhạy 5 trục (một trục thay đổi, các trục còn lại mặc định) – GBM + Greedy vs MILP trên policy_val
    # ========================================================================
    def run_scenario(K=K_PER_SHIFT, B=B_MIN, F=F_MIN, bf=B_FULL, src="final"):
        Rr = realized(P, bf); Zo = solve_all(Rr, solve_milp, K=K, B=B, F=F)
        V = value_matrix(P, f"p_Y_short_{src}", f"p_Y_full_{src}", bf)
        o = {}
        for nm, fn in (("Greedy", solve_greedy), ("MILP", solve_milp)):
            Z = solve_all(V, fn, K=K, B=B, F=F) if fn is solve_milp else solve_all(V, fn, K=K, B=B, F=F)
            o[nm] = (100 * regret_per_cell(Z, Rr, Zo).mean(), (Z * Rr).sum(1).mean(), np.mean([cost_min(z, F=F) for z in Z]))
        return o

    axes = {"K (trạm/ca)": ("K", [4, 6, 8, 10, 12]), "Ngân sách B (phút)": ("B", [100, 120, 140, 170, 230]),
            "Phí vào khu F (phút)": ("F", [10, 20, 30, 40]), "Giá trị đầy/thiếu (b_đầy)": ("bf", [0.5, 0.7, 1.0]),
            "Nguồn xác suất": ("src", ["raw", "final", "platt"])}
    srows = []
    for an, (key, vals) in axes.items():
        for vv in vals:
            o = run_scenario(**{key: vv})
            for nm in ("Greedy", "MILP"):
                srows.append({"truc": an, "gia_tri": vv, "chinh_sach": nm, "regret_%": o[nm][0], "gia_tri_thuc_TB": o[nm][1], "phut_TB": o[nm][2]})
    sens = pd.DataFrame(srows).round(3)
    sens.to_csv(RESULTS / "toi_uu_do_nhay.csv", index=False, encoding="utf-8-sig")
    pv = sens.pivot_table(index=["truc", "gia_tri"], columns="chinh_sach", values="regret_%", sort=False)
    pv["MILP_hon_Greedy_(diem_%)"] = pv["Greedy"] - pv["MILP"]
    print("\n[5] Độ nhạy – regret % (thấp hơn là tốt):"); print(pv.round(2).to_string())


    #%% ======================================================================
    # 6. ROI (giả định: hiệu ứng ghé 30%, 1,5 $/phút xe tải, 3 $/chuyến mất) và điểm hoà vốn
    # Hoà vốn: giá trị tối thiểu của MỘT sự kiện (tính theo số chuyến bị mất) để lợi ích >= chi phí.
    # ========================================================================
    roi_rows = []
    for name in ["Ngẫu nhiên + MILP", "GBM + Greedy", "GBM + MILP", "SPO+ + MILP"]:
        Z = Zs[name]; units = (Z * Rpo).sum(1).mean(); mins = np.mean([cost_min(z) for z in Z]); cost = mins * COST_MIN
        vstar = cost / (EFF * units) if units > 0 else np.nan
        roi_rows.append({"chinh_sach": name, "su_kien_ky_vong_moi_ca": units, "phut_moi_ca": mins, "chi_phi_$_moi_ca": cost,
                         "hoa_von_$_moi_su_kien": vstar, "hoa_von_so_chuyen_mat_moi_su_kien": vstar / RIDE_VALUE})
    roi = pd.DataFrame(roi_rows).round(2)
    dG = (Zs["GBM + MILP"] * Rpo).sum(1).mean() - (Zs["GBM + Greedy"] * Rpo).sum(1).mean()
    dC = (np.mean([cost_min(z) for z in Zs["GBM + MILP"]]) - np.mean([cost_min(z) for z in Zs["GBM + Greedy"]])) * COST_MIN
    roi.to_csv(RESULTS / "toi_uu_roi.csv", index=False, encoding="utf-8-sig")
    print("\n[6] ROI / hoà vốn (so với KHÔNG cân bằng = 0 lợi ích, 0 chi phí):"); print(roi.to_string(index=False))
    print(f"      MILP so với Greedy: Δ sự kiện/ca = {dG:+.3f}, Δ chi phí = {dC:+.2f} $/ca")

    # hình
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    order = ["Cùng ca tuần trước + MILP", "Ngẫu nhiên + MILP", "GBM + Greedy", "GBM + MILP", "Ridge + MILP", "SPO+ + MILP"]
    t = cmp_df.set_index("chinh_sach").loc[order]
    ax[0].barh(range(len(t)), t["regret_TB_%"], color=["#C0C0C0", "#C0C0C0", COLOR["Greedy"], COLOR["MILP"], COLOR["Ridge"], COLOR["SPO+"]])
    ax[0].set_yticks(range(len(t))); ax[0].set_yticklabels(order, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel("Regret trung bình (%) – thấp hơn là tốt"); ax[0].set_title("Hình 10a. Regret các chính sách (policy_val)")
    for key, ls in (("Ngân sách B (phút)", "-"),):
        s_ = sens[sens.truc == key]
        for nm in ("Greedy", "MILP"):
            q = s_[s_.chinh_sach == nm]; ax[1].plot(q.gia_tri, q["regret_%"], marker="o", ls=ls, color=COLOR[nm], label=nm)
    ax[1].set_xlabel("Ngân sách thời gian mỗi ca B (phút)"); ax[1].set_ylabel("Regret (%)"); ax[1].legend()
    ax[1].set_title("Hình 10b. MILP so với Greedy khi ngân sách chặt/lỏng")
    plt.tight_layout(); plt.savefig(RESULTS / "toi_uu_hinh10_regret.png"); plt.close()

    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True)
    for a, (an, _) in zip(ax, axes.items()):
        s_ = sens[sens.truc == an]
        for nm in ("Greedy", "MILP"):
            q = s_[s_.chinh_sach == nm]; a.plot(q.gia_tri.astype(str), q["regret_%"], marker="o", color=COLOR[nm], label=nm)
        a.set_title(an, fontsize=9)
    ax[0].set_ylabel("Regret (%)"); ax[0].legend(fontsize=8)
    plt.suptitle("Hình 11. Độ nhạy 5 trục (policy_val)"); plt.tight_layout(); plt.savefig(RESULTS / "toi_uu_hinh11_do_nhay.png"); plt.close()

    joblib.dump({"theta_spo": theta_spo, "theta_ridge": theta_ridge, "mu": mu, "sd": sd, "zone": zone, "params": dict(S=S_MIN, F=F_MIN, B=B_MIN, K=K_PER_SHIFT, b_full=B_FULL)},
                DATA / "models" / "decision_bundle.joblib")
    print(f"[✓] BƯỚC 7 hoàn tất sau {time.perf_counter() - t_start:.0f}s")

#%% ======================================================================
def buoc_08_router():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 8: Action Router – biến kế hoạch tối ưu thành lệnh có kiểm soát
    #
    # Router nhận xác suất + kết quả tối ưu tại t0 = 22:00 ngày D−1 và phát ra danh sách trạm cho xe tải.
    # Ba lớp guardrail:
    #   Lớp 1 (cứng, KHÔNG ai được ghi đè): trạm hợp lệ (có hoạt động gần đây), tối đa K trạm, tổng phút <= B,
    #          mỗi trạm một lần/ca, nhóm CONTROL không được chuyển sang can thiệp.
    #   Lớp 2 (mềm, người điều phối ĐƯỢC ghi đè kèm lý do): điểm giá trị tối thiểu, mỗi trạm tối đa 3 lần/5 ngày làm việc.
    #   Lớp 3 (hệ thống): PSI của điểm dự báo (cửa sổ 5 ngày) >= 0,10 -> cảnh báo; chuyển sang chính sách dự phòng
    #          (cùng ca tuần trước + Greedy) CHỈ KHI PSI >= 0,25 VÀ kết quả thực 10 ngày gần nhất (đã biết tại t0) của mô hình
    #          kém hơn quy tắc dự phòng. Bản đầu chỉ dùng PSI làm 36/88 ô chuyển nhầm (mùa khác chứ không phải mô hình hỏng).
    # CONTROL: băm cố định (sha256) của "muối|trạm|ngày|ca", 10% trạm được chọn bị giữ lại không ghé để đo hiệu ứng thật.
    # Idempotency: khoá = chính_sách|trạm|ngày|ca, chạy lại không sinh lệnh trùng.
    # Chạy trên policy_val; chưa đụng locked/oot.
    # ========================================================================
    nap_bo_giai()

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    POLICY_ID, CONTROL_SALT, CONTROL_PCT = "bike-rebalance-v1", "ctl-v1", 10
    V_MIN, MAX_VISITS_5D, PSI_WARN, PSI_STOP = 0.10, 3, 0.10, 0.25
    CA_ORDER = {"sang": 0, "chieu": 1}
    sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str}).sort_values("station_id").reset_index(drop=True)
    stations = sel.station_id.tolist(); N = len(stations)
    tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})
    TAU_OUT_MED, TAU_IN_MED = tau.tau_out.median(), tau.tau_in.median()

    def file_hash(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    MODEL_VERSION = file_hash(DATA / "models" / "forecast_bundle.joblib") + "+" + file_hash(DATA / "models" / "decision_bundle.joblib")

    pr = pd.read_parquet(DATA / "pred_dev.parquet")
    lagc = [f"n_out_lag{k}" for k in range(1, 6)] + [f"n_in_lag{k}" for k in range(1, 6)]
    ft = pd.read_parquet(DATA / "features_dev.parquet")[["station_id", "ngay", "ca", "net_out_lag5", "net_in_lag5"] + lagc]
    d = pr.merge(ft, on=["station_id", "ngay", "ca"], how="left")
    d["ca_ord"] = d.ca.map(CA_ORDER)
    d = d.sort_values(["ngay", "ca_ord", "station_id"]).reset_index(drop=True)
    assert (d.groupby(["ngay", "ca"]).size() == N).all()
    d["hop_le"] = d[lagc].sum(axis=1) > 0
    d["v"] = d.p_Y_short_final + B_FULL * d.p_Y_full_final
    d["v_du_phong"] = d.net_out_lag5.clip(lower=0) / TAU_OUT_MED + B_FULL * d.net_in_lag5.clip(lower=0) / TAU_IN_MED
    print(f"[0] Phiên bản chính sách {POLICY_ID} | mô hình {MODEL_VERSION} | {d.groupby(['ngay','ca']).ngroups} ô dev")


    #%% ======================================================================
    # 1. PSI của điểm dự báo (lớp 3): tham chiếu = validation, cửa sổ = 5 ngày làm việc gần nhất đến hết ngày D
    # ========================================================================
    def psi(ref, cur, bins=10):
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        a = np.histogram(ref, edges)[0] / len(ref); b = np.histogram(cur, edges)[0] / len(cur)
        a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
        return float(((b - a) * np.log(b / a)).sum())

    ref = {y: d[d.khoi == "validation"][f"p_{y}_final"].values for y in ("Y_short", "Y_full")}
    days_all = np.sort(d.ngay.unique())
    psi_rows = []
    for i, day in enumerate(days_all):
        w = d[d.ngay.isin(days_all[max(0, i - 4): i + 1])]
        if len(days_all[max(0, i - 4): i + 1]) < 5: continue
        psi_rows.append({"ngay": day, "khoi": d[d.ngay == day].khoi.iloc[0],
                         "psi_short": psi(ref["Y_short"], w.p_Y_short_final.values), "psi_full": psi(ref["Y_full"], w.p_Y_full_final.values)})
    psi_df = pd.DataFrame(psi_rows); psi_df["psi_max"] = psi_df[["psi_short", "psi_full"]].max(axis=1)
    psi_df.to_csv(RESULTS / "router_psi_scores_daily.csv", index=False, encoding="utf-8-sig")
    pol_psi = psi_df[psi_df.khoi == "policy_val"]
    print("[1] PSI điểm dự báo trong policy_val: " + f"TB {pol_psi.psi_max.mean():.3f}, lớn nhất {pol_psi.psi_max.max():.3f}; "
          f"số ngày >= {PSI_WARN}: {(pol_psi.psi_max >= PSI_WARN).sum()}/{len(pol_psi)}, >= {PSI_STOP}: {(pol_psi.psi_max >= PSI_STOP).sum()}")
    PSI_BY_DAY = psi_df.set_index("ngay").psi_max.to_dict()

    # hiệu quả thực 10 ngày gần nhất (kết quả đến hết ngày D-1, đã biết tại t0) của mô hình và của quy tắc dự phòng
    cellperf = []
    for (day, co), g in d.groupby(["ngay", "ca_ord"], sort=True):
        yy = (g.Y_short + B_FULL * g.Y_full).values
        cellperf.append({"ngay": day, "m": yy[np.argsort(-g.v.values)[:K_PER_SHIFT]].sum(), "f": yy[np.argsort(-g.v_du_phong.values)[:K_PER_SHIFT]].sum()})
    cellperf = pd.DataFrame(cellperf)
    PERF_BY_DAY = {}
    for day in pol_psi.ngay:
        prev = cellperf[cellperf.ngay < day].tail(20)
        PERF_BY_DAY[day] = (prev.m.mean(), prev.f.mean())
    pf = pd.DataFrame([{"ngay": k, "mo_hinh": v[0], "du_phong": v[1]} for k, v in PERF_BY_DAY.items()])
    print(f"      Hiệu quả thực 10 ngày trước (policy_val): mô hình TB {pf.mo_hinh.mean():.2f} vs quy tắc dự phòng {pf.du_phong.mean():.2f}; "
          f"số ngày mô hình kém hơn dự phòng: {(pf.mo_hinh < pf.du_phong).sum()}/{len(pf)}")


    #%% ======================================================================
    # 2. Router
    # ========================================================================
    def control_bucket(station, day, shift):
        h = hashlib.sha256(f"{CONTROL_SALT}|{station}|{pd.Timestamp(day).date()}|{shift}".encode()).hexdigest()
        return int(h[:8], 16) % 100

    def route(df, psi_by_day, psi_override=None, shuffle=False, perf_override=None, use_soft=True, use_l3=True):
        """df: các ô (ngày, ca) x 80 trạm. Trả về (kế hoạch các trạm được chọn, nhật ký theo ô)."""
        plans, logs, hist = [], [], {}
        cells = list(df.groupby(["ngay", "ca_ord"], sort=True))
        day_index = {dd: i for i, dd in enumerate(sorted({c[0][0] for c in cells}))}
        for (day, co), g in cells:
            t1 = time.perf_counter()
            g = g.sample(frac=1, random_state=1) if shuffle else g
            g = g.sort_values("station_id") if not shuffle else g
            shift = g.ca.iloc[0]
            psi_v = psi_by_day.get(day, np.nan) if psi_override is None else psi_override
            pm, pfb = (PERF_BY_DAY.get(day, (np.nan, np.nan)) if perf_override is None else perf_override)
            fallback = bool(use_l3 and psi_v >= PSI_STOP and pm < pfb)
            warn = bool(psi_v >= PSI_WARN)
            v = (g.v_du_phong if fallback else g.v).values.copy()
            # lớp 1: trạm không hợp lệ -> không được chọn
            n_invalid = int((~g.hop_le.values).sum()); v[~g.hop_le.values] = 0
            order = np.argsort(g.station_id.values) if shuffle else np.arange(len(g))
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            solver = solve_greedy if fallback else solve_milp
            # lớp 2: điểm thấp và giới hạn số lần ghé (đếm số trạm LẼ RA được chọn mà bị loại)
            n_low = n_block = 0
            if use_soft:
                z0 = solver(v[order])[inv]
                low = (v < V_MIN) & (not fallback)
                di = day_index[day]
                blocked = np.array([sum(1 for dj in hist.get(s, []) if di - 4 <= dj <= di) >= MAX_VISITS_5D for s in g.station_id])
                n_low, n_block = int((z0 & low).sum()), int((z0 & blocked & ~low).sum())
                v[low] = 0; v[blocked] = 0
            zs = solver(v[order])                                   # thứ tự trạm chuẩn cho bộ giải
            z = zs[inv]
            di = day_index[day]
            for i in np.where(z)[0]:
                r = g.iloc[i]; sid = r.station_id
                bucket = control_bucket(sid, day, shift)
                arm = "CONTROL" if bucket < CONTROL_PCT else "TREATMENT"
                if arm == "TREATMENT": hist.setdefault(sid, []).append(di)
                direction = "SHORTAGE" if r.p_Y_short_final >= B_FULL * r.p_Y_full_final else "OVERFLOW"
                plans.append({"idempotency_key": f"{POLICY_ID}|{sid}|{pd.Timestamp(day).date()}|{shift}", "policy_id": POLICY_ID,
                              "model_version": MODEL_VERSION, "t0": (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=22)).isoformat(),
                              "ngay": day, "ca": shift, "station_id": sid, "arm": arm, "action": "VISIT" if arm == "TREATMENT" else "HOLD_CONTROL",
                              "reason_code": direction, "p_short": r.p_Y_short_final, "p_full": r.p_Y_full_final, "v": r.v,
                              "mode": "FALLBACK_RULE" if fallback else "MODEL", "bucket": bucket, "Y_short": r.Y_short, "Y_full": r.Y_full})
            logs.append({"ngay": day, "ca": shift, "psi": psi_v, "mode": "FALLBACK_RULE" if fallback else "MODEL", "canh_bao_psi": warn,
                         "bi_loai_khong_hop_le": n_invalid, "bi_loai_diem_thap": n_low, "bi_loai_gioi_han_lan_ghe": n_block,
                         "so_tram_chon": int(z.sum()), "phut": float(cost_min(zs)), "ms": (time.perf_counter() - t1) * 1000})
        return pd.DataFrame(plans), pd.DataFrame(logs)

    dp = d[d.khoi == "policy_val"].copy()
    plan, log = route(dp, PSI_BY_DAY)
    plan.to_csv(RESULTS / "router_plan_policy_val.csv", index=False, encoding="utf-8-sig")
    log.to_csv(RESULTS / "router_cell_log.csv", index=False, encoding="utf-8-sig")
    print(f"[2] Router chạy xong: {len(log)} ô, {len(plan)} lệnh; chế độ: {log['mode'].value_counts().to_dict()}; "
          f"độ trễ/ô TB {log.ms.mean():.1f} ms, p95 {log.ms.quantile(.95):.1f} ms")
    print("      Lớp 1 loại (trạm không hợp lệ):", int(log.bi_loai_khong_hop_le.sum()), "| lớp 2 loại (điểm thấp):", int(log.bi_loai_diem_thap.sum()),
          "| lớp 2 loại (giới hạn lần ghé):", int(log.bi_loai_gioi_han_lan_ghe.sum()), "| ô bị cảnh báo PSI:", int(log.canh_bao_psi.sum()))


    #%% ======================================================================
    # 3. Nhóm CONTROL: cân bằng và kiểm tra A/A (không có chuyến thật nào nên kết cục không được khác giữa hai nhóm)
    # ========================================================================
    rows = []
    for c in ["p_short", "p_full", "v"]:
        a, b = plan[plan.arm == "TREATMENT"][c], plan[plan.arm == "CONTROL"][c]
        rows.append({"bien": c, "TB_can_thiep": a.mean(), "TB_control": b.mean(), "SMD": (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)})
    plan["Y_any"] = plan.Y_short * 1.0 + plan.Y_full * B_FULL
    a, b = plan[plan.arm == "TREATMENT"].Y_any, plan[plan.arm == "CONTROL"].Y_any
    udays = plan.ngay.unique()
    g_ = plan.groupby(["ngay", "arm"]).Y_any.agg(["sum", "size"]).unstack().fillna(0)
    bs = []
    for _ in range(1000):
        ds = rng.choice(udays, len(udays)); s_ = g_.loc[ds].sum()
        bs.append(s_[("sum", "TREATMENT")] / s_[("size", "TREATMENT")] - s_[("sum", "CONTROL")] / s_[("size", "CONTROL")])
    rows.append({"bien": "kết cục thực (A/A)", "TB_can_thiep": a.mean(), "TB_control": b.mean(), "SMD": np.nan})
    bal = pd.DataFrame(rows).round(4)
    bal.to_csv(RESULTS / "router_control_can_bang.csv", index=False, encoding="utf-8-sig")
    share = (plan.arm == "CONTROL").mean()
    print(f"\n[3] CONTROL chiếm {share:.1%} lệnh ({(plan.arm == 'CONTROL').sum()}/{len(plan)}); cân bằng:"); print(bal.to_string(index=False))
    print(f"      A/A: chênh lệch kết cục (can thiệp − control) = {np.mean(bs):+.3f}, CI95 theo ngày [{np.percentile(bs, 2.5):+.3f}; {np.percentile(bs, 97.5):+.3f}]")


    #%% ======================================================================
    # 4. Chi phí của guardrail và của việc giữ CONTROL (giá trị thực = sự kiện xảy ra tại trạm được chọn)
    # ========================================================================
    pure_val = []
    for (day, co), g in dp.groupby(["ngay", "ca_ord"]):
        z = solve_milp(g.sort_values("station_id").v.values)
        gg = g.sort_values("station_id"); pure_val.append(((gg.Y_short + B_FULL * gg.Y_full).values * z).sum())
    ncell = len(log)
    plan_l2, _ = route(dp, PSI_BY_DAY, use_l3=False)                       # chỉ lớp 1 + lớp 2
    plan_l1, _ = route(dp, PSI_BY_DAY, use_soft=False, use_l3=False)       # chỉ lớp 1
    plan_l1["Y_any"] = plan_l1.Y_short * 1.0 + plan_l1.Y_full * B_FULL
    plan_l2["Y_any"] = plan_l2.Y_short * 1.0 + plan_l2.Y_full * B_FULL
    cost_tab = pd.DataFrame([
        {"buoc": "MILP thuần (không guardrail)", "gia_tri_thuc_moi_o": np.mean(pure_val)},
        {"buoc": "+ Lớp 1 (quy tắc cứng)", "gia_tri_thuc_moi_o": plan_l1.Y_any.sum() / ncell},
        {"buoc": "+ Lớp 2 (quy tắc mềm: điểm tối thiểu, giới hạn lần ghé)", "gia_tri_thuc_moi_o": plan_l2.Y_any.sum() / ncell},
        {"buoc": "+ Lớp 3 (PSI + xác nhận hiệu quả thực)", "gia_tri_thuc_moi_o": plan.Y_any.sum() / ncell},
        {"buoc": "Thực thi thật (bỏ nhóm CONTROL)", "gia_tri_thuc_moi_o": plan[plan.arm == "TREATMENT"].Y_any.sum() / ncell}]).round(3)
    cost_tab.to_csv(RESULTS / "router_chi_phi_guardrail.csv", index=False, encoding="utf-8-sig")
    print("\n[4] Chi phí guardrail / CONTROL (giá trị thực mỗi ô):"); print(cost_tab.to_string(index=False))


    #%% ======================================================================
    # 5. Bộ kiểm tra tự động (lưu thành bảng cho báo cáo)
    # ========================================================================
    tests = []
    def T(name, ok, detail=""):
        tests.append({"kiem_tra": name, "ket_qua": "Đạt" if ok else "KHÔNG ĐẠT", "chi_tiet": detail}); print(f"      [{'Đạt' if ok else 'KHÔNG'}] {name} {detail}")

    T("Idempotency: không có khoá trùng", plan.idempotency_key.is_unique, f"({len(plan)} khoá duy nhất)")
    plan2, log2 = route(dp, PSI_BY_DAY)
    cols = ["idempotency_key", "arm", "reason_code", "mode"]
    T("Chạy lại cho kết quả y hệt (replay)", plan[cols].equals(plan2[cols]) and np.allclose(plan.v, plan2.v))
    plan3, _ = route(dp.sample(frac=1, random_state=7), PSI_BY_DAY, shuffle=True)
    k1, k2 = set(zip(plan.idempotency_key, plan.arm)), set(zip(plan3.idempotency_key, plan3.arm))
    T("Đảo thứ tự dữ liệu đầu vào không đổi lệnh và nhóm CONTROL", k1 == k2, f"(khác biệt {len(k1 ^ k2)})")
    T("Lớp 1: mỗi ca <= K trạm", (log.so_tram_chon <= K_PER_SHIFT).all(), f"(tối đa {log.so_tram_chon.max()})")
    T("Lớp 1: mỗi ca <= B phút", (log.phut <= B_MIN).all(), f"(tối đa {log.phut.max():.0f} phút)")
    T("Lớp 1: mỗi trạm một lần mỗi ca", not plan.duplicated(["station_id", "ngay", "ca"]).any())
    T(f"CONTROL trong 8–12% (mục tiêu {CONTROL_PCT}%)", 0.08 <= share <= 0.12, f"({share:.1%})")
    T("CONTROL cân bằng điểm dự báo (|SMD| < 0,3)", bal[bal.bien.isin(['p_short', 'p_full', 'v'])].SMD.abs().max() < 0.3,
      f"(max |SMD| {bal[bal.bien.isin(['p_short','p_full','v'])].SMD.abs().max():.2f})")
    T("A/A: kết cục không khác giữa hai nhóm (CI chứa 0)", np.percentile(bs, 2.5) <= 0 <= np.percentile(bs, 97.5))
    _, lg_stress = route(dp.iloc[:N * 4], PSI_BY_DAY, psi_override=0.40, perf_override=(1.0, 2.0))
    T("Lớp 3: PSI cao và mô hình kém hơn quy tắc dự phòng -> chuyển dự phòng", (lg_stress["mode"] == "FALLBACK_RULE").all())
    _, lg_keep = route(dp.iloc[:N * 4], PSI_BY_DAY, psi_override=0.40, perf_override=(2.0, 1.0))
    T("Lớp 3: PSI cao nhưng mô hình vẫn tốt hơn -> giữ mô hình và cảnh báo", (lg_keep["mode"] == "MODEL").all() and lg_keep.canh_bao_psi.all())
    _, lg_ok = route(dp.iloc[:N * 4], PSI_BY_DAY, psi_override=0.02, perf_override=(1.0, 2.0))
    T("Lớp 3: PSI thấp giữ mô hình dù hiệu quả gần đây kém", (lg_ok["mode"] == "MODEL").all())

    # HITL: ghi đè
    def apply_override(cell_plan, cell_stations_valid, action, station, reason, user="dispatcher"):
        """Trả về (chấp nhận?, lý do). Quy tắc cứng không bao giờ bị ghi đè; mọi ghi đè đều cần lý do và được ghi nhật ký."""
        if not reason: return False, "thiếu lý do"
        cur = cell_plan[cell_plan.station_id == station]
        if action == "REMOVE":
            return (True, "quy tắc mềm") if len(cur) else (False, "trạm không có trong kế hoạch")
        if action == "ADD":
            if len(cur): return False, "DUP_STATION (lớp 1)"
            if station not in cell_stations_valid: return False, "STATION_INVALID (lớp 1)"
            sel_ids = cell_plan.station_id.tolist() + [station]
            zz = np.zeros(N, bool); zz[[stations.index(s) for s in sel_ids]] = True
            if zz.sum() > K_PER_SHIFT: return False, "OVER_K (lớp 1)"
            if cost_min(zz) > B_MIN: return False, "OVER_BUDGET (lớp 1)"
            return True, "quy tắc mềm"
        if action == "CONTROL_TO_TREATMENT": return False, "CONTROL_INTEGRITY (lớp 1)"
        return False, "hành động không hỗ trợ"

    ex = plan[(plan.ngay == plan.ngay.iloc[0]) & (plan.ca == plan.ca.iloc[0])]
    gcell = dp[(dp.ngay == plan.ngay.iloc[0]) & (dp.ca == plan.ca.iloc[0])]
    valid_ids = set(gcell[gcell.hop_le].station_id)
    other = [s for s in stations if s not in set(ex.station_id) and s in valid_ids]
    full_ex = ex.copy()
    okA, _ = apply_override(ex, valid_ids, "REMOVE", ex.station_id.iloc[0], "xe tải hỏng một tuyến")
    okB, _ = apply_override(ex, valid_ids, "REMOVE", ex.station_id.iloc[0], "")
    okC, whyC = apply_override(ex, valid_ids, "ADD", other[0], "sự kiện địa phương") if len(ex) >= K_PER_SHIFT else (False, "ca chưa đầy")
    okD, _ = apply_override(ex, valid_ids, "CONTROL_TO_TREATMENT", ex.station_id.iloc[0], "quản lý yêu cầu")
    okE, _ = apply_override(ex, valid_ids, "ADD", ex.station_id.iloc[0], "thêm trùng")
    T("HITL: gỡ trạm kèm lý do được chấp nhận (quy tắc mềm)", okA)
    T("HITL: ghi đè không có lý do bị từ chối", not okB)
    T("HITL: thêm trạm khi ca đã đủ K bị từ chối (lớp 1)", not okC, f"({whyC})")
    T("HITL: chuyển CONTROL sang can thiệp bị từ chối (lớp 1)", not okD)
    T("HITL: thêm trạm đã có bị từ chối (lớp 1)", not okE)
    tdf = pd.DataFrame(tests); tdf.to_csv(RESULTS / "router_kiem_tra.csv", index=False, encoding="utf-8-sig")
    print(f"[5] {int((tdf.ket_qua == 'Đạt').sum())}/{len(tdf)} kiểm tra đạt")

    # hình
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    pp = psi_df[psi_df.khoi == "policy_val"]
    ax[0].plot(pp.ngay, pp.psi_short, label="điểm thiếu xe", color="#1F3A5F"); ax[0].plot(pp.ngay, pp.psi_full, label="điểm đầy xe", color="#D9822B")
    ax[0].axhline(PSI_WARN, color="k", ls=":", lw=1); ax[0].axhline(PSI_STOP, color="r", ls="--", lw=1)
    ax[0].set_title("Hình 12a. PSI điểm dự báo (cửa sổ 5 ngày, policy_val)"); ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)
    ct = cost_tab.set_index("buoc").gia_tri_thuc_moi_o
    ax[1].barh(range(len(ct)), ct.values, color=["#8A8A8A", "#8A8A8A", "#D9822B", "#1F3A5F", "#2E8B57"]); ax[1].set_yticks(range(len(ct))); ax[1].set_yticklabels([x[:46] for x in ct.index], fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xlabel("Sự kiện thực tại trạm được chọn, mỗi ô"); ax[1].set_title("Hình 12b. Cái giá của guardrail và CONTROL")
    plt.tight_layout(); plt.savefig(RESULTS / "router_hinh12_psi_chi_phi.png"); plt.close()
    print(f"[✓] BƯỚC 8 hoàn tất sau {time.perf_counter() - t_start:.0f}s")

#%% ======================================================================
def buoc_09_dong_bang():
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 9: Đóng băng bản phát hành và CÔNG BỐ tiêu chí chấp nhận TRƯỚC khi mở locked
    #
    # Mục tiêu: (1) khẳng định locked/OOT chưa bị mở; (2) công bố tiêu chí chấp nhận (ngưỡng đặt từ policy_val và lập luận,
    # không từ locked); (3) băm SHA-256 mô hình, mã, tham số, dữ liệu trung gian và 33 file nguồn vào release manifest;
    # (4) ghi kế hoạch triển khai và điều kiện huấn luyện lại. Sau bước này KHÔNG sửa bất kỳ file nào trong manifest:
    # 10_locked_oot.py (chế độ real) sẽ từ chối chạy nếu có băm nào lệch.
    # ========================================================================
    nap_bo_giai(); nap_bo_router(); nap_bo_dac_trung()

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    CODE = CODE_DIR


    def sha(p, chunk=1 << 22):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while b := f.read(chunk): h.update(b)
        return h.hexdigest()


    #%% ======================================================================
    # 1. Locked/OOT chưa bị mở
    # ========================================================================
    dev = pd.read_parquet(DATA / "features_dev.parquet")
    pred_dev = pd.read_parquet(DATA / "pred_dev.parquet")
    checks = [("Chưa có khoá locked", not (RESULTS / "locked_access.lock").exists()),
              ("features_dev.parquet kết thúc ≤ hết policy_val", dev.ngay.max() <= pd.Timestamp(DEV_END)),
              ("pred_dev.parquet chỉ có validation/calibration/policy_val", set(pred_dev.khoi) <= {"validation", "calibration", "policy_val"}),
              ("Không có kết quả locked/oot nào trong Results", not any(RESULTS.glob("oot_*")) and not any(p for p in RESULTS.glob("locked_*") if not p.name.startswith("dry_")))]
    for n_, ok in checks: print(f"  [{'Đạt' if ok else 'KHÔNG'}] {n_}")
    assert all(ok for _, ok in checks), "Locked/OOT có dấu hiệu đã bị mở; dừng đóng băng"
    print("[1] Xác nhận locked/OOT chưa mở")


    #%% ======================================================================
    # 2. Tham chiếu policy_val bằng đúng hàm đánh giá sẽ dùng cho locked, rồi công bố tiêu chí
    # ========================================================================
    bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
    pred = predict_frozen(dev[dev.khoi != "train"], bundle)
    d_all = prep_cells(pred, dev)
    psi_df = psi_daily(d_all); cperf = cell_perf(d_all)
    ref = full_block_metrics(d_all, psi_df, cperf, "policy_val")["m"]
    ref.update({"replay_max_diff": 0.0, "leakage_max_diff": 0.0, "router_replay_diff": 0})
    NOW = pd.Timestamp.now().isoformat(timespec="seconds")

    C = [  # (mã, khối, nội dung, chỉ số, toán tử, ngưỡng, lý do chọn ngưỡng)
     ("L01", "locked", "Lift@8 của GBM, nhãn thiếu xe", "lift_short", ">=", "1.5", "Bằng 60% mức đo trên policy_val (2,5); nhỏ hơn 1,5 thì xếp hạng gần như không hơn ngẫu nhiên"),
     ("L02", "locked", "Lift@8 của GBM, nhãn đầy xe", "lift_full", ">=", "1.5", "Như L01 (policy_val 3,4)"),
     ("L03", "locked", "PR-AUC GBM hơn mốc 'cùng ca tuần trước', thiếu xe", "prauc_hon_lag5_short", ">", "0", "Mô hình phải hơn quy tắc đơn giản đang có sẵn"),
     ("L04", "locked", "PR-AUC GBM hơn mốc 'cùng ca tuần trước', đầy xe", "prauc_hon_lag5_full", ">", "0", "Như L03"),
     ("L05", "locked", "Tỷ số xác suất dự báo TB / tỷ lệ thực, thiếu xe", "ti_so_short", "trong", "0.7;1.3", "Sai lệch ±30% là biên chịu được vì tỷ lệ sự kiện đổi theo mùa 7,7%–11,9%"),
     ("L06", "locked", "Tỷ số xác suất dự báo TB / tỷ lệ thực, đầy xe", "ti_so_full", "trong", "0.7;1.3", "Như L05"),
     ("L07", "locked", "ECE xác suất thiếu xe", "ece_short", "<=", "0.03", "Gấp rưỡi mức policy_val (0,020)"),
     ("L08", "locked", "ECE xác suất đầy xe", "ece_full", "<=", "0.03", "Gấp đôi mức policy_val (0,015)"),
     ("L09", "locked", "Giá trị thực GBM+MILP hơn ngẫu nhiên+MILP: cận dưới CI95 theo ngày", "gbm_tru_ngau_nhien_CI_duoi", ">", "0", "Phải hơn ngẫu nhiên một cách có ý nghĩa, không chỉ về điểm"),
     ("L10", "locked", "Giá trị thực GBM+MILP hơn 'cùng ca tuần trước+MILP' (trung bình)", "gbm_tru_lag5_TB", ">", "0", "Mô hình phải đáng giá hơn quy tắc dự phòng"),
     ("L11", "locked", "Tỷ lệ giá trị so với oracle", "ti_le_so_oracle", ">=", "0.25", "Policy_val 0,36, ngẫu nhiên 0,21: cần ở trên mức ngẫu nhiên rõ rệt"),
     ("L12", "locked", "MILP không kém Greedy về giá trị dự báo ở mọi ô", "milp_ge_greedy_du_bao", ">=", "1", "Tính chất toán học của nghiệm tối ưu; vi phạm = lỗi bộ giải"),
     ("L13", "locked", "Số kiểm tra tự động của router không đạt", "router_tests_fail", "==", "0", "Ràng buộc cứng và idempotency không có ngoại lệ"),
     ("L14", "locked", "Tỷ lệ ô phải dùng chính sách dự phòng", "fallback_share", "<=", "0.2", "Policy_val 2,3%; trên 20% nghĩa là mô hình không còn dùng được"),
     ("L15", "locked", "Giá trị mất do 3 tầng guardrail (sự kiện/ô)", "guardrail_mat_gia_tri", "<=", "0.3", "Policy_val 0,08; guardrail không được ăn quá ~10% giá trị"),
     ("L16", "locked", "Độ trễ router p95 mỗi ô (ms)", "latency_p95_ms", "<=", "500", "Thời hạn từ t0 22:00 đến đầu ca 06:00 là 8 giờ; 0,5 giây là rất rộng"),
     ("L17", "locked", "Replay: dựng lại dev trùng bản đã lưu (đặc trưng, xác suất)", "replay_max_diff", "<=", "0", "Quy trình phải tái lập được"),
     ("L18", "locked", "Thí nghiệm rò rỉ trên ngày locked/OOT", "leakage_max_diff", "<=", "0", "Đặc trưng ngày D không được đổi khi ghi đè tương lai"),
     ("L19", "locked", "Router replay trên policy_val trùng kết quả bước 8 (số khác biệt)", "router_replay_diff", "<=", "0", "Bản router dùng ở bước 10 phải đúng là bản đã duyệt"),
     ("O01", "oot", "Lift@8 của GBM, thiếu xe (OOT, stress)", "lift_short", ">=", "1.2", "Nới hơn locked: OOT xa hơn, chỉ cần còn hơn ngẫu nhiên rõ"),
     ("O02", "oot", "Lift@8 của GBM, đầy xe (OOT, stress)", "lift_full", ">=", "1.2", "Như O01"),
     ("O03", "oot", "Giá trị thực GBM+MILP hơn ngẫu nhiên+MILP (trung bình)", "gbm_tru_ngau_nhien_TB", ">", "0", "Tối thiểu không tệ hơn ngẫu nhiên"),
     ("O04", "oot", "Số kiểm tra tự động của router không đạt", "router_tests_fail", "==", "0", "Hệ thống phải còn an toàn dù dữ liệu trôi"),
     ("O05", "oot", "Giá trị mất do 3 tầng guardrail (sự kiện/ô)", "guardrail_mat_gia_tri", "<=", "0.3", "Như L15"),
     ("O06", "oot", "Độ trễ router p95 mỗi ô (ms)", "latency_p95_ms", "<=", "500", "Như L16"),
    ]
    crit = pd.DataFrame(C, columns=["ma", "khoi", "noi_dung", "chi_so", "toan_tu", "nguong", "ly_do_chon_nguong"])
    crit["gia_tri_policy_val"] = [round(float(ref[k]), 4) for k in crit.chi_so]
    R = [("Q1", "quy_tac", "Locked chỉ mở MỘT lần; kết quả được công bố nguyên trạng, không chỉnh mô hình/ngưỡng τ/tham số rồi chạy lại", "", "", "", "Chống chọn mô hình theo locked"),
         ("Q2", "quy_tac", "Đạt toàn bộ L01–L19 -> được phép chạy thử có kiểm soát (canary)", "", "", "", ""),
         ("Q3", "quy_tac", "Không đạt bất kỳ L nào -> chỉ chạy bóng (shadow): ghi dự báo và lệnh nhưng KHÔNG giao xe tải; điều tra nguyên nhân trước khi chạy lại ở một khoá mới", "", "", "", ""),
         ("Q4", "quy_tac", "OOT là cửa sổ stress: không đạt O01–O06 không bác bỏ bản phát hành nhưng bắt buộc xem xét huấn luyện lại; không dùng OOT để chỉnh mô hình", "", "", "", ""),
         ("Q5", "quy_tac", "Ngưỡng được đặt SAU khi đã xem policy_val (không phải locked) và dựa trên lập luận vận hành; điều này được thú nhận trong báo cáo", "", "", "", "")]
    crit = pd.concat([crit, pd.DataFrame(R, columns=["ma", "khoi", "noi_dung", "chi_so", "toan_tu", "nguong", "ly_do_chon_nguong"])], ignore_index=True)
    crit["cong_bo_luc"] = NOW
    crit.to_csv(RESULTS / "acceptance_criteria.csv", index=False, encoding="utf-8-sig")
    print(f"[2] Đã công bố {int(crit.khoi.isin(['locked', 'oot']).sum())} tiêu chí lúc {NOW}"); print(crit[crit.khoi.isin(["locked", "oot"])][["ma", "chi_so", "toan_tu", "nguong", "gia_tri_policy_val"]].to_string(index=False))


    #%% ======================================================================
    # 3. Kế hoạch triển khai và điều kiện huấn luyện lại
    # ========================================================================
    plan = pd.DataFrame([
     ("1. Chạy bóng (shadow)", "Qua locked; ≥ 4 tuần", "Router sinh lệnh mỗi ngày nhưng không giao xe tải; so lệnh với quyết định điều phối viên", "Còn đạt L01–L11 trên cửa sổ trượt 4 tuần", "Quay lại huấn luyện lại nếu không đạt"),
     ("2. Canary", "Hết shadow đạt", "Một khu (13–15 trạm) chạy thật, CONTROL 10% giữ cố định, HITL bật; các khu còn lại giữ như cũ", "Không vi phạm quy tắc cứng; A/A và cân bằng CONTROL đạt; điều phối viên chấp nhận ≥ 80% lệnh", "Tắt router, quay về quy tắc 'cùng ca tuần trước + Greedy'"),
     ("3. Mở rộng", "Hết canary đạt", "Mở dần cả 6 khu; bắt đầu pilot ngẫu nhiên (K trong 2K) để đo hiệu ứng thật thay cho giả định b", "Pilot đạt đủ ngày theo bảng sức mạnh (20–90 ngày)", "Rút về canary"),
     ("Điều kiện huấn luyện lại A", "Giám sát hằng ngày", "PSI điểm dự báo ≥ 0,25 liên tiếp ≥ 5 ngày làm việc VÀ mô hình kém quy tắc dự phòng trên 20 ô gần nhất", "", "Huấn luyện lại bằng dữ liệu mới, kiểm lại bằng tập khoá mới"),
     ("Điều kiện huấn luyện lại B", "Giám sát hằng ngày", "Mô hình kém quy tắc dự phòng liên tiếp ≥ 10 ngày làm việc", "", "Như A"),
     ("Điều kiện huấn luyện lại C", "Lịch", "Quá 3 tháng kể từ lần huấn luyện gần nhất", "", "Huấn luyện lại định kỳ, bắt buộc đổi mùa"),
     ("Ngắt khẩn cấp", "Bất kỳ lúc nào", "Vi phạm quy tắc cứng hoặc kiểm tra tự động không đạt", "", "Tự chuyển quy tắc dự phòng và báo người phụ trách"),
    ], columns=["giai_doan", "thoi_diem", "noi_dung", "dieu_kien_dat", "neu_khong_dat"])
    plan.to_csv(RESULTS / "release_ke_hoach_trien_khai.csv", index=False, encoding="utf-8-sig")
    print("[3] Kế hoạch triển khai và điều kiện huấn luyện lại đã ghi")


    #%% ======================================================================
    # 4. Release manifest: SHA-256 mã, mô hình, tham số, dữ liệu trung gian, 33 file nguồn
    # ========================================================================
    rows = []
    def add(loai, p):
        p = Path(p); rows.append({"loai": loai, "duong_dan": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    for p in sorted(CODE.glob("*.py")): add("ma", p)
    for p in [DATA / "models" / "forecast_bundle.joblib", DATA / "models" / "decision_bundle.joblib"]: add("mo_hinh", p)
    for n_ in ["tau_thresholds.csv", "stations_selected.csv", "feature_list.csv", "acceptance_criteria.csv", "release_ke_hoach_trien_khai.csv"]: add("tham_so_tieu_chi", RESULTS / n_)
    for n_ in ["features_dev.parquet", "pred_dev.parquet", "station_hour.parquet"]: add("du_lieu_trung_gian", DATA / n_)
    audit = pd.read_csv(RESULTS / "data_audit.csv")
    n_raw_ok = 0
    for r in audit.itertuples():
        p = BASE / "Datasets" / r.file_name
        h = sha(p); n_raw_ok += (h == r.sha256)
        rows.append({"loai": "du_lieu_nguon", "duong_dan": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size, "sha256": h})
    print(f"[4a] Băm {len(rows)} file; {n_raw_ok}/{len(audit)} file nguồn trùng băm đã ghi ở bước 1 (data_audit.csv)")
    assert n_raw_ok == len(audit), "Dữ liệu nguồn đã đổi so với lúc đọc ở bước 1"
    meta = {"policy_id": POLICY_ID, "model_version": MODEL_VERSION, "dong_bang_luc": NOW, "seed": SEED, "k_moi_ca": K_PER_SHIFT,
            "ngan_sach_phut": B_MIN, "tau_quantile": TAU_QUANTILE, "dev_end": DEV_END, "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
            "bo_hieu_chinh_Y_short": bundle["calibrators"][("Y_short", "GBM")]["chosen"], "bo_hieu_chinh_Y_full": bundle["calibrators"][("Y_full", "GBM")]["chosen"],
            "PSI_canh_bao": PSI_WARN, "PSI_dung": PSI_STOP, "CONTROL_pct": CONTROL_PCT, "V_MIN": V_MIN, "MAX_VISITS_5D": MAX_VISITS_5D}
    for k, v in meta.items(): rows.append({"loai": "meta", "duong_dan": k, "bytes": 0, "sha256": str(v)})
    man = pd.DataFrame(rows)
    man.to_csv(RESULTS / "release_manifest.csv", index=False, encoding="utf-8-sig")
    msha = sha(RESULTS / "release_manifest.csv")
    (RESULTS / "release_manifest.sha256").write_text(msha + "  release_manifest.csv\n", encoding="utf-8")
    print(man.groupby("loai").size().to_dict()); print(f"[4b] SHA-256 manifest: {msha}")
    print(f"[✓] BƯỚC 9 (đóng băng) hoàn tất sau {time.perf_counter() - t_start:.0f}s")

#%% ======================================================================
def buoc_10_locked_oot(mode="dry"):
    # BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
    # BƯỚC 10: Kiểm tra locked (một lần) và cửa sổ OOT (stress) với mô hình ĐÃ ĐÓNG BĂNG
    #
    # Cách chạy:  py -3 10_locked_oot.py dry   -> diễn tập trên policy_val (chưa đụng locked/oot, không tạo khoá)
    #             py -3 10_locked_oot.py real  -> chạy thật: kiểm tra manifest + tiêu chí đã công bố, mở locked MỘT lần, rồi OOT
    # Trình tự chống "nhìn rồi mới chọn": (1) 09_dong_bang.py đã băm mô hình/code/dữ liệu và công bố tiêu chí chấp nhận;
    # (2) ở chế độ real, script từ chối chạy nếu băm lệch hoặc khoá locked đã tồn tại; (3) khoá ghi lại thời điểm mở.
    # Locked khác OOT: locked = kiểm tra chấp nhận cuối, mở một lần (kiểm soát truy cập bằng khoá + nhật ký);
    # OOT = cửa sổ xa hơn để quan sát trôi dữ liệu và thử cơ chế giám sát, tiêu chí nới hơn, không dùng để chỉnh mô hình.
    # Không có huấn luyện lại, không đổi ngưỡng τ, không đổi tham số bộ giải ở bước này.
    # ========================================================================
    nap_bo_giai(); nap_bo_router(); nap_bo_dac_trung()

    warnings.filterwarnings("ignore")
    t_start = time.perf_counter()
    MODE = mode
    assert MODE in ("dry", "real")
    END = DEV_END if MODE == "dry" else BLOCKS["oot"][1]
    LOCK_BLK = "policy_val" if MODE == "dry" else "locked"
    PFX = "dry_" if MODE == "dry" else ""
    LOCK_FILE = RESULTS / "locked_access.lock"
    plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


    #%% ======================================================================
    # 0. Cổng kiểm soát truy cập (chỉ chế độ real)
    # ========================================================================
    if MODE == "real":
        mf = pd.read_csv(RESULTS / "release_manifest.csv")
        bad = [r.duong_dan for r in mf.itertuples() if r.loai != "meta" and sha(BASE / r.duong_dan) != r.sha256]
        assert not bad, f"Băm lệch so với manifest đã đóng băng, từ chối mở locked: {bad}"
        crit_hash = mf[mf.duong_dan == "Results/acceptance_criteria.csv"].sha256.iloc[0]
        assert sha(RESULTS / "acceptance_criteria.csv") == crit_hash, "Tiêu chí chấp nhận đã bị sửa sau khi công bố"
        assert not LOCK_FILE.exists(), f"Locked đã được mở trước đó ({LOCK_FILE.read_text(encoding='utf-8').splitlines()[0]}); không chạy lại"
        MANIFEST_SHA = sha(RESULTS / "release_manifest.csv")
        print(f"[0] Manifest khớp ({len(mf)} dòng), tiêu chí chưa bị sửa, locked chưa từng mở. SHA manifest {MANIFEST_SHA[:16]}…")
    else:
        print(f"[0] Chế độ dry: dùng policy_val thay locked, không tạo khoá, không đọc dữ liệu sau {DEV_END}")


    #%% ======================================================================
    # 1. Dựng đặc trưng đến END bằng mã đã đóng băng + τ đóng băng; nạp mô hình đóng băng
    # ========================================================================
    bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
    FEATURES = bundle["features"]
    df, base, city = build_dataset(END, FEATURES)
    for k in (["policy_val"] if MODE == "dry" else ["locked", "oot"]):
        n_exp = len(stations_fe) * 2 * len(pd.bdate_range(*BLOCKS[k]))
        assert (df.khoi == k).sum() == n_exp, f"{k}: {(df.khoi == k).sum()} dòng, kỳ vọng {n_exp}"
    print(f"[1] Dựng xong {len(df):,} dòng đến {END}; số dòng theo khối:", df.khoi.value_counts().to_dict())


    #%% ======================================================================
    # 2. Kiểm tra phát hành: replay (dựng lại dev trùng bản đã lưu), rò rỉ, router replay
    # ========================================================================
    rel = []
    dev_old = pd.read_parquet(DATA / "features_dev.parquet")
    dev_new = df[df.ngay <= DEV_END]
    mg = dev_old[["station_id", "ngay", "ca"] + FEATURES].merge(dev_new[["station_id", "ngay", "ca"] + FEATURES], on=["station_id", "ngay", "ca"], suffixes=("_o", "_n"), how="outer", indicator=True)
    assert (mg._merge == "both").all(), "tập dòng dev khác bản đã lưu"
    f_diff = max(float(np.nanmax(np.abs(mg[f + "_o"].astype(float) - mg[f + "_n"].astype(float)))) for f in FEATURES)
    f_raw = f_diff; f_diff = 0.0 if f_diff < 1e-9 else f_diff        # dung sai sai số làm tròn số thực (cỡ 1e-15) khi cộng trượt
    rel.append({"kiem_tra": f"Replay: đặc trưng dev dựng lại trùng features_dev.parquet (lệch lớn nhất {f_raw:.1e}, dung sai 1e-9)", "gia_tri": f_diff, "ket_qua": "Đạt" if f_diff == 0 else "KHÔNG ĐẠT"})

    pred = predict_frozen(df[df.khoi != "train"], bundle)
    po_old = pd.read_parquet(DATA / "pred_dev.parquet")
    pm = po_old.merge(pred, on=["station_id", "ngay", "ca"], suffixes=("_o", "_n"))
    assert len(pm) == len(po_old)
    p_diff = max(float(np.abs(pm[f"p_{y}_final_o"] - pm[f"p_{y}_final_n"]).max()) for y in LABELS)
    p_diff = 0.0 if p_diff < 1e-12 else p_diff
    rel.append({"kiem_tra": "Replay: xác suất dev tính lại trùng pred_dev.parquet", "gia_tri": p_diff, "ket_qua": "Đạt" if p_diff == 0 else "KHÔNG ĐẠT"})

    rng_l = np.random.default_rng(SEED)
    if MODE == "real":
        ld = list(rng_l.choice(df[df.khoi == "locked"].ngay.unique(), 2, replace=False)) + list(rng_l.choice(df[df.khoi == "oot"].ngay.unique(), 1))
    else:
        ld = list(rng_l.choice(df[df.khoi == "policy_val"].ngay.unique(), 1))
    leak = leakage_check(df, base, city, FEATURES, sorted(ld))
    leak.to_csv(RESULTS / f"{PFX}leakage_experiment_locked_oot.csv", index=False, encoding="utf-8-sig")
    l_diff = float(leak.chenh_lech_lon_nhat.max())
    rel.append({"kiem_tra": "Rò rỉ: ghi đè tương lai bằng 99999 không làm đổi đặc trưng", "gia_tri": l_diff, "ket_qua": "Đạt" if l_diff == 0 else "KHÔNG ĐẠT"})
    print("[2a] Replay đặc trưng/xác suất và rò rỉ:"); print(pd.DataFrame(rel).to_string(index=False)); print(leak.to_string(index=False))

    d_all = prep_cells(pred, df)
    psi_df = psi_daily(d_all); cperf = cell_perf(d_all)
    psi_old = pd.read_csv(RESULTS / "router_psi_scores_daily.csv", parse_dates=["ngay"])
    pm2 = psi_old.merge(psi_df, on="ngay", suffixes=("_o", "_n"))
    assert len(pm2) == len(psi_old) and np.allclose(pm2.psi_max_o, pm2.psi_max_n)
    dp_dev = d_all[d_all.khoi == "policy_val"]
    plan_new, _ = route(dp_dev, psi_df.set_index("ngay").psi_max.to_dict(), perf_by_day(cperf, sorted(dp_dev.ngay.unique())))
    plan_old = pd.read_csv(RESULTS / "router_plan_policy_val.csv", parse_dates=["ngay"])
    cols = ["idempotency_key", "arm", "reason_code", "mode"]
    r_diff = int((plan_old[cols].reset_index(drop=True) != plan_new[cols].reset_index(drop=True)).to_numpy().sum()) if len(plan_old) == len(plan_new) else 999999
    rel.append({"kiem_tra": "Replay: router trên policy_val trùng router_plan_policy_val.csv (08)", "gia_tri": r_diff, "ket_qua": "Đạt" if r_diff == 0 else "KHÔNG ĐẠT"})
    pd.DataFrame(rel).to_csv(RESULTS / f"{PFX}release_kiem_tra.csv", index=False, encoding="utf-8-sig")
    print(f"[2b] Router replay: {r_diff} khác biệt; PSI dev trùng 08. Phiên bản mô hình {MODEL_VERSION}")


    #%% ======================================================================
    # 3. LOCKED (hoặc policy_val ở chế độ dry): đánh giá một lần theo tiêu chí đã công bố
    # ========================================================================
    crit_path = RESULTS / "acceptance_criteria.csv"
    crit = pd.read_csv(crit_path, dtype={"nguong": str}) if crit_path.exists() else None
    res = {}
    res[LOCK_BLK] = full_block_metrics(d_all, psi_df, cperf, LOCK_BLK)
    res[LOCK_BLK]["m"].update({"replay_max_diff": max(f_diff, p_diff), "leakage_max_diff": l_diff, "router_replay_diff": r_diff})
    R = res[LOCK_BLK]
    R["fc"].round(4).to_csv(RESULTS / f"{PFX}locked_du_bao.csv", index=False, encoding="utf-8-sig")
    R["tab"].round(4).to_csv(RESULTS / f"{PFX}locked_gia_tri_quyet_dinh.csv", index=False, encoding="utf-8-sig")
    R["boot"].round(4).to_csv(RESULTS / f"{PFX}locked_bootstrap.csv", index=False, encoding="utf-8-sig")
    R["rr"]["tests"].to_csv(RESULTS / f"{PFX}locked_router_kiem_tra.csv", index=False, encoding="utf-8-sig")
    R["rr"]["plan"].to_csv(RESULTS / f"{PFX}locked_router_plan.csv", index=False, encoding="utf-8-sig")
    R["rr"]["log"].to_csv(RESULTS / f"{PFX}locked_router_cell_log.csv", index=False, encoding="utf-8-sig")
    print(f"[3] Khối {LOCK_BLK}: {R['extra']['so_o']} ô, {R['extra']['so_ngay']} ngày")
    print(R["fc"].round(3).to_string(index=False)); print(R["tab"].round(3).to_string(index=False)); print(R["boot"].round(3).to_string(index=False))
    print(f"      Router: {int((R['rr']['tests'].ket_qua == 'Đạt').sum())}/{len(R['rr']['tests'])} kiểm tra đạt; dự phòng {R['m']['fallback_share']:.1%} số ô; "
          f"CONTROL {R['m']['control_share']:.1%}; A/A {R['rr']['aa'][0]:+.3f} [{R['rr']['aa'][1]:+.3f}; {R['rr']['aa'][2]:+.3f}]; p95 {R['m']['latency_p95_ms']:.0f} ms")
    if crit is not None:
        cl = crit[crit.khoi == "locked"].copy(); cl["khoi"] = LOCK_BLK
        out = check_criteria(cl, {k: v['m'] for k, v in res.items()})
        out["khoi"] = "locked" if MODE == "real" else "policy_val (diễn tập)"
        out.to_csv(RESULTS / f"{PFX}acceptance_ket_qua_locked.csv", index=False, encoding="utf-8-sig")
        print("\n  Tiêu chí chấp nhận (locked):"); print(out[["ma", "noi_dung", "toan_tu", "nguong", "gia_tri_do_duoc", "ket_qua"]].to_string(index=False))
        n_fail = int((out.ket_qua != "Đạt").sum())
        print(f"  => {len(out) - n_fail}/{len(out)} đạt; kết luận: {'ĐẠT -> được phép canary' if n_fail == 0 else 'KHÔNG ĐẠT -> chỉ chạy bóng (shadow), không chỉnh lại trên locked'}")
    if MODE == "real":
        LOCK_FILE.write_text(f"locked mở lúc {pd.Timestamp.now().isoformat()} bởi 10_locked_oot.py real; SHA manifest {MANIFEST_SHA}; "
                             f"model_version {MODEL_VERSION}; kết luận {n_fail} tiêu chí không đạt\nKHÔNG chạy lại, KHÔNG chỉnh mô hình/ngưỡng dựa trên kết quả này.\n", encoding="utf-8")
        print(f"      Đã ghi khoá {LOCK_FILE.name}")


    #%% ======================================================================
    # 4. OOT (stress) + giám sát: PSI, hiệu quả theo tháng, điều kiện huấn luyện lại
    # ========================================================================
    mon = d_all[d_all.khoi.isin(["policy_val", "locked", "oot"])].copy()
    mon["thang"] = mon.ngay.dt.strftime("%Y-%m")
    rng_m = np.random.default_rng(SEED); mrows = []
    for tm, g in mon.groupby("thang"):
        for y in LABELS:
            rate = g[y].mean()
            mrows.append({"thang": tm, "label": y, "khoi": g.khoi.iloc[0], "ty_le_su_kien": rate,
                          "Lift_GBM": precision_at_k(g, g[f"p_{y}_final"].values, y, rng_m).mean() / rate,
                          "Lift_lag5": precision_at_k(g, g[LAG[y]].values.astype(float), y, rng_m).mean() / rate,
                          "TB_xac_suat": g[f"p_{y}_final"].mean(), "tram_khong_hoat_dong": float((~g.hop_le).mean())})
    mon_df = pd.DataFrame(mrows).round(4); mon_df.to_csv(RESULTS / f"{PFX}giam_sat_theo_thang.csv", index=False, encoding="utf-8-sig")
    print("\n[4a] Giám sát theo tháng (lift@8, tỷ lệ sự kiện, trạm không hoạt động):"); print(mon_df.to_string(index=False))

    # điều kiện huấn luyện lại: A = PSI>=0,25 liên tiếp >=5 ngày VÀ mô hình kém dự phòng (20 ô); B = mô hình kém dự phòng liên tiếp >=10 ngày; C = quá 3 tháng
    days_m = psi_df[psi_df.khoi.isin(["policy_val", "locked", "oot"])].copy()
    pbd = perf_by_day(cperf, days_m.ngay)
    days_m["mo_hinh_20"] = [pbd[d][0] for d in days_m.ngay]; days_m["du_phong_20"] = [pbd[d][1] for d in days_m.ngay]
    days_m["kem"] = days_m.mo_hinh_20 < days_m.du_phong_20
    run = lambda s: s.groupby((~s).cumsum()).cumsum()
    days_m["psi_lien_tiep"] = run(days_m.psi_max >= PSI_STOP); days_m["kem_lien_tiep"] = run(days_m.kem)
    days_m["A_troi_du_lieu"] = (days_m.psi_lien_tiep >= 5) & days_m.kem
    days_m["B_hieu_qua"] = days_m.kem_lien_tiep >= 10
    days_m["huan_luyen_lai"] = days_m.A_troi_du_lieu | days_m.B_hieu_qua
    days_m.to_csv(RESULTS / f"{PFX}giam_sat_theo_ngay.csv", index=False, encoding="utf-8-sig")
    trig = days_m.groupby("khoi").agg(so_ngay=("ngay", "size"), psi_TB=("psi_max", "mean"), ngay_psi_ge_025=("psi_max", lambda s: int((s >= 0.25).sum())),
                                      ngay_mo_hinh_kem=("kem", "sum"), ngay_gan_co_A=("A_troi_du_lieu", "sum"), ngay_gan_co_B=("B_hieu_qua", "sum")).round(3)
    trig.to_csv(RESULTS / f"{PFX}giam_sat_tong_hop_khoi.csv", encoding="utf-8-sig")
    first = days_m[days_m.huan_luyen_lai].ngay.min()
    print("\n[4b] Điều kiện huấn luyện lại (A: PSI>=0,25 x5 ngày & kém dự phòng; B: kém dự phòng x10 ngày):"); print(trig.to_string())
    print(f"      Ngày đầu tiên gắn cờ huấn luyện lại: {first.date() if pd.notna(first) else 'chưa có'}")

    OOT_BLK = "oot" if MODE == "real" else "calibration"                  # dry: dùng calibration thay OOT để diễn tập đúng đường đi của mã
    res[OOT_BLK] = full_block_metrics(d_all, psi_df, cperf, OOT_BLK)
    res[OOT_BLK]["m"].update({"replay_max_diff": max(f_diff, p_diff), "leakage_max_diff": l_diff, "router_replay_diff": r_diff})
    O = res[OOT_BLK]
    O["fc"].round(4).to_csv(RESULTS / f"{PFX}oot_du_bao.csv", index=False, encoding="utf-8-sig")
    O["tab"].round(4).to_csv(RESULTS / f"{PFX}oot_gia_tri_quyet_dinh.csv", index=False, encoding="utf-8-sig")
    O["boot"].round(4).to_csv(RESULTS / f"{PFX}oot_bootstrap.csv", index=False, encoding="utf-8-sig")
    O["rr"]["tests"].to_csv(RESULTS / f"{PFX}oot_router_kiem_tra.csv", index=False, encoding="utf-8-sig")
    O["rr"]["plan"].to_csv(RESULTS / f"{PFX}oot_router_plan.csv", index=False, encoding="utf-8-sig")
    O["rr"]["log"].to_csv(RESULTS / f"{PFX}oot_router_cell_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n[4c] Khối {OOT_BLK}: {O['extra']['so_o']} ô, {O['extra']['so_ngay']} ngày"); print(O["fc"].round(3).to_string(index=False)); print(O["tab"].round(3).to_string(index=False)); print(O["boot"].round(3).to_string(index=False))
    print(f"      Router: {int((O['rr']['tests'].ket_qua == 'Đạt').sum())}/{len(O['rr']['tests'])} đạt; dự phòng {O['m']['fallback_share']:.1%}; CONTROL {O['m']['control_share']:.1%}")
    if crit is not None:
        co = crit[crit.khoi == "oot"].copy(); co["khoi"] = OOT_BLK
        out_o = check_criteria(co, {k: v["m"] for k, v in res.items()}); out_o["khoi"] = "oot" if MODE == "real" else "calibration (diễn tập)"
        out_o.to_csv(RESULTS / f"{PFX}acceptance_ket_qua_oot.csv", index=False, encoding="utf-8-sig")
        print("\n  Tiêu chí OOT (stress, nới hơn locked):"); print(out_o[["ma", "noi_dung", "toan_tu", "nguong", "gia_tri_do_duoc", "ket_qua"]].to_string(index=False))

    # hình 13: PSI + lift theo tháng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    cols_b = {"policy_val": "#8A8A8A", "locked": "#D9822B", "oot": "#1F3A5F"}
    for k, c in cols_b.items():
        t = days_m[days_m.khoi == k]
        if len(t): ax[0].plot(t.ngay, t.psi_max, color=c, label={"policy_val": "policy_val", "locked": "locked", "oot": "OOT"}[k])
    ax[0].axhline(PSI_WARN, color="k", ls=":", lw=1); ax[0].axhline(PSI_STOP, color="r", ls="--", lw=1)
    ax[0].set_title("Hình 13a. PSI điểm dự báo theo ngày"); ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)
    for y, ls in (("Y_short", "-"), ("Y_full", "--")):
        t = mon_df[mon_df.label == y]
        ax[1].plot(t.thang, t.Lift_GBM, ls, marker="o", color="#1F3A5F", label=f"GBM – {'thiếu' if y == 'Y_short' else 'đầy'}")
        ax[1].plot(t.thang, t.Lift_lag5, ls, marker="s", color="#8A8A8A", label=f"Cùng ca tuần trước – {'thiếu' if y == 'Y_short' else 'đầy'}")
    ax[1].axhline(1, color="k", ls=":", lw=1); ax[1].set_title("Hình 13b. Lift@8 theo tháng (policy_val → OOT)"); ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(RESULTS / f"{PFX}giam_sat_hinh13_psi_lift.png"); plt.close()

    pd.DataFrame([{"python": platform.python_version(), "che_do": MODE, "model_version": MODEL_VERSION, "seed": SEED, "thoi_diem": pd.Timestamp.now().isoformat()}]) \
        .to_csv(RESULTS / f"{PFX}chay_locked_oot_ghi_chu.csv", index=False, encoding="utf-8-sig")
    print(f"[✓] BƯỚC 10 ({MODE}) hoàn tất sau {time.perf_counter() - t_start:.0f}s")

#%% ======================================================================
# ĐIỂM VÀO
# ========================================================================
RUN = {"1": buoc_01_gop_du_lieu, "2": buoc_02_eda_chon_tram, "3": buoc_03_dac_trung, "4": buoc_04_du_bao_hieu_chinh,
       "5": buoc_05_pfi_shap, "6": buoc_06_nhan_qua_pilot, "7": buoc_07_toi_uu, "8": buoc_08_router,
       "freeze": buoc_09_dong_bang}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["-h"]
    if todo == ["-h"] or any(t not in list(RUN) + ["dry", "real"] for t in todo):
        sys.exit(__doc__)
    locked = (RESULTS / "locked_access.lock").exists()
    for t in todo:
        if t == "real":
            buoc_10_locked_oot("real")
        elif t == "dry":
            buoc_10_locked_oot("dry")
        elif locked:
            sys.exit(f"Bước '{t}' bị từ chối: Results/locked_access.lock đã tồn tại, kết quả 1–9 đã đóng băng và locked đã mở. Muốn chạy lại từ đầu phải làm trên một bản sao sạch của thư mục.")
        else:
            RUN[t]()
