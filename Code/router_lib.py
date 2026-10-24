"""Router dùng chung cho đánh giá locked/OOT (bản sao đúng logic của 08_router.py; replay test so với đầu ra của 08)."""
import time, hashlib
import numpy as np
import pandas as pd
import solver_lib as S
from config import *

POLICY_ID, CONTROL_SALT, CONTROL_PCT = "bike-rebalance-v1", "ctl-v1", 10
V_MIN, MAX_VISITS_5D, PSI_WARN, PSI_STOP = 0.10, 3, 0.10, 0.25
CA_ORDER = {"sang": 0, "chieu": 1}
sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str}).sort_values("station_id").reset_index(drop=True)
stations = sel.station_id.tolist(); N = len(stations)
tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})
TAU_OUT_MED, TAU_IN_MED = tau.tau_out.median(), tau.tau_in.median()
LAGC = [f"n_out_lag{k}" for k in range(1, 6)] + [f"n_in_lag{k}" for k in range(1, 6)]


def file_hash(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


MODEL_VERSION = file_hash(DATA / "models" / "forecast_bundle.joblib") + "+" + file_hash(DATA / "models" / "decision_bundle.joblib")


def prep_cells(pred, feat):
    """pred: xác suất + nhãn theo (trạm, ngày, ca); feat: bảng đặc trưng cùng khoá. Trả về bảng đã sắp theo (ngày, ca, trạm)."""
    ft = feat[["station_id", "ngay", "ca", "net_out_lag5", "net_in_lag5"] + LAGC]
    d = pred.merge(ft, on=["station_id", "ngay", "ca"], how="left")
    d["ca_ord"] = d.ca.map(CA_ORDER)
    d = d.sort_values(["ngay", "ca_ord", "station_id"]).reset_index(drop=True)
    assert (d.groupby(["ngay", "ca"]).size() == N).all()
    d["hop_le"] = d[LAGC].sum(axis=1) > 0
    d["v"] = d.p_Y_short_final + S.B_FULL * d.p_Y_full_final
    d["v_du_phong"] = d.net_out_lag5.clip(lower=0) / TAU_OUT_MED + S.B_FULL * d.net_in_lag5.clip(lower=0) / TAU_IN_MED
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
        yy = (g.Y_short + S.B_FULL * g.Y_full).values
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
        solver = S.solve_greedy if fallback else S.solve_milp
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
            direction = "SHORTAGE" if r.p_Y_short_final >= S.B_FULL * r.p_Y_full_final else "OVERFLOW"
            plans.append({"idempotency_key": f"{POLICY_ID}|{sid}|{pd.Timestamp(day).date()}|{shift}", "policy_id": POLICY_ID,
                          "model_version": MODEL_VERSION, "t0": (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=22)).isoformat(),
                          "ngay": day, "ca": shift, "station_id": sid, "arm": arm, "action": "VISIT" if arm == "TREATMENT" else "HOLD_CONTROL",
                          "reason_code": direction, "p_short": r.p_Y_short_final, "p_full": r.p_Y_full_final, "v": r.v,
                          "mode": "FALLBACK_RULE" if fallback else "MODEL", "bucket": bucket, "Y_short": r.Y_short, "Y_full": r.Y_full})
        logs.append({"ngay": day, "ca": shift, "psi": psi_v, "mode": "FALLBACK_RULE" if fallback else "MODEL", "canh_bao_psi": warn,
                     "bi_loai_khong_hop_le": n_invalid, "bi_loai_diem_thap": n_low, "bi_loai_gioi_han_lan_ghe": n_block,
                     "so_tram_chon": int(z.sum()), "phut": float(S.cost_min(zs)), "ms": (time.perf_counter() - t1) * 1000})
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
        if S.cost_min(zz) > S.B_MIN: return False, "OVER_BUDGET (lớp 1)"
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
    plan["Y_any"] = plan.Y_short * 1.0 + plan.Y_full * S.B_FULL
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
    T("Lớp 1: mỗi ca <= B phút", (log.phut <= S.B_MIN).all(), f"(tối đa {log.phut.max():.0f})")
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
