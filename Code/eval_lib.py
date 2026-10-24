"""Đánh giá một khối dữ liệu (dự báo + quyết định) bằng đúng một hàm, để mọi khối (policy_val, locked, OOT) dùng chung công thức."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import solver_lib as S
from config import *

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
    R = _to_mat(ds, "Y_short") * 1.0 + S.B_FULL * _to_mat(ds, "Y_full")
    V, VL = _to_mat(ds, "v"), _to_mat(ds, "v_du_phong")
    Z = {"Oracle": np.array([S.solve_milp(r) for r in R]), "GBM + MILP": np.array([S.solve_milp(v) for v in V]),
         "GBM + Greedy": np.array([S.solve_greedy(v) for v in V]), "Cùng ca tuần trước + MILP": np.array([S.solve_milp(v) for v in VL])}
    val = {k: (z * R).sum(1) for k, z in Z.items()}
    val["Ngẫu nhiên + MILP"] = np.mean([(np.array([S.solve_milp(r) for r in rng.random(R.shape)]) * R).sum(1) for _ in range(N_RAND)], axis=0)
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
    import router_lib as RL
    dp = d_all[d_all.khoi == blk]
    pbd = RL.perf_by_day(cperf, sorted(dp.ngay.unique()))
    fc, tab, boot, m, extra = evaluate(dp)
    rr = RL.router_run_and_test(dp, psi_df.set_index("ngay").psi_max.to_dict(), pbd)
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
