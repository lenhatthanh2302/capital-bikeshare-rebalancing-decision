"""Phân tích bổ sung SAU khi đã có kết quả locked/OOT (post-hoc). Không thay thế số chính thức, không huấn luyện lại mô hình đóng băng.
A. Độ nhạy lỗi đặc trưng city_*: tổng toàn thành phố dùng cả 22:00-23:59 của D-1. Sửa (chỉ giờ < 22), huấn luyện lại trên train với đúng tham số, đánh giá trên validation/policy_val.
B. Baseline công bằng hơn (lag5 chuẩn hóa theo τ, tỷ lệ sự kiện 20 ngày) và bootstrap theo khối tuần, trên mô hình đóng băng.
Chạy:  py -3 11_phan_tich_bo_sung.py     (kết quả vào Results/bo_sung/)"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, joblib, duckdb
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from config import *
import features_lib as FL
import solver_lib as S
import eval_lib as E
import router_lib as RL

warnings.filterwarnings("ignore")
OUT = RESULTS / "bo_sung"; OUT.mkdir(exist_ok=True)
bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
F = bundle["features"]
LBL = ["Y_short", "Y_full"]
rng = np.random.default_rng(SEED)


def lift_prauc(d, s, y, n_seed=20):
    rate = d[y].mean()
    pk = np.mean([E.precision_at_k(d, s, y, np.random.default_rng(i)).mean() for i in range(n_seed)])
    return average_precision_score(d[y], s), pk / rate


# ---------------------------------------------------------------- A
print("[A] Độ nhạy city_* (chỉ dùng dữ liệu đến hết policy_val)")
df_old, base, city_old = FL.build_dataset(DEV_END, F)
SH = (DATA / "station_hour.parquet").as_posix()
days = pd.bdate_range(BLOCKS["train"][0], DEV_END)
c2 = duckdb.connect().execute(f"""SELECT CAST(hour_ts AS DATE) AS ngay, sum(n_out) AS city_out FROM read_parquet('{SH}')
    WHERE isodow(hour_ts) <= 5 AND hour(hour_ts) < 22 AND CAST(hour_ts AS DATE) <= DATE '{DEV_END}' GROUP BY 1 ORDER BY 1""").df()
c2["ngay"] = pd.to_datetime(c2["ngay"]); c2 = c2.set_index("ngay").reindex(days, fill_value=0)
feat2 = FL.build_features(base, c2)
lbl = base[["station_id", "ngay", "ca", "net_out", "net_in", "Y_short", "Y_full", "khoi"]]
df_new = feat2.merge(lbl, on=["station_id", "ngay", "ca"], how="left")
df_new = df_new[df_new.ngay >= days[FL.WARMUP_DAYS]].dropna(subset=F).reset_index(drop=True)
assert len(df_new) == len(df_old)
chg = {f: float(np.nanmax(np.abs(df_new[f].values - df_old[f].values))) for f in ("city_out_lag1", "city_out_m20", "city_ratio")}
corr = {f: float(np.corrcoef(df_new[f], df_old[f])[0, 1]) for f in chg}
print("  chênh lệch lớn nhất:", chg, "| tương quan:", corr)

rows = []
for y in LBL:
    proto = bundle["models"][(y, "GBM")]
    tr_o, tr_n = df_old[df_old.khoi == "train"], df_new[df_new.khoi == "train"]
    m_ctrl = clone(proto).fit(tr_o[F], tr_o[y])      # đối chứng: huấn luyện lại với đặc trưng cũ, phải trùng mô hình đã lưu
    m_fix = clone(proto).fit(tr_n[F], tr_n[y])
    saved = proto.predict_proba(df_old[F])[:, 1]
    ctrl = m_ctrl.predict_proba(df_old[F])[:, 1]
    print(f"  {y}: đối chứng vs mô hình đã lưu, chênh lớn nhất {np.abs(saved - ctrl).max():.2e}")
    fix = m_fix.predict_proba(df_new[F])[:, 1]
    for blk in ("validation", "calibration", "policy_val"):
        msk = (df_old.khoi == blk).values
        d = df_old[msk]
        for nm, s in (("cũ (đã lưu)", saved[msk]), ("sửa city_* (<22h)", fix[msk])):
            pa, li = lift_prauc(d, s, y)
            rows.append({"nhan": y, "khoi": blk, "phien_ban": nm, "PR_AUC": pa, "Lift@8": li})
A = pd.DataFrame(rows)
A.round(4).to_csv(OUT / "bo_sung_A_city_do_nhay.csv", index=False, encoding="utf-8-sig")
print(A.pivot_table(index=["nhan", "khoi"], columns="phien_ban", values=["PR_AUC", "Lift@8"]).round(4).to_string())

# ---------------------------------------------------------------- B
print("\n[B] Baseline công bằng hơn + bootstrap theo tuần (mô hình đóng băng, không huấn luyện lại)")
df, base, city = FL.build_dataset(BLOCKS["oot"][1], F)
pred = E.predict_frozen(df[df.khoi != "train"], bundle)
d_all = RL.prep_cells(pred, df)
extra = df[["station_id", "ngay", "ca", "net_out_lag5_over_tau", "net_in_lag5_over_tau", "ev_short_r20", "ev_full_r20"]]
d_all = d_all.merge(extra, on=["station_id", "ngay", "ca"], how="left")

SC = {"Y_short": {"GBM": "p_Y_short_final", "Tuần trước (đếm thô)": "net_out_lag5", "Tuần trước / τ": "net_out_lag5_over_tau",
                  "Tỷ lệ sự kiện 20 ngày": "ev_short_r20", "Tỷ lệ lịch sử": "rate_Y_short"},
      "Y_full": {"GBM": "p_Y_full_final", "Tuần trước (đếm thô)": "net_in_lag5", "Tuần trước / τ": "net_in_lag5_over_tau",
                 "Tỷ lệ sự kiện 20 ngày": "ev_full_r20", "Tỷ lệ lịch sử": "rate_Y_full"}}
rows = []
for blk in ("validation", "policy_val", "locked", "oot"):
    d = d_all[d_all.khoi == blk]
    for y in LBL:
        for nm, col in SC[y].items():
            pa, li = lift_prauc(d, d[col].values.astype(float), y)
            rows.append({"khoi": blk, "nhan": y, "mo_hinh": nm, "PR_AUC": pa, "Lift@8": li})
B1 = pd.DataFrame(rows)
B1.round(4).to_csv(OUT / "bo_sung_B_baseline_du_bao.csv", index=False, encoding="utf-8-sig")
print(B1.pivot_table(index=["khoi", "nhan"], columns="mo_hinh", values="Lift@8").round(2).to_string())
print(B1.pivot_table(index=["khoi", "nhan"], columns="mo_hinh", values="PR_AUC").round(3).to_string())


def mat(ds, col): return ds[col].values.reshape(-1, ds.station_id.nunique())


def decision(d):
    ds = d.sort_values(["ngay", "ca_ord", "station_id"])
    dayv = ds.groupby(["ngay", "ca_ord"]).ngay.first().values
    R = mat(ds, "Y_short") * 1.0 + S.B_FULL * mat(ds, "Y_full")
    cl = lambda c: np.clip(mat(ds, c), 0, None)
    V = {"GBM + MILP": mat(ds, "v"), "Tuần trước (đếm thô) + MILP": mat(ds, "v_du_phong"),
         "Tuần trước / τ + MILP": cl("net_out_lag5_over_tau") + S.B_FULL * cl("net_in_lag5_over_tau"),
         "Tỷ lệ sự kiện 20 ngày + MILP": mat(ds, "ev_short_r20") + S.B_FULL * mat(ds, "ev_full_r20")}
    val = {"Oracle": (np.array([S.solve_milp(r) for r in R]) * R).sum(1)}
    for k, v in V.items(): val[k] = (np.array([S.solve_milp(x) for x in v]) * R).sum(1)
    val["GBM + Greedy"] = (np.array([S.solve_greedy(x) for x in V["GBM + MILP"]]) * R).sum(1)
    r2 = np.random.default_rng(SEED)
    val["Ngẫu nhiên + MILP"] = np.mean([(np.array([S.solve_milp(r) for r in r2.random(R.shape)]) * R).sum(1) for _ in range(E.N_RAND)], axis=0)
    return dayv, val


def boot(dayv, x, n=2000):
    s = pd.Series(x).groupby(dayv).mean()                       # theo ngày
    wk = s.index.to_period("W"); uw = wk.unique(); g = {w: s[wk == w].values for w in uw}
    r = np.random.default_rng(SEED)
    b_day = [s.reindex(r.choice(s.index, len(s))).mean() for _ in range(n)]
    b_wk = [np.concatenate([g[w] for w in r.choice(uw, len(uw))]).mean() for _ in range(n)]
    q = lambda b: (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    return float(s.mean()), q(b_day), q(b_wk), len(s), len(uw)


tabs, bts = [], []
for blk in ("policy_val", "locked", "oot"):
    dayv, val = decision(d_all[d_all.khoi == blk])
    orc = val["Oracle"].mean()
    for k, v in val.items(): tabs.append({"khoi": blk, "chinh_sach": k, "gia_tri_TB_moi_o": v.mean(), "ti_le_so_oracle": v.mean() / orc})
    for b in ("Ngẫu nhiên + MILP", "Tuần trước (đếm thô) + MILP", "Tuần trước / τ + MILP", "Tỷ lệ sự kiện 20 ngày + MILP", "GBM + Greedy"):
        m, (l1, u1), (l2, u2), nd, nw = boot(dayv, val["GBM + MILP"] - val[b])
        bts.append({"khoi": blk, "A": "GBM + MILP", "B": b, "chenh_TB": m, "CI_ngay_duoi": l1, "CI_ngay_tren": u1,
                    "CI_tuan_duoi": l2, "CI_tuan_tren": u2, "so_ngay": nd, "so_tuan": nw})
T, BT = pd.DataFrame(tabs), pd.DataFrame(bts)
T.round(4).to_csv(OUT / "bo_sung_B_gia_tri_quyet_dinh.csv", index=False, encoding="utf-8-sig")
BT.round(4).to_csv(OUT / "bo_sung_B_bootstrap_tuan.csv", index=False, encoding="utf-8-sig")
print(T.round(3).to_string(index=False)); print(BT.round(3).to_string(index=False))
