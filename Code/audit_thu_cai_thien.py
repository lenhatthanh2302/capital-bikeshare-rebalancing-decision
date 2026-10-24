#%% ======================================================================
# KIỂM TRA / THỬ CẢI THIỆN dự báo (chỉ train -> validation; không đụng calibration/policy_val/locked/oot)
# Mục tiêu: xem thêm (a) đặc trưng tương đối so với ngưỡng τ của trạm, (b) tần suất sự kiện gần đây,
# (c) thời tiết dự báo của ngày D có giúp validation hơn bản hiện tại không.
# ========================================================================
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss
from config import *

warnings.filterwarnings("ignore")
LABELS = ["Y_short", "Y_full"]
SEEDS = [0, 1, 2]

df = pd.read_parquet(DATA / "features_dev.parquet")
BASE_F = pd.read_csv(RESULTS / "feature_list.csv").feature.tolist()
df = df[df.khoi.isin(["train", "validation"])].copy()   # chủ ý: chỉ hai khối này

# ---- (a)+(b) đặc trưng tương đối so với τ và tần suất vượt ngưỡng gần đây
new_tau = []
for v, t in (("net_out", "tau_out"), ("net_in", "tau_in")):
    for k in (1, 2, 5):
        df[f"{v}_lag{k}_over_tau"] = df[f"{v}_lag{k}"] / df[t].clip(lower=1)
        new_tau.append(f"{v}_lag{k}_over_tau")
    for w in (5, 10, 20, 60):
        df[f"{v}_m{w}_over_tau"] = df[f"{v}_m{w}"] / df[t].clip(lower=1)
        new_tau.append(f"{v}_m{w}_over_tau")
    df[f"{t}_feat"] = df[t]; new_tau.append(f"{t}_feat")

# tần suất vượt τ trong N ngày làm việc trước đó (shift(1) nên không thấy ngày D)
df = df.sort_values(["station_id", "ca", "ngay"]).reset_index(drop=True)
g = df.groupby(["station_id", "ca"], sort=False)
for y, nm in (("Y_short", "ev_short"), ("Y_full", "ev_full")):
    s1 = g[y].shift(1)
    for w in (10, 20, 60):
        df[f"{nm}_r{w}"] = s1.groupby([df.station_id, df.ca], sort=False).transform(lambda s: s.rolling(w, min_periods=5).mean())
        new_tau.append(f"{nm}_r{w}")
    df[f"{nm}_lag1"] = s1; new_tau.append(f"{nm}_lag1")

# ---- (c) thời tiết dự báo (Open-Meteo historical forecast) cho khung giờ của ca ngày D và ngày D-1
w = pd.read_parquet(DATA / "weather_hourly.parquet")
w["ngay"] = w.time.dt.normalize(); w["h"] = w.time.dt.hour
rows = []
for ca, (a, b) in SHIFTS.items():
    x = w[(w.h >= a) & (w.h < b)].groupby("ngay").agg(
        w_temp=("temperature_2m", "mean"), w_rain=("precipitation", "sum"), w_wind=("wind_speed_10m", "max")).reset_index()
    x["ca"] = ca; rows.append(x)
wx = pd.concat(rows)
df = df.merge(wx, on=["ngay", "ca"], how="left")
df["w_rain_flag"] = (df.w_rain >= 0.5).astype(int)
weather = ["w_temp", "w_rain", "w_wind", "w_rain_flag"]
print("thiếu thời tiết:", df[weather].isna().mean().round(3).to_dict())

tr, va = df[df.khoi == "train"], df[df.khoi == "validation"]


def p_at_k(d, s, y, k=K_PER_SHIFT):
    t = d[["ngay", "ca", y]].copy(); t["s"] = np.asarray(s)
    t["r"] = t.groupby(["ngay", "ca"]).s.rank(ascending=False, method="first")
    return t[t.r <= k].groupby(["ngay", "ca"])[y].mean().mean()


def run(feats, name):
    out = {}
    for y in LABELS:
        ap, pk, ll = [], [], []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=50,
                                               max_features=0.7, max_depth=5, max_iter=300, random_state=sd)
            p = m.fit(tr[feats], tr[y]).predict_proba(va[feats])[:, 1]
            ap.append(average_precision_score(va[y], p)); pk.append(p_at_k(va, p, y))
            ll.append(log_loss(va[y], np.clip(p, 1e-6, 1 - 1e-6)))
        out[y] = (np.mean(ap), np.mean(pk), np.mean(ll))
    print(f"{name:34s} | short AP {out['Y_short'][0]:.4f} P@8 {out['Y_short'][1]:.4f} LL {out['Y_short'][2]:.4f}"
          f" | full AP {out['Y_full'][0]:.4f} P@8 {out['Y_full'][1]:.4f} LL {out['Y_full'][2]:.4f}")
    return {"bo_dac_trung": name, **{f"{y}_{k}": v for y in LABELS for k, v in zip(("PR_AUC", "P@8", "LogLoss"), out[y])}}


res = [run(BASE_F, "A. hiện tại (51 đặc trưng)"),
       run(BASE_F + weather, "B. + thời tiết"),
       run(BASE_F + new_tau, "C. + tương đối τ + tần suất sự kiện"),
       run(BASE_F + new_tau + weather, "D. C + thời tiết")]
pd.DataFrame(res).round(4).to_csv(RESULTS / "audit_cai_thien_validation.csv", index=False, encoding="utf-8-sig")
print("[✓] xong")
