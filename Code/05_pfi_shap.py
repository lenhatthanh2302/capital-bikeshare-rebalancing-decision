#%% ======================================================================
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
import time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from config import *

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
