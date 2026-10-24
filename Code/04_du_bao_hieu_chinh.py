#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
# BƯỚC 4: Dự báo xác suất rủi ro (thiếu xe / đầy xe) và hiệu chỉnh (calibration)
#
# Mục tiêu: huấn luyện trên TRAIN, chọn mô hình trên VALIDATION, học bộ hiệu chỉnh trên
# CALIBRATION (tập riêng), kiểm tra hiệu chỉnh trên POLICY_VAL. Không đụng locked/oot.
# So với hai mốc đơn giản: "cùng ca tuần trước" và "tỷ lệ lịch sử của trạm-ca (train)".
# Chỉ số: PR-AUC, Precision@K (K = số trạm xe tải ghé được mỗi ca), Brier, log loss, ECE.
# Độ ổn định: 5 seed + bootstrap theo khối ngày (không bootstrap từng dòng vì các ô cùng ngày tương quan).
# ========================================================================
import time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from config import *

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
