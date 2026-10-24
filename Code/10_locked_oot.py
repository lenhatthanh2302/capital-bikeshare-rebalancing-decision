#%% ======================================================================
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
import sys, time, hashlib, platform, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import features_lib as FL
import router_lib as RL
import eval_lib as EV
from config import *

warnings.filterwarnings("ignore")
t_start = time.perf_counter()
MODE = sys.argv[1] if len(sys.argv) > 1 else "dry"
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
df, base, city = FL.build_dataset(END, FEATURES)
for k in (["policy_val"] if MODE == "dry" else ["locked", "oot"]):
    n_exp = len(FL.stations) * 2 * len(pd.bdate_range(*BLOCKS[k]))
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

pred = EV.predict_frozen(df[df.khoi != "train"], bundle)
po_old = pd.read_parquet(DATA / "pred_dev.parquet")
pm = po_old.merge(pred, on=["station_id", "ngay", "ca"], suffixes=("_o", "_n"))
assert len(pm) == len(po_old)
p_diff = max(float(np.abs(pm[f"p_{y}_final_o"] - pm[f"p_{y}_final_n"]).max()) for y in EV.LABELS)
p_diff = 0.0 if p_diff < 1e-12 else p_diff
rel.append({"kiem_tra": "Replay: xác suất dev tính lại trùng pred_dev.parquet", "gia_tri": p_diff, "ket_qua": "Đạt" if p_diff == 0 else "KHÔNG ĐẠT"})

rng_l = np.random.default_rng(SEED)
if MODE == "real":
    ld = list(rng_l.choice(df[df.khoi == "locked"].ngay.unique(), 2, replace=False)) + list(rng_l.choice(df[df.khoi == "oot"].ngay.unique(), 1))
else:
    ld = list(rng_l.choice(df[df.khoi == "policy_val"].ngay.unique(), 1))
leak = FL.leakage_check(df, base, city, FEATURES, sorted(ld))
leak.to_csv(RESULTS / f"{PFX}leakage_experiment_locked_oot.csv", index=False, encoding="utf-8-sig")
l_diff = float(leak.chenh_lech_lon_nhat.max())
rel.append({"kiem_tra": "Rò rỉ: ghi đè tương lai bằng 99999 không làm đổi đặc trưng", "gia_tri": l_diff, "ket_qua": "Đạt" if l_diff == 0 else "KHÔNG ĐẠT"})
print("[2a] Replay đặc trưng/xác suất và rò rỉ:"); print(pd.DataFrame(rel).to_string(index=False)); print(leak.to_string(index=False))

d_all = RL.prep_cells(pred, df)
psi_df = RL.psi_daily(d_all); cperf = RL.cell_perf(d_all)
psi_old = pd.read_csv(RESULTS / "router_psi_scores_daily.csv", parse_dates=["ngay"])
pm2 = psi_old.merge(psi_df, on="ngay", suffixes=("_o", "_n"))
assert len(pm2) == len(psi_old) and np.allclose(pm2.psi_max_o, pm2.psi_max_n)
dp_dev = d_all[d_all.khoi == "policy_val"]
plan_new, _ = RL.route(dp_dev, psi_df.set_index("ngay").psi_max.to_dict(), RL.perf_by_day(cperf, sorted(dp_dev.ngay.unique())))
plan_old = pd.read_csv(RESULTS / "router_plan_policy_val.csv", parse_dates=["ngay"])
cols = ["idempotency_key", "arm", "reason_code", "mode"]
r_diff = int((plan_old[cols].reset_index(drop=True) != plan_new[cols].reset_index(drop=True)).to_numpy().sum()) if len(plan_old) == len(plan_new) else 999999
rel.append({"kiem_tra": "Replay: router trên policy_val trùng router_plan_policy_val.csv (08)", "gia_tri": r_diff, "ket_qua": "Đạt" if r_diff == 0 else "KHÔNG ĐẠT"})
pd.DataFrame(rel).to_csv(RESULTS / f"{PFX}release_kiem_tra.csv", index=False, encoding="utf-8-sig")
print(f"[2b] Router replay: {r_diff} khác biệt; PSI dev trùng 08. Phiên bản mô hình {RL.MODEL_VERSION}")


#%% ======================================================================
# 3. LOCKED (hoặc policy_val ở chế độ dry): đánh giá một lần theo tiêu chí đã công bố
# ========================================================================
crit_path = RESULTS / "acceptance_criteria.csv"
crit = pd.read_csv(crit_path, dtype={"nguong": str}) if crit_path.exists() else None
res = {}
res[LOCK_BLK] = EV.full_block_metrics(d_all, psi_df, cperf, LOCK_BLK)
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
    out = EV.check_criteria(cl, {k: v['m'] for k, v in res.items()})
    out["khoi"] = "locked" if MODE == "real" else "policy_val (diễn tập)"
    out.to_csv(RESULTS / f"{PFX}acceptance_ket_qua_locked.csv", index=False, encoding="utf-8-sig")
    print("\n  Tiêu chí chấp nhận (locked):"); print(out[["ma", "noi_dung", "toan_tu", "nguong", "gia_tri_do_duoc", "ket_qua"]].to_string(index=False))
    n_fail = int((out.ket_qua != "Đạt").sum())
    print(f"  => {len(out) - n_fail}/{len(out)} đạt; kết luận: {'ĐẠT -> được phép canary' if n_fail == 0 else 'KHÔNG ĐẠT -> chỉ chạy bóng (shadow), không chỉnh lại trên locked'}")
if MODE == "real":
    LOCK_FILE.write_text(f"locked mở lúc {pd.Timestamp.now().isoformat()} bởi 10_locked_oot.py real; SHA manifest {MANIFEST_SHA}; "
                         f"model_version {RL.MODEL_VERSION}; kết luận {n_fail} tiêu chí không đạt\nKHÔNG chạy lại, KHÔNG chỉnh mô hình/ngưỡng dựa trên kết quả này.\n", encoding="utf-8")
    print(f"      Đã ghi khoá {LOCK_FILE.name}")


#%% ======================================================================
# 4. OOT (stress) + giám sát: PSI, hiệu quả theo tháng, điều kiện huấn luyện lại
# ========================================================================
mon = d_all[d_all.khoi.isin(["policy_val", "locked", "oot"])].copy()
mon["thang"] = mon.ngay.dt.strftime("%Y-%m")
rng_m = np.random.default_rng(SEED); mrows = []
for tm, g in mon.groupby("thang"):
    for y in EV.LABELS:
        rate = g[y].mean()
        mrows.append({"thang": tm, "label": y, "khoi": g.khoi.iloc[0], "ty_le_su_kien": rate,
                      "Lift_GBM": EV.precision_at_k(g, g[f"p_{y}_final"].values, y, rng_m).mean() / rate,
                      "Lift_lag5": EV.precision_at_k(g, g[EV.LAG[y]].values.astype(float), y, rng_m).mean() / rate,
                      "TB_xac_suat": g[f"p_{y}_final"].mean(), "tram_khong_hoat_dong": float((~g.hop_le).mean())})
mon_df = pd.DataFrame(mrows).round(4); mon_df.to_csv(RESULTS / f"{PFX}giam_sat_theo_thang.csv", index=False, encoding="utf-8-sig")
print("\n[4a] Giám sát theo tháng (lift@8, tỷ lệ sự kiện, trạm không hoạt động):"); print(mon_df.to_string(index=False))

# điều kiện huấn luyện lại: A = PSI>=0,25 liên tiếp >=5 ngày VÀ mô hình kém dự phòng (20 ô); B = mô hình kém dự phòng liên tiếp >=10 ngày; C = quá 3 tháng
days_m = psi_df[psi_df.khoi.isin(["policy_val", "locked", "oot"])].copy()
pbd = RL.perf_by_day(cperf, days_m.ngay)
days_m["mo_hinh_20"] = [pbd[d][0] for d in days_m.ngay]; days_m["du_phong_20"] = [pbd[d][1] for d in days_m.ngay]
days_m["kem"] = days_m.mo_hinh_20 < days_m.du_phong_20
run = lambda s: s.groupby((~s).cumsum()).cumsum()
days_m["psi_lien_tiep"] = run(days_m.psi_max >= RL.PSI_STOP); days_m["kem_lien_tiep"] = run(days_m.kem)
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
res[OOT_BLK] = EV.full_block_metrics(d_all, psi_df, cperf, OOT_BLK)
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
    out_o = EV.check_criteria(co, {k: v["m"] for k, v in res.items()}); out_o["khoi"] = "oot" if MODE == "real" else "calibration (diễn tập)"
    out_o.to_csv(RESULTS / f"{PFX}acceptance_ket_qua_oot.csv", index=False, encoding="utf-8-sig")
    print("\n  Tiêu chí OOT (stress, nới hơn locked):"); print(out_o[["ma", "noi_dung", "toan_tu", "nguong", "gia_tri_do_duoc", "ket_qua"]].to_string(index=False))

# hình 13: PSI + lift theo tháng
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
cols_b = {"policy_val": "#8A8A8A", "locked": "#D9822B", "oot": "#1F3A5F"}
for k, c in cols_b.items():
    t = days_m[days_m.khoi == k]
    if len(t): ax[0].plot(t.ngay, t.psi_max, color=c, label={"policy_val": "policy_val", "locked": "locked", "oot": "OOT"}[k])
ax[0].axhline(RL.PSI_WARN, color="k", ls=":", lw=1); ax[0].axhline(RL.PSI_STOP, color="r", ls="--", lw=1)
ax[0].set_title("Hình 13a. PSI điểm dự báo theo ngày"); ax[0].legend(fontsize=8); ax[0].tick_params(axis="x", rotation=30)
for y, ls in (("Y_short", "-"), ("Y_full", "--")):
    t = mon_df[mon_df.label == y]
    ax[1].plot(t.thang, t.Lift_GBM, ls, marker="o", color="#1F3A5F", label=f"GBM – {'thiếu' if y == 'Y_short' else 'đầy'}")
    ax[1].plot(t.thang, t.Lift_lag5, ls, marker="s", color="#8A8A8A", label=f"Cùng ca tuần trước – {'thiếu' if y == 'Y_short' else 'đầy'}")
ax[1].axhline(1, color="k", ls=":", lw=1); ax[1].set_title("Hình 13b. Lift@8 theo tháng (policy_val → OOT)"); ax[1].legend(fontsize=7); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(RESULTS / f"{PFX}giam_sat_hinh13_psi_lift.png"); plt.close()

pd.DataFrame([{"python": platform.python_version(), "che_do": MODE, "model_version": RL.MODEL_VERSION, "seed": SEED, "thoi_diem": pd.Timestamp.now().isoformat()}]) \
    .to_csv(RESULTS / f"{PFX}chay_locked_oot_ghi_chu.csv", index=False, encoding="utf-8-sig")
print(f"[✓] BƯỚC 10 ({MODE}) hoàn tất sau {time.perf_counter() - t_start:.0f}s")
