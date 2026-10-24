#%% ======================================================================
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
import time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from config import *

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
