#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
# BƯỚC 3: Lập lưới (trạm, ngày thường, ca), nhãn Y và đặc trưng tại t0
#
# Mục tiêu: tạo bảng dữ liệu cho mô hình, trong đó mọi đặc trưng của ngày D chỉ dùng
# thông tin có trước t0 = 22:00 ngày D−1. Kỹ thuật: mọi đặc trưng đều là shift(1) theo
# ngày làm việc (lag/rolling kết thúc ở ngày làm việc liền trước), nên không thể chứa
# kết quả của ngày D. Cuối file có thí nghiệm kiểm tra rò rỉ để chứng minh điều đó.
# Chỉ dựng đến hết policy_val (DEV_END); locked/oot chưa được đụng tới.
# ========================================================================
import time
import numpy as np
import pandas as pd
import duckdb
from config import *

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
