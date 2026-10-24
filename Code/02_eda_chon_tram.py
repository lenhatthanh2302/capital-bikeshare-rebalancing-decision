#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
# BƯỚC 2: Khám phá dữ liệu (EDA) và chọn tập trạm cho bài toán tối ưu
#
# Mục tiêu: (a) vẽ các hình mô tả dữ liệu cho báo cáo; (b) chọn 80 trạm chỉ dựa
# vào giai đoạn TRAIN (202401–202510) để không rò rỉ thông tin từ validation/test;
# (c) xem phân phối dòng xe ròng theo ca để biết việc chọn ngưỡng τ có khả thi không.
# Mọi thống kê dùng để chọn trạm/ngưỡng chỉ tính trên train.
# ========================================================================
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
TRAIN_START, TRAIN_END = "2024-01-01", "2025-11-01"     # train = 202401–202510 (nửa mở)
N_STATIONS = 80
SHIFTS = {"sang": (6, 10), "chieu": (16, 20)}            # giờ bắt đầu (gồm) -> kết thúc (không gồm)
COLOR = {"main": "#1F3A5F", "alt": "#D9822B", "grey": "#8A8A8A"}

BASE = Path(__file__).resolve().parent.parent
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
