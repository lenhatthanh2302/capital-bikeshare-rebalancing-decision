#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare (từ dự báo nhu cầu đến điều xe)
# BƯỚC 1: Gộp 33 file CSV -> parquet, làm sạch, đối chiếu nhất quán
#
# Mục tiêu: biến 33 file CSV (~3,2 GB, ~17,6 triệu chuyến) thành các bảng nhẹ để
# mọi bước sau chỉ đọc parquet, đồng thời để lại bằng chứng tái lập (SHA-256 từng
# file gốc, số dòng bị loại theo từng lý do, các phép đối chiếu Đạt/Không).
# Chưa chia train/test và chưa tính bất kỳ ngưỡng nào ở bước này, nên chưa có rò rỉ.
# ========================================================================
import os, time, hashlib, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pandas as pd

SEED = 42
MIN_DUR_S = 60            # chuyến dưới 60 giây coi là lỗi/ghi nhầm (trang nguồn cũng nói đã loại)
MAX_DUR_S = 24 * 3600     # chuyến trên 24 giờ coi là xe không được trả đúng cách
TONG_DONG_GOC_DA_BIET = 17_605_262   # số dòng thu được ở lần khảo sát trước, dùng để đối chiếu

BASE = Path(__file__).resolve().parent.parent          # thư mục "Bài nhóm"
SRC = BASE / "Datasets"
DATA = BASE / "Data"
RESULTS = BASE / "Results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

files = sorted(SRC.glob("*-capitalbikeshare-tripdata.csv"))
assert len(files) == 33, f"Cần 33 file, đang có {len(files)}"
t_start = time.perf_counter()
print(f"[0] Tìm thấy {len(files)} file CSV: {files[0].name} ... {files[-1].name}")


#%% ======================================================================
# 1. SHA-256 từng file gốc
# Mục tiêu: ghi dấu vân tay dữ liệu trước khi đụng vào, để Chương 8 chứng minh
# được dữ liệu đầu vào không bị sửa giữa lúc làm bài và lúc nộp.
# ========================================================================
def sha256_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as ex:       # hashlib nhả GIL nên chạy luồng là đủ nhanh
    hashes = list(ex.map(sha256_of, files))
hash_df = pd.DataFrame({
    "file_month": [f.name[:6] for f in files],
    "file_name": [f.name for f in files],
    "bytes": [f.stat().st_size for f in files],
    "sha256": hashes,
})
print(f"[1/6] SHA-256 xong {len(files)} file trong {time.perf_counter() - t0:.1f}s")


#%% ======================================================================
# 2. Nạp CSV vào DuckDB (dạng chuỗi rồi ép kiểu)
# Mục tiêu: đọc cả 33 file một lần. Đọc thành chuỗi trước rồi TRY_CAST để dòng lỗi
# timestamp không làm hỏng cả file (từ 202406 timestamp có thêm mili-giây, trước đó thì không).
# Ô trống ở cột trạm là NULL thật: đây là các chuyến e-bike không đỗ tại trạm.
# ========================================================================
work_db = Path(tempfile.gettempdir()) / "bikeshare_work.duckdb"   # DB tạm nằm ngoài OneDrive để khỏi đồng bộ file lớn
if work_db.exists():
    work_db.unlink()
con = duckdb.connect(str(work_db))
con.execute("SET memory_limit='5GB'")
con.execute("SET threads=8")
con.execute(f"SET temp_directory='{(Path(tempfile.gettempdir()) / 'duckdb_spill').as_posix()}'")

paths_sql = "[" + ",".join("'" + f.as_posix().replace("'", "''") + "'" for f in files) + "]"
t0 = time.perf_counter()
con.execute(f"""
CREATE TABLE raw AS
SELECT
  ride_id,
  rideable_type,
  TRY_CAST(started_at AS TIMESTAMP) AS started_at,
  TRY_CAST(ended_at   AS TIMESTAMP) AS ended_at,
  NULLIF(trim(start_station_name), '') AS start_station_name,
  NULLIF(trim(start_station_id),   '') AS start_station_id,
  NULLIF(trim(end_station_name),   '') AS end_station_name,
  NULLIF(trim(end_station_id),     '') AS end_station_id,
  TRY_CAST(start_lat AS DOUBLE) AS start_lat,
  TRY_CAST(start_lng AS DOUBLE) AS start_lng,
  TRY_CAST(end_lat   AS DOUBLE) AS end_lat,
  TRY_CAST(end_lng   AS DOUBLE) AS end_lng,
  member_casual,
  regexp_extract(filename, '(\\d{{6}})-capitalbikeshare', 1) AS file_month
FROM read_csv({paths_sql}, all_varchar=true, header=true, filename=true)
""")
n_raw = con.execute("SELECT count(*) FROM raw").fetchone()[0]
print(f"[2/6] Nạp xong {n_raw:,} dòng trong {time.perf_counter() - t0:.1f}s")


#%% ======================================================================
# 3. Gắn nhãn lý do loại (mỗi dòng chỉ mang MỘT lý do, theo thứ tự ưu tiên)
# Mục tiêu: biết chính xác bao nhiêu dòng bị loại và vì sao; quy tắc áp dụng giống
# nhau cho mọi tháng, nên sau này chia tập xong không phải làm sạch lại.
# Trùng ride_id chỉ xét giữa các dòng đã hợp lệ (nếu bản đầu bị loại vì lỗi thì
# bản sau không bị coi là trùng). Giữ bản ở file sớm hơn.
# ========================================================================
t0 = time.perf_counter()
con.execute(f"""
CREATE TABLE flagged AS
WITH a AS (
  SELECT *,
    CASE
      WHEN started_at IS NULL OR ended_at IS NULL                    THEN '1_timestamp_loi'
      WHEN ended_at < started_at                                     THEN '2_thoi_luong_am'
      WHEN date_diff('second', started_at, ended_at) < {MIN_DUR_S}   THEN '3_duoi_60s'
      WHEN date_diff('second', started_at, ended_at) > {MAX_DUR_S}   THEN '4_tren_24h'
    END AS reason1
  FROM raw
), b AS (
  SELECT *,
    row_number() OVER (PARTITION BY ride_id, (reason1 IS NULL)
                       ORDER BY file_month, started_at) AS rn
  FROM a
)
SELECT * EXCLUDE (reason1, rn),
       COALESCE(reason1, CASE WHEN rn > 1 THEN '5_ride_id_trung' END) AS reason
FROM b
""")
con.execute("DROP TABLE raw")

reason_overall = con.execute("""
SELECT COALESCE(reason, '0_giu_lai') AS ly_do, count(*) AS so_dong
FROM flagged GROUP BY 1 ORDER BY 1
""").df()
reason_overall["ty_le_%"] = (reason_overall["so_dong"] / n_raw * 100).round(3)
print(reason_overall.to_string(index=False))

per_file = con.execute("""
SELECT file_month,
       count(*)                                         AS dong_goc,
       count(*) FILTER (WHERE reason = '1_timestamp_loi') AS timestamp_loi,
       count(*) FILTER (WHERE reason = '2_thoi_luong_am') AS thoi_luong_am,
       count(*) FILTER (WHERE reason = '3_duoi_60s')      AS duoi_60s,
       count(*) FILTER (WHERE reason = '4_tren_24h')      AS tren_24h,
       count(*) FILTER (WHERE reason = '5_ride_id_trung') AS ride_id_trung,
       count(*) FILTER (WHERE reason IS NULL)             AS dong_giu_lai
FROM flagged GROUP BY 1 ORDER BY 1
""").df()
audit = hash_df.merge(per_file, on="file_month", how="left")
audit.to_csv(RESULTS / "data_audit.csv", index=False, encoding="utf-8-sig")
reason_overall.to_csv(RESULTS / "data_cleaning_summary.csv", index=False, encoding="utf-8-sig")
print(f"[3/6] Gắn nhãn xong trong {time.perf_counter() - t0:.1f}s")


#%% ======================================================================
# 4. Xuất các bảng parquet
# Mục tiêu: chỉ giữ cột cần. trips_clean để tính biến theo mốc t0 sau này (ví dụ
# chỉ dùng chuyến có ended_at <= t0); station_hour là bảng thưa trạm x giờ (ô
# bằng 0 sẽ được thêm ở bước lập lưới, từ ngày trạm xuất hiện lần đầu);
# nostation_hour giữ riêng dòng chuyến không có mã trạm vì không gán được vào trạm.
# ========================================================================
t0 = time.perf_counter()
con.execute("CREATE TABLE clean AS SELECT * EXCLUDE (reason, start_station_name, end_station_name) "
            "FROM flagged WHERE reason IS NULL")
con.execute("CREATE TABLE names AS SELECT start_station_id, start_station_name, end_station_id, end_station_name "
            "FROM flagged WHERE reason IS NULL")
con.execute("DROP TABLE flagged")

p_trips = (DATA / "trips_clean.parquet").as_posix()
con.execute(f"""
COPY (SELECT ride_id, rideable_type, member_casual, started_at, ended_at,
             start_station_id, end_station_id, start_lat, start_lng, end_lat, end_lng, file_month
      FROM clean ORDER BY started_at)
TO '{p_trips}' (FORMAT parquet, COMPRESSION zstd)
""")

p_sh = (DATA / "station_hour.parquet").as_posix()
con.execute(f"""
COPY (
  WITH o AS (SELECT start_station_id AS station_id, date_trunc('hour', started_at) AS hour_ts, count(*) AS n_out
             FROM clean WHERE start_station_id IS NOT NULL GROUP BY 1, 2),
       i AS (SELECT end_station_id AS station_id, date_trunc('hour', ended_at) AS hour_ts, count(*) AS n_in
             FROM clean WHERE end_station_id IS NOT NULL GROUP BY 1, 2)
  SELECT COALESCE(o.station_id, i.station_id) AS station_id,
         COALESCE(o.hour_ts,    i.hour_ts)    AS hour_ts,
         COALESCE(n_out, 0) AS n_out,
         COALESCE(n_in,  0) AS n_in
  FROM o FULL JOIN i ON o.station_id = i.station_id AND o.hour_ts = i.hour_ts
  ORDER BY station_id, hour_ts
) TO '{p_sh}' (FORMAT parquet, COMPRESSION zstd)
""")

p_ns = (DATA / "nostation_hour.parquet").as_posix()
con.execute(f"""
COPY (
  SELECT 'out' AS huong, date_trunc('hour', started_at) AS hour_ts, rideable_type, count(*) AS n
  FROM clean WHERE start_station_id IS NULL GROUP BY 2, 3
  UNION ALL
  SELECT 'in',  date_trunc('hour', ended_at),   rideable_type, count(*)
  FROM clean WHERE end_station_id IS NULL GROUP BY 2, 3
  ORDER BY 1, 2, 3
) TO '{p_ns}' (FORMAT parquet, COMPRESSION zstd)
""")

# Một id có thể mang nhiều tên theo thời gian -> lấy tên xuất hiện nhiều nhất; toạ độ lấy trung vị.
p_st = (DATA / "stations.parquet").as_posix()
con.execute(f"""
COPY (
  WITH pts AS (
    SELECT c.start_station_id AS station_id, c.start_lat AS lat, c.start_lng AS lng, c.started_at AS ts
    FROM clean c WHERE c.start_station_id IS NOT NULL
    UNION ALL
    SELECT c.end_station_id, c.end_lat, c.end_lng, c.ended_at
    FROM clean c WHERE c.end_station_id IS NOT NULL
  ), nm AS (
    SELECT station_id, mode(name) AS station_name FROM (
      SELECT start_station_id AS station_id, start_station_name AS name FROM names WHERE start_station_id IS NOT NULL
      UNION ALL
      SELECT end_station_id, end_station_name FROM names WHERE end_station_id IS NOT NULL
    ) GROUP BY 1
  )
  SELECT p.station_id, nm.station_name,
         median(p.lat) AS lat, median(p.lng) AS lng,
         min(p.ts) AS first_seen, max(p.ts) AS last_seen, count(*) AS n_su_kien
  FROM pts p LEFT JOIN nm USING (station_id)
  GROUP BY p.station_id, nm.station_name ORDER BY p.station_id
) TO '{p_st}' (FORMAT parquet, COMPRESSION zstd)
""")
print(f"[4/6] Xuất parquet xong trong {time.perf_counter() - t0:.1f}s")
for p in [p_trips, p_sh, p_ns, p_st]:
    print(f"      {Path(p).name:24s} {os.path.getsize(p) / 1e6:8.1f} MB")


#%% ======================================================================
# 5. Phép đối chiếu nhất quán giữa các bảng
# Mục tiêu: lỗi khi gộp 17,6 triệu dòng rất khó thấy bằng mắt, nên mỗi bảng phải
# khớp lại với bảng gốc bằng một phép kiểm có kết quả Đạt/Không.
# ========================================================================
checks = []
def add_check(ten, ky_vong, thuc_te, dung=None):
    ok = (ky_vong == thuc_te) if dung is None else dung
    checks.append({"phep_kiem": ten, "ky_vong": ky_vong, "thuc_te": thuc_te, "ket_qua": "Đạt" if ok else "KHÔNG ĐẠT"})

q = lambda sql: con.execute(sql).fetchone()[0]
reason_sum = int(reason_overall.loc[reason_overall["ly_do"] != "0_giu_lai", "so_dong"].sum())
n_clean = q("SELECT count(*) FROM clean")

add_check("Tổng dòng 33 file = số dòng đã biết từ khảo sát trước", TONG_DONG_GOC_DA_BIET, n_raw)
add_check("Tổng dòng theo từng file cộng lại = tổng dòng nạp", n_raw, int(audit["dong_goc"].sum()))
add_check("Dòng gốc = dòng giữ lại + dòng bị loại", n_raw, n_clean + reason_sum)
add_check("trips_clean.parquet có đúng số dòng giữ lại", n_clean,
          q(f"SELECT count(*) FROM read_parquet('{p_trips}')"))
add_check("ride_id duy nhất trong dữ liệu sạch", n_clean,
          q(f"SELECT count(DISTINCT ride_id) FROM read_parquet('{p_trips}')"))
add_check("Thời lượng nhỏ nhất trong dữ liệu sạch >= 60s",
          f">= {MIN_DUR_S}", q(f"SELECT min(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')"),
          dung=q(f"SELECT min(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')") >= MIN_DUR_S)
add_check("Thời lượng lớn nhất trong dữ liệu sạch <= 24h",
          f"<= {MAX_DUR_S}", q(f"SELECT max(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')"),
          dung=q(f"SELECT max(date_diff('second', started_at, ended_at)) FROM read_parquet('{p_trips}')") <= MAX_DUR_S)
add_check("Tổng n_out (station_hour) = số chuyến sạch có mã trạm đi",
          q("SELECT count(*) FROM clean WHERE start_station_id IS NOT NULL"),
          q(f"SELECT sum(n_out) FROM read_parquet('{p_sh}')"))
add_check("Tổng n_in (station_hour) = số chuyến sạch có mã trạm đến",
          q("SELECT count(*) FROM clean WHERE end_station_id IS NOT NULL"),
          q(f"SELECT sum(n_in) FROM read_parquet('{p_sh}')"))
add_check("nostation_hour (out) = số chuyến sạch thiếu mã trạm đi",
          q("SELECT count(*) FROM clean WHERE start_station_id IS NULL"),
          q(f"SELECT coalesce(sum(n),0) FROM read_parquet('{p_ns}') WHERE huong='out'"))
add_check("nostation_hour (in) = số chuyến sạch thiếu mã trạm đến",
          q("SELECT count(*) FROM clean WHERE end_station_id IS NULL"),
          q(f"SELECT coalesce(sum(n),0) FROM read_parquet('{p_ns}') WHERE huong='in'"))
add_check("Chuyến thiếu mã trạm đi mà KHÔNG phải e-bike (kỳ vọng 0)", 0,
          q("SELECT count(*) FROM clean WHERE start_station_id IS NULL AND rideable_type <> 'electric_bike'"))
add_check("Mỗi (trạm, giờ) xuất hiện đúng một dòng trong station_hour",
          q(f"SELECT count(*) FROM read_parquet('{p_sh}')"),
          q(f"SELECT count(*) FROM (SELECT DISTINCT station_id, hour_ts FROM read_parquet('{p_sh}'))"))
add_check("Mọi trạm trong station_hour đều có trong stations.parquet", 0,
          q(f"SELECT count(*) FROM (SELECT DISTINCT station_id FROM read_parquet('{p_sh}')) s "
            f"WHERE s.station_id NOT IN (SELECT station_id FROM read_parquet('{p_st}'))"))
add_check("Đủ 33 tháng trong dữ liệu sạch", 33,
          q(f"SELECT count(DISTINCT file_month) FROM read_parquet('{p_trips}')"))

chk = pd.DataFrame(checks)
chk.to_csv(RESULTS / "consistency_checks.csv", index=False, encoding="utf-8-sig")
print(chk.to_string(index=False))
n_fail = int((chk["ket_qua"] != "Đạt").sum())
print(f"[5/6] {len(chk) - n_fail}/{len(chk)} phép kiểm Đạt")


#%% ======================================================================
# 6. Tóm tắt nhanh để dùng trong báo cáo
# ========================================================================
n_st = q(f"SELECT count(*) FROM read_parquet('{p_st}')")
rng = con.execute(f"SELECT min(started_at), max(started_at) FROM read_parquet('{p_trips}')").fetchone()
print(f"      Chuyến sạch: {n_clean:,} / {n_raw:,} ({n_clean / n_raw * 100:.2f}%)  | số trạm: {n_st}")
print(f"      Khoảng thời gian: {rng[0]} -> {rng[1]}")
con.close()
work_db.unlink(missing_ok=True)
print(f"[✓] BƯỚC 1 hoàn tất sau {time.perf_counter() - t_start:.1f}s"
      + ("" if n_fail == 0 else f"  – CÓ {n_fail} PHÉP KIỂM KHÔNG ĐẠT, cần xem lại!"))
