"""Dựng bảng đặc trưng đến một ngày bất kỳ (bản sao đúng logic của 03_dac_trung.py, dùng τ đã đóng băng).
Tính đúng đắn của bản sao được kiểm bằng replay test ở 10_locked_oot.py: các dòng dev dựng lại phải trùng features_dev.parquet."""
import numpy as np
import pandas as pd
import duckdb
from config import *

LAGS, WINS, VARS, WARMUP_DAYS = [1, 2, 3, 4, 5], [5, 10, 20], ["net_out", "net_in", "n_out", "n_in"], 20
sel = pd.read_csv(RESULTS / "stations_selected.csv", dtype={"station_id": str})
stations = sel.station_id.tolist()


def load_grid(end_date):
    """Lưới (trạm, ngày thường, ca) và tổng chuyến toàn hệ thống, chỉ đọc dữ liệu đến end_date."""
    sid = "(" + ",".join("'" + s + "'" for s in stations) + ")"
    con = duckdb.connect(); SH = (DATA / "station_hour.parquet").as_posix()
    case_shift = " ".join(f"WHEN hour(hour_ts) >= {a} AND hour(hour_ts) < {b} THEN '{s}'" for s, (a, b) in SHIFTS.items())
    agg = con.execute(f"""
    SELECT station_id, CAST(hour_ts AS DATE) AS ngay, CASE {case_shift} END AS ca, sum(n_out) AS n_out, sum(n_in) AS n_in
    FROM read_parquet('{SH}') WHERE station_id IN {sid} AND isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{end_date}'
    GROUP BY 1, 2, 3 HAVING ca IS NOT NULL""").df()
    agg["ngay"] = pd.to_datetime(agg["ngay"])
    days = pd.bdate_range(BLOCKS["train"][0], end_date)
    idx = pd.MultiIndex.from_product([stations, days, list(SHIFTS)], names=["station_id", "ngay", "ca"])
    base = agg.set_index(["station_id", "ngay", "ca"]).reindex(idx, fill_value=0).reset_index()
    base[["n_out", "n_in"]] = base[["n_out", "n_in"]].astype(int)
    base["net_out"] = base.n_out - base.n_in
    base["net_in"] = -base.net_out
    city = con.execute(f"""SELECT CAST(hour_ts AS DATE) AS ngay, sum(n_out) AS city_out FROM read_parquet('{SH}')
    WHERE isodow(hour_ts) <= 5 AND CAST(hour_ts AS DATE) <= DATE '{end_date}' GROUP BY 1 ORDER BY 1""").df()
    city["ngay"] = pd.to_datetime(city["ngay"])
    city = city.set_index("ngay").reindex(days, fill_value=0)
    return base, city, days


def add_tau_labels(base):
    tau = pd.read_csv(RESULTS / "tau_thresholds.csv", dtype={"station_id": str})     # τ đóng băng, học trên train ở bước 3
    base = base.merge(tau, on=["station_id", "ca"], how="left")
    base["Y_short"] = ((base.net_out >= base.tau_out) & (base.net_out > 0)).astype(int)
    base["Y_full"] = ((base.net_in >= base.tau_in) & (base.net_in > 0)).astype(int)
    base["khoi"] = block_of(base.ngay)
    return base


def build_features(b, c):
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
    opp = b[["station_id", "ngay", "ca", "net_out", "n_out", "n_in"]].copy()
    opp["ca"] = opp.ca.map({"sang": "chieu", "chieu": "sang"})
    nxt = {d: n for d, n in zip(c.index[:-1], c.index[1:])}
    opp["ngay"] = opp.ngay.map(nxt)
    opp = opp.dropna(subset=["ngay"]).rename(columns={"net_out": "opp_net_out_lag1", "n_out": "opp_n_out_lag1", "n_in": "opp_n_in_lag1"})
    f = f.merge(opp, on=["station_id", "ngay", "ca"], how="left")
    cs = c.city_out.shift(1)
    cf = pd.DataFrame({"city_out_lag1": cs, "city_out_m20": cs.rolling(20, min_periods=20).mean()})
    cf["city_ratio"] = cf.city_out_lag1 / cf.city_out_m20
    f = f.merge(cf.reset_index().rename(columns={"index": "ngay"}), on="ngay", how="left")
    f["dow"] = f.ngay.dt.dayofweek
    f["thang"] = f.ngay.dt.month
    f["thang_sin"] = np.sin(2 * np.pi * f.thang / 12)
    f["thang_cos"] = np.cos(2 * np.pi * f.thang / 12)
    f["la_ca_sang"] = (f.ca == "sang").astype(int)
    f["la_ngay_le"] = f.ngay.isin(HOLIDAYS).astype(int)
    prev = {n: d for d, n in nxt.items()}
    f["hom_qua_la_le"] = f.ngay.map(prev).isin(HOLIDAYS).astype(int)
    f = f.merge(sel.set_index("station_id")[["lat", "lng"]], left_on="station_id", right_index=True, how="left")
    return f


def build_dataset(end_date, features):
    """Bảng (đặc trưng + nhãn + khối) đến end_date, đã bỏ 20 ngày khởi động; trả thêm (base, city) để chạy thí nghiệm rò rỉ."""
    base, city, days = load_grid(end_date)
    base = add_tau_labels(base)
    feat = build_features(base, city)
    lbl = base[["station_id", "ngay", "ca", "net_out", "net_in", "Y_short", "Y_full", "khoi"]]
    df = feat.merge(lbl, on=["station_id", "ngay", "ca"], how="left")
    df = df[df.ngay >= days[WARMUP_DAYS]].dropna(subset=features)
    return df.reset_index(drop=True), base, city


def leakage_check(df, base, city, features, days_to_test):
    """Ghi đè mọi kết quả từ ngày D trở đi bằng 99999, dựng lại, so với bảng gốc: chênh lệch phải bằng 0."""
    rows = []
    for d in days_to_test:
        d = pd.Timestamp(d)
        b2, c2 = base.copy(), city.copy()
        for v in VARS: b2.loc[b2.ngay >= d, v] = 99999
        c2.loc[c2.index >= d, "city_out"] = 99999
        f2 = build_features(b2, c2)
        a = f2[f2.ngay == d].set_index(["station_id", "ca"])[features].sort_index()
        o = df[df.ngay == d].set_index(["station_id", "ca"])[features].sort_index()
        diff = float(np.nanmax(np.abs(a.values.astype(float) - o.values.astype(float))))
        rows.append({"ngay_kiem_tra": d.date(), "khoi": df[df.ngay == d].khoi.iloc[0], "chenh_lech_lon_nhat": diff,
                     "ket_qua": "Đạt" if diff == 0 else "KHÔNG ĐẠT"})
    return pd.DataFrame(rows)
