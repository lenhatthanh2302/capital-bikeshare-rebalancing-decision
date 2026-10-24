#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
# BƯỚC 9: Đóng băng bản phát hành và CÔNG BỐ tiêu chí chấp nhận TRƯỚC khi mở locked
#
# Mục tiêu: (1) khẳng định locked/OOT chưa bị mở; (2) công bố tiêu chí chấp nhận (ngưỡng đặt từ policy_val và lập luận,
# không từ locked); (3) băm SHA-256 mô hình, mã, tham số, dữ liệu trung gian và 33 file nguồn vào release manifest;
# (4) ghi kế hoạch triển khai và điều kiện huấn luyện lại. Sau bước này KHÔNG sửa bất kỳ file nào trong manifest:
# 10_locked_oot.py (chế độ real) sẽ từ chối chạy nếu có băm nào lệch.
# ========================================================================
import sys, time, hashlib, platform, warnings
import numpy as np
import pandas as pd
import joblib
import sklearn, scipy
import eval_lib as EV
import router_lib as RL
from config import *

warnings.filterwarnings("ignore")
t_start = time.perf_counter()
CODE = Path(__file__).resolve().parent


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()


#%% ======================================================================
# 1. Locked/OOT chưa bị mở
# ========================================================================
dev = pd.read_parquet(DATA / "features_dev.parquet")
pred_dev = pd.read_parquet(DATA / "pred_dev.parquet")
checks = [("Chưa có khoá locked", not (RESULTS / "locked_access.lock").exists()),
          ("features_dev.parquet kết thúc ≤ hết policy_val", dev.ngay.max() <= pd.Timestamp(DEV_END)),
          ("pred_dev.parquet chỉ có validation/calibration/policy_val", set(pred_dev.khoi) <= {"validation", "calibration", "policy_val"}),
          ("Không có kết quả locked/oot nào trong Results", not any(RESULTS.glob("oot_*")) and not any(p for p in RESULTS.glob("locked_*") if not p.name.startswith("dry_")))]
for n_, ok in checks: print(f"  [{'Đạt' if ok else 'KHÔNG'}] {n_}")
assert all(ok for _, ok in checks), "Locked/OOT có dấu hiệu đã bị mở; dừng đóng băng"
print("[1] Xác nhận locked/OOT chưa mở")


#%% ======================================================================
# 2. Tham chiếu policy_val bằng đúng hàm đánh giá sẽ dùng cho locked, rồi công bố tiêu chí
# ========================================================================
bundle = joblib.load(DATA / "models" / "forecast_bundle.joblib")
pred = EV.predict_frozen(dev[dev.khoi != "train"], bundle)
d_all = RL.prep_cells(pred, dev)
psi_df = RL.psi_daily(d_all); cperf = RL.cell_perf(d_all)
ref = EV.full_block_metrics(d_all, psi_df, cperf, "policy_val")["m"]
ref.update({"replay_max_diff": 0.0, "leakage_max_diff": 0.0, "router_replay_diff": 0})
NOW = pd.Timestamp.now().isoformat(timespec="seconds")

C = [  # (mã, khối, nội dung, chỉ số, toán tử, ngưỡng, lý do chọn ngưỡng)
 ("L01", "locked", "Lift@8 của GBM, nhãn thiếu xe", "lift_short", ">=", "1.5", "Bằng 60% mức đo trên policy_val (2,5); nhỏ hơn 1,5 thì xếp hạng gần như không hơn ngẫu nhiên"),
 ("L02", "locked", "Lift@8 của GBM, nhãn đầy xe", "lift_full", ">=", "1.5", "Như L01 (policy_val 3,4)"),
 ("L03", "locked", "PR-AUC GBM hơn mốc 'cùng ca tuần trước', thiếu xe", "prauc_hon_lag5_short", ">", "0", "Mô hình phải hơn quy tắc đơn giản đang có sẵn"),
 ("L04", "locked", "PR-AUC GBM hơn mốc 'cùng ca tuần trước', đầy xe", "prauc_hon_lag5_full", ">", "0", "Như L03"),
 ("L05", "locked", "Tỷ số xác suất dự báo TB / tỷ lệ thực, thiếu xe", "ti_so_short", "trong", "0.7;1.3", "Sai lệch ±30% là biên chịu được vì tỷ lệ sự kiện đổi theo mùa 7,7%–11,9%"),
 ("L06", "locked", "Tỷ số xác suất dự báo TB / tỷ lệ thực, đầy xe", "ti_so_full", "trong", "0.7;1.3", "Như L05"),
 ("L07", "locked", "ECE xác suất thiếu xe", "ece_short", "<=", "0.03", "Gấp rưỡi mức policy_val (0,020)"),
 ("L08", "locked", "ECE xác suất đầy xe", "ece_full", "<=", "0.03", "Gấp đôi mức policy_val (0,015)"),
 ("L09", "locked", "Giá trị thực GBM+MILP hơn ngẫu nhiên+MILP: cận dưới CI95 theo ngày", "gbm_tru_ngau_nhien_CI_duoi", ">", "0", "Phải hơn ngẫu nhiên một cách có ý nghĩa, không chỉ về điểm"),
 ("L10", "locked", "Giá trị thực GBM+MILP hơn 'cùng ca tuần trước+MILP' (trung bình)", "gbm_tru_lag5_TB", ">", "0", "Mô hình phải đáng giá hơn quy tắc dự phòng"),
 ("L11", "locked", "Tỷ lệ giá trị so với oracle", "ti_le_so_oracle", ">=", "0.25", "Policy_val 0,36, ngẫu nhiên 0,21: cần ở trên mức ngẫu nhiên rõ rệt"),
 ("L12", "locked", "MILP không kém Greedy về giá trị dự báo ở mọi ô", "milp_ge_greedy_du_bao", ">=", "1", "Tính chất toán học của nghiệm tối ưu; vi phạm = lỗi bộ giải"),
 ("L13", "locked", "Số kiểm tra tự động của router không đạt", "router_tests_fail", "==", "0", "Ràng buộc cứng và idempotency không có ngoại lệ"),
 ("L14", "locked", "Tỷ lệ ô phải dùng chính sách dự phòng", "fallback_share", "<=", "0.2", "Policy_val 2,3%; trên 20% nghĩa là mô hình không còn dùng được"),
 ("L15", "locked", "Giá trị mất do 3 tầng guardrail (sự kiện/ô)", "guardrail_mat_gia_tri", "<=", "0.3", "Policy_val 0,08; guardrail không được ăn quá ~10% giá trị"),
 ("L16", "locked", "Độ trễ router p95 mỗi ô (ms)", "latency_p95_ms", "<=", "500", "Thời hạn từ t0 22:00 đến đầu ca 06:00 là 8 giờ; 0,5 giây là rất rộng"),
 ("L17", "locked", "Replay: dựng lại dev trùng bản đã lưu (đặc trưng, xác suất)", "replay_max_diff", "<=", "0", "Quy trình phải tái lập được"),
 ("L18", "locked", "Thí nghiệm rò rỉ trên ngày locked/OOT", "leakage_max_diff", "<=", "0", "Đặc trưng ngày D không được đổi khi ghi đè tương lai"),
 ("L19", "locked", "Router replay trên policy_val trùng kết quả bước 8 (số khác biệt)", "router_replay_diff", "<=", "0", "Bản router dùng ở bước 10 phải đúng là bản đã duyệt"),
 ("O01", "oot", "Lift@8 của GBM, thiếu xe (OOT, stress)", "lift_short", ">=", "1.2", "Nới hơn locked: OOT xa hơn, chỉ cần còn hơn ngẫu nhiên rõ"),
 ("O02", "oot", "Lift@8 của GBM, đầy xe (OOT, stress)", "lift_full", ">=", "1.2", "Như O01"),
 ("O03", "oot", "Giá trị thực GBM+MILP hơn ngẫu nhiên+MILP (trung bình)", "gbm_tru_ngau_nhien_TB", ">", "0", "Tối thiểu không tệ hơn ngẫu nhiên"),
 ("O04", "oot", "Số kiểm tra tự động của router không đạt", "router_tests_fail", "==", "0", "Hệ thống phải còn an toàn dù dữ liệu trôi"),
 ("O05", "oot", "Giá trị mất do 3 tầng guardrail (sự kiện/ô)", "guardrail_mat_gia_tri", "<=", "0.3", "Như L15"),
 ("O06", "oot", "Độ trễ router p95 mỗi ô (ms)", "latency_p95_ms", "<=", "500", "Như L16"),
]
crit = pd.DataFrame(C, columns=["ma", "khoi", "noi_dung", "chi_so", "toan_tu", "nguong", "ly_do_chon_nguong"])
crit["gia_tri_policy_val"] = [round(float(ref[k]), 4) for k in crit.chi_so]
R = [("Q1", "quy_tac", "Locked chỉ mở MỘT lần; kết quả được công bố nguyên trạng, không chỉnh mô hình/ngưỡng τ/tham số rồi chạy lại", "", "", "", "Chống chọn mô hình theo locked"),
     ("Q2", "quy_tac", "Đạt toàn bộ L01–L19 -> được phép chạy thử có kiểm soát (canary)", "", "", "", ""),
     ("Q3", "quy_tac", "Không đạt bất kỳ L nào -> chỉ chạy bóng (shadow): ghi dự báo và lệnh nhưng KHÔNG giao xe tải; điều tra nguyên nhân trước khi chạy lại ở một khoá mới", "", "", "", ""),
     ("Q4", "quy_tac", "OOT là cửa sổ stress: không đạt O01–O06 không bác bỏ bản phát hành nhưng bắt buộc xem xét huấn luyện lại; không dùng OOT để chỉnh mô hình", "", "", "", ""),
     ("Q5", "quy_tac", "Ngưỡng được đặt SAU khi đã xem policy_val (không phải locked) và dựa trên lập luận vận hành; điều này được thú nhận trong báo cáo", "", "", "", "")]
crit = pd.concat([crit, pd.DataFrame(R, columns=["ma", "khoi", "noi_dung", "chi_so", "toan_tu", "nguong", "ly_do_chon_nguong"])], ignore_index=True)
crit["cong_bo_luc"] = NOW
crit.to_csv(RESULTS / "acceptance_criteria.csv", index=False, encoding="utf-8-sig")
print(f"[2] Đã công bố {int(crit.khoi.isin(['locked', 'oot']).sum())} tiêu chí lúc {NOW}"); print(crit[crit.khoi.isin(["locked", "oot"])][["ma", "chi_so", "toan_tu", "nguong", "gia_tri_policy_val"]].to_string(index=False))


#%% ======================================================================
# 3. Kế hoạch triển khai và điều kiện huấn luyện lại
# ========================================================================
plan = pd.DataFrame([
 ("1. Chạy bóng (shadow)", "Qua locked; ≥ 4 tuần", "Router sinh lệnh mỗi ngày nhưng không giao xe tải; so lệnh với quyết định điều phối viên", "Còn đạt L01–L11 trên cửa sổ trượt 4 tuần", "Quay lại huấn luyện lại nếu không đạt"),
 ("2. Canary", "Hết shadow đạt", "Một khu (13–15 trạm) chạy thật, CONTROL 10% giữ cố định, HITL bật; các khu còn lại giữ như cũ", "Không vi phạm quy tắc cứng; A/A và cân bằng CONTROL đạt; điều phối viên chấp nhận ≥ 80% lệnh", "Tắt router, quay về quy tắc 'cùng ca tuần trước + Greedy'"),
 ("3. Mở rộng", "Hết canary đạt", "Mở dần cả 6 khu; bắt đầu pilot ngẫu nhiên (K trong 2K) để đo hiệu ứng thật thay cho giả định b", "Pilot đạt đủ ngày theo bảng sức mạnh (20–90 ngày)", "Rút về canary"),
 ("Điều kiện huấn luyện lại A", "Giám sát hằng ngày", "PSI điểm dự báo ≥ 0,25 liên tiếp ≥ 5 ngày làm việc VÀ mô hình kém quy tắc dự phòng trên 20 ô gần nhất", "", "Huấn luyện lại bằng dữ liệu mới, kiểm lại bằng tập khoá mới"),
 ("Điều kiện huấn luyện lại B", "Giám sát hằng ngày", "Mô hình kém quy tắc dự phòng liên tiếp ≥ 10 ngày làm việc", "", "Như A"),
 ("Điều kiện huấn luyện lại C", "Lịch", "Quá 3 tháng kể từ lần huấn luyện gần nhất", "", "Huấn luyện lại định kỳ, bắt buộc đổi mùa"),
 ("Ngắt khẩn cấp", "Bất kỳ lúc nào", "Vi phạm quy tắc cứng hoặc kiểm tra tự động không đạt", "", "Tự chuyển quy tắc dự phòng và báo người phụ trách"),
], columns=["giai_doan", "thoi_diem", "noi_dung", "dieu_kien_dat", "neu_khong_dat"])
plan.to_csv(RESULTS / "release_ke_hoach_trien_khai.csv", index=False, encoding="utf-8-sig")
print("[3] Kế hoạch triển khai và điều kiện huấn luyện lại đã ghi")


#%% ======================================================================
# 4. Release manifest: SHA-256 mã, mô hình, tham số, dữ liệu trung gian, 33 file nguồn
# ========================================================================
rows = []
def add(loai, p):
    p = Path(p); rows.append({"loai": loai, "duong_dan": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
for p in sorted(CODE.glob("*.py")): add("ma", p)
for p in [DATA / "models" / "forecast_bundle.joblib", DATA / "models" / "decision_bundle.joblib"]: add("mo_hinh", p)
for n_ in ["tau_thresholds.csv", "stations_selected.csv", "feature_list.csv", "acceptance_criteria.csv", "release_ke_hoach_trien_khai.csv"]: add("tham_so_tieu_chi", RESULTS / n_)
for n_ in ["features_dev.parquet", "pred_dev.parquet", "station_hour.parquet"]: add("du_lieu_trung_gian", DATA / n_)
audit = pd.read_csv(RESULTS / "data_audit.csv")
n_raw_ok = 0
for r in audit.itertuples():
    p = BASE / "Datasets" / r.file_name
    h = sha(p); n_raw_ok += (h == r.sha256)
    rows.append({"loai": "du_lieu_nguon", "duong_dan": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size, "sha256": h})
print(f"[4a] Băm {len(rows)} file; {n_raw_ok}/{len(audit)} file nguồn trùng băm đã ghi ở bước 1 (data_audit.csv)")
assert n_raw_ok == len(audit), "Dữ liệu nguồn đã đổi so với lúc đọc ở bước 1"
meta = {"policy_id": RL.POLICY_ID, "model_version": RL.MODEL_VERSION, "dong_bang_luc": NOW, "seed": SEED, "k_moi_ca": K_PER_SHIFT,
        "ngan_sach_phut": RL.S.B_MIN, "tau_quantile": TAU_QUANTILE, "dev_end": DEV_END, "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
        "bo_hieu_chinh_Y_short": bundle["calibrators"][("Y_short", "GBM")]["chosen"], "bo_hieu_chinh_Y_full": bundle["calibrators"][("Y_full", "GBM")]["chosen"],
        "PSI_canh_bao": RL.PSI_WARN, "PSI_dung": RL.PSI_STOP, "CONTROL_pct": RL.CONTROL_PCT, "V_MIN": RL.V_MIN, "MAX_VISITS_5D": RL.MAX_VISITS_5D}
for k, v in meta.items(): rows.append({"loai": "meta", "duong_dan": k, "bytes": 0, "sha256": str(v)})
man = pd.DataFrame(rows)
man.to_csv(RESULTS / "release_manifest.csv", index=False, encoding="utf-8-sig")
msha = sha(RESULTS / "release_manifest.csv")
(RESULTS / "release_manifest.sha256").write_text(msha + "  release_manifest.csv\n", encoding="utf-8")
print(man.groupby("loai").size().to_dict()); print(f"[4b] SHA-256 manifest: {msha}")
print(f"[✓] BƯỚC 9 (đóng băng) hoàn tất sau {time.perf_counter() - t_start:.0f}s")
