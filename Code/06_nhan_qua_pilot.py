#%% ======================================================================
# BÀI NHÓM PTDLNC – Đề D: Capital Bikeshare
# BƯỚC 6: Chương nhân quả trung thực – hợp đồng estimand và thiết kế pilot ngẫu nhiên
#
# Mục tiêu: dữ liệu KHÔNG ghi chuyến xe tải nào ghé trạm nào (không có nhóm can thiệp), nên không ước lượng được
# hiệu ứng nhân quả của việc cân bằng xe từ dữ liệu này. Thay vì ngụy tạo, ta (1) viết hợp đồng estimand,
# (2) thiết kế pilot ngẫu nhiên và (3) tính cần bao nhiêu ngày pilot để phát hiện được một hiệu ứng cho trước,
# dựa trên xác suất dự báo thật của bước 4 (policy_val). Hiệu ứng giả định chỉ là kịch bản, không phải kết quả.
# ========================================================================
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

t_start = time.perf_counter()
SEEDV = SEED
N_SIM = 2000
CAND = 2 * K_PER_SHIFT                       # ứng viên = 2K trạm xếp hạng cao nhất; ngẫu nhiên K trạm được ghé
LABELS = {"Y_short": "thiếu xe", "Y_full": "đầy xe"}
EFFECTS = [0.2, 0.3, 0.5]                    # giảm tương đối xác suất sự kiện (KỊCH BẢN giả định)
DAYS_GRID = [20, 40, 60, 90, 120, 180]
plt.rcParams.update({"savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

pr = pd.read_parquet(DATA / "pred_dev.parquet")
po = pr[pr.khoi == "policy_val"].copy()


#%% ======================================================================
# 1. Hợp đồng estimand (bảng cho báo cáo)
# ========================================================================
contract = pd.DataFrame([
    ("Đơn vị phân tích", "Một (trạm, ngày thường, ca) nằm trong tập 2K = 16 ứng viên rủi ro cao nhất của ca đó"),
    ("Can thiệp", "Xe tải ghé trạm trong ca để cân bằng xe (nhận xe nếu thiếu, lấy xe nếu đầy)"),
    ("Đối chứng", "Không ghé trạm trong ca đó (vẫn theo dõi như bình thường)"),
    ("Cách gán", "Ngẫu nhiên hoá trong từng (ngày, ca): K trong 2K ứng viên được ghé, K còn lại là đối chứng; hạt giống ghi lại"),
    ("Estimand", "ATE trên nhóm ứng viên: chênh lệch xác suất xảy ra sự kiện hết/đầy xe giữa được ghé và không được ghé"),
    ("Kết cục đo đúng", "Số phút trạm hết xe (hoặc đầy chỗ) từ nguồn tình trạng trạm theo thời gian thực (cần xác minh có thu thập được)"),
    ("Vì sao KHÔNG dùng Y hiện tại làm kết cục pilot", "Y dựa trên chuyến đã diễn ra (nhu cầu bị cắt cụt): trạm được thêm xe sẽ có thêm chuyến, "
                                                      "làm net_out tăng chứ không giảm, nên Y bị méo theo chiều ngược"),
    ("Giả định cần có", "Không nhiễu lẫn nhờ ngẫu nhiên hoá; không tràn tác động (xe lấy từ trạm này làm trạm khác thiếu) -> "
                       "theo dõi cả trạm cho xe; xe tải thực sự đến đúng trạm; đo kết cục độc lập với việc gán"),
    ("Rủi ro đạo đức/vận hành", "Không dữ liệu cá nhân của người thuê; đối chứng bị bỏ qua chủ yếu là trạm biên (hạng 9–16), không phải trạm rủi ro cao nhất; "
                              "dừng pilot nếu chỉ số dịch vụ toàn hệ thống xấu đi quá ngưỡng"),
    ("Sau pilot", "Hiệu ứng ước lượng thay cho giả định b (lợi ích mỗi lần ghé) ở bước tối ưu; xem dị biệt theo nhóm trạm; "
                 "giữ nhóm đối chứng cố định 8–10% khi vận hành thật"),
], columns=["Mục", "Nội dung"])
contract.to_csv(RESULTS / "estimand_contract.csv", index=False, encoding="utf-8-sig")
print("[1/4] Hợp đồng estimand xong"); print(contract.to_string(index=False, max_colwidth=90))


#%% ======================================================================
# 2. Tỷ lệ nền của nhóm ứng viên và chi phí cơ hội của việc ngẫu nhiên hoá (policy_val)
# ========================================================================
def groups(y):
    """Trả về ma trận (số ô ngày-ca) x CAND: giá trị Y của 2K ứng viên cao nhất, xếp theo xác suất cuối cùng."""
    t = po[["ngay", "ca", y, f"p_{y}_final"]].copy()
    t["r"] = t.groupby(["ngay", "ca"])[f"p_{y}_final"].rank(ascending=False, method="first")
    t = t[t.r <= CAND].sort_values(["ngay", "ca", "r"])
    return t.groupby(["ngay", "ca"])[y].apply(lambda s: s.values).pipe(lambda s: np.vstack(s.values))

base_rows, mats = [], {}
for y in LABELS:
    M = groups(y); mats[y] = M
    top_k, top_2k = M[:, :K_PER_SHIFT].mean(), M.mean()
    base_rows.append({"label": y, "so_o_ngay_ca": len(M), "ty_le_su_kien_toan_bo": po[y].mean(),
                      "ty_le_top_K": top_k, "ty_le_top_2K": top_2k,
                      "chi_phi_co_hoi_su_kien_moi_ca": (top_k - top_2k) * K_PER_SHIFT})
base = pd.DataFrame(base_rows).round(4)
base.to_csv(RESULTS / "pilot_ty_le_nen.csv", index=False, encoding="utf-8-sig")
print("\n[2/4] Tỷ lệ nền và chi phí cơ hội của ngẫu nhiên hoá (policy_val):"); print(base.to_string(index=False))


#%% ======================================================================
# 3. Mô phỏng sức mạnh kiểm định: cần bao nhiêu ngày pilot?
# Mỗi lần mô phỏng: lấy mẫu có hoàn lại các ô (ngày, ca) thật của policy_val -> ngẫu nhiên gán K/2K ứng viên được ghé ->
# với ứng viên được ghé, xoá sự kiện với xác suất = hiệu ứng tương đối -> tính chênh lệch từng ô (cụm theo ô),
# kiểm định t một mẫu trên các ô (không coi từng trạm là độc lập).
# ========================================================================
rng = np.random.default_rng(SEEDV)
from scipy import stats

def power(M, n_days, eff):
    n_cells = n_days * 2
    rej = 0
    for _ in range(N_SIM):
        ii = rng.integers(0, len(M), n_cells)
        Y = M[ii].copy()
        order = np.argsort(rng.random(Y.shape), axis=1)
        treat = np.zeros_like(Y, dtype=bool)
        np.put_along_axis(treat, order[:, :K_PER_SHIFT], True, axis=1)
        flip = (rng.random(Y.shape) < eff) & (Y == 1) & treat
        Y[flip] = 0
        d = (Y * treat).sum(1) / K_PER_SHIFT - (Y * ~treat).sum(1) / K_PER_SHIFT
        t, p = stats.ttest_1samp(d, 0.0)
        rej += (p < 0.05) and (d.mean() < 0)
    return rej / N_SIM

rows = []
for y in LABELS:
    for eff in EFFECTS:
        for nd in DAYS_GRID:
            rows.append({"label": y, "hieu_ung_tuong_doi": eff, "so_ngay": nd, "suc_manh": power(mats[y], nd, eff)})
pw = pd.DataFrame(rows).round(3)
pw.to_csv(RESULTS / "pilot_suc_manh.csv", index=False, encoding="utf-8-sig")
need = (pw[pw.suc_manh >= 0.8].groupby(["label", "hieu_ung_tuong_doi"]).so_ngay.min().unstack().reindex(columns=EFFECTS))
need.to_csv(RESULTS / "pilot_so_ngay_can.csv", encoding="utf-8-sig")
print("\n[3/4] Số ngày làm việc tối thiểu để đạt sức mạnh 80% (NaN = chưa đạt trong 180 ngày):"); print(need.to_string())

# tỷ lệ I (sai dương): hiệu ứng = 0 -> phải xấp xỉ 5% một phía
fp = {y: power(mats[y], 60, 0.0) for y in LABELS}
print("      Kiểm tra: tỷ lệ bác bỏ khi KHÔNG có hiệu ứng (60 ngày, kỳ vọng ~0,025 vì một phía):", {k: round(v, 3) for k, v in fp.items()})

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for j, y in enumerate(LABELS):
    for eff, c in zip(EFFECTS, ["#8A8A8A", "#2E8B57", "#1F3A5F"]):
        t = pw[(pw.label == y) & (pw.hieu_ung_tuong_doi == eff)]
        ax[j].plot(t.so_ngay, t.suc_manh, marker="o", color=c, label=f"giảm {int(eff*100)}% sự kiện")
    ax[j].axhline(0.8, color="k", ls=":", lw=1)
    ax[j].set_title(f"Hình 9{'ab'[j]}. Sức mạnh pilot theo số ngày – {LABELS[y]}")
    ax[j].set_xlabel("Số ngày làm việc pilot"); ax[j].set_ylabel("Sức mạnh kiểm định"); ax[j].legend(fontsize=8)
plt.tight_layout(); plt.savefig(RESULTS / "nhanqua_hinh9_suc_manh_pilot.png"); plt.close()
print(f"[4/4] [✓] BƯỚC 6 (nhân quả) hoàn tất sau {time.perf_counter() - t_start:.1f}s")
