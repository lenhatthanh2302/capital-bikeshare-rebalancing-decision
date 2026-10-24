"""Cấu hình dùng chung cho mọi bước của bài nhóm (đổi ở đây, không sửa rải rác trong từng file)."""
from pathlib import Path
import pandas as pd

SEED = 42
BASE = Path(__file__).resolve().parent.parent
DATA, RESULTS = BASE / "Data", BASE / "Results"

N_STATIONS = 80
K_PER_SHIFT = 8                       # số trạm xe tải ghé được mỗi ca (giả định, có sensitivity ở Bước 6)
SHIFTS = {"sang": (6, 10), "chieu": (16, 20)}   # [giờ bắt đầu, giờ kết thúc)
TAU_QUANTILE = 0.90                   # ngưỡng sự kiện = phân vị 90 theo (trạm, ca), học trên train

# 5 vai trò dữ liệu + out-of-time (ngày D của đơn vị phân tích, gồm cả hai đầu)
BLOCKS = {
    "train":       ("2024-01-01", "2025-10-31"),
    "validation":  ("2025-11-01", "2025-12-31"),
    "calibration": ("2026-01-01", "2026-02-28"),
    "policy_val":  ("2026-03-01", "2026-04-30"),
    "locked":      ("2026-05-01", "2026-06-30"),
    "oot":         ("2026-07-01", "2026-09-30"),
}
DEV_END = BLOCKS["policy_val"][1]     # Bước 3–5 chỉ dựng dữ liệu đến hết policy_val; locked/oot mở ở bước đóng băng

# Ngày lễ liên bang Mỹ rơi vào ngày thường (lịch công bố trước nên biết được tại t0)
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
])


def block_of(d: pd.Series) -> pd.Series:
    out = pd.Series("khac", index=d.index, dtype=object)
    for k, (a, b) in BLOCKS.items():
        out[(d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))] = k
    return out
