"""Vẽ sơ đồ quy trình nghiên cứu rút gọn (10 bước). Chạy: py -3 12_ve_so_do_quy_trinh.py -> Results/bo_sung/hinh_quy_trinh_nghien_cuu.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from config import RESULTS

OUT = RESULTS / "bo_sung"; OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(16, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

COL = {"data": ("#DCE9F7", "#2F6DB5"), "fc": ("#DDF0DC", "#3C8D40"), "dec": ("#FCE8CC", "#D98310"), "gov": ("#E8DDF3", "#7A4BA6")}
steps = [
    ("data", "Gộp và làm sạch\ndữ liệu"),
    ("data", "Khám phá dữ liệu,\nchọn trạm"),
    ("data", "Hợp đồng quyết định,\nđặc trưng, nhãn"),
    ("fc", "Dự báo và hiệu chỉnh\nxác suất"),
    ("fc", "Giải thích mô hình"),
    ("fc", "Nhân quả\n(thiết kế thử nghiệm)"),
    ("dec", "Tối ưu hóa\nchọn trạm"),
    ("dec", "Action Router\n(bộ lọc an toàn)"),
    ("gov", "Đóng băng và\nkiểm tra locked"),
    ("gov", "OOT và\ngiám sát"),
]
W, H = 16.2, 11.5
xs = [2 + i * 19.2 for i in range(5)]
ys = [23.5, 5.5]
pos = []
for i, (k, title) in enumerate(steps):
    x, y = xs[i % 5], ys[i // 5]
    fc, ec = COL[k]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.2,rounding_size=1.0", fc=fc, ec=ec, lw=2.2))
    ax.add_patch(Circle((x + 2.6, y + H - 2.6), 1.7, fc=ec, ec="white", lw=1.5, zorder=3))
    ax.text(x + 2.6, y + H - 2.6, str(i + 1), color="white", fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(x + W / 2, y + 4.2, title, fontsize=12, fontweight="bold", ha="center", va="center", color="#222", linespacing=1.35)
    pos.append((x, y))


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#555"))


for r in (0, 1):
    for c in range(4):
        x, y = pos[r * 5 + c]; arrow((x + W + 0.35, y + H / 2), (xs[c + 1] - 0.35, y + H / 2))
xa, xb = pos[4][0] + W / 2, pos[5][0] + W / 2
ym = (ys[0] + ys[1] + H) / 2
ax.plot([xa, xa, xb], [ys[0] - 0.3, ym, ym], color="#555", lw=2.4, solid_capstyle="round")
arrow((xb, ym), (xb, ys[1] + H + 0.3))

ax.text(50, 40.3, "Quy trình nghiên cứu", fontsize=18, fontweight="bold", ha="center", va="center")
for i, (k, lab) in enumerate((("data", "Dữ liệu"), ("fc", "Dự báo và giải thích"), ("dec", "Quyết định"), ("gov", "Kiểm soát và vận hành"))):
    fc, ec = COL[k]
    x0 = 24 + i * 16
    ax.add_patch(Rectangle((x0, 36.0), 1.4, 1.5, fc=fc, ec=ec, lw=1.5)); ax.text(x0 + 2.2, 36.75, lab, fontsize=10.5, va="center")
fig.savefig(OUT / "hinh_quy_trinh_nghien_cuu.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved", OUT / "hinh_quy_trinh_nghien_cuu.png")
