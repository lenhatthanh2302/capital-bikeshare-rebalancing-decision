# Điều phối xe đạp chia sẻ Capital Bikeshare: từ dự báo đến quyết định có kiểm soát

Bài nhóm A1.1 – môn **Phân tích dữ liệu nâng cao** (lớp 261MIE401601, Cao học UEL), **Đề D: Capital Bikeshare**. Giảng viên: TS. Nguyễn Thôn Dã. Nhóm 2.

Kho này chứa **mã nguồn, kết quả và bằng chứng đóng băng** của bài. Báo cáo Word và slide nộp riêng. Dữ liệu gốc (3,2 GB) và các bảng trung gian lớn không đưa lên đây, cách lấy lại nằm ở mục 4.

## 1. Bài toán

Mỗi ca (sáng 06:00–10:00, chiều 16:00–20:00) của ngày thường, nhà vận hành chỉ ghé được **K = 8 trong 80 trạm** để điều phối xe. Quyết định được đưa ra lúc **22:00 ngày hôm trước**. Bài trả lời hai câu hỏi: (1) dữ liệu chuyến đi công khai giúp chọn trạm tốt hơn các quy tắc đơn giản đến mức nào; (2) kết luận nào còn đứng vững trên giai đoạn chưa từng dùng để xây dựng mô hình.

Dữ liệu: 33 tệp chuyến đi tháng 01/2024–09/2026, 17.605.262 chuyến, còn 17.208.188 chuyến sau làm sạch. Nhãn "thiếu xe/đầy xe" được **xấp xỉ** từ dòng xe ròng của trạm vì dữ liệu không ghi số xe tại trạm.

## 2. Quy trình 10 bước và tệp mã tương ứng

| Bước | Nội dung | Tệp trong `Code/` |
|---|---|---|
| 1 | Gộp 33 tệp CSV, làm sạch, đối chiếu nhất quán | `01_gop_du_lieu.py` |
| 2 | EDA và chọn 80 trạm (chỉ dùng giai đoạn huấn luyện) | `02_eda_chon_tram.py` |
| 3 | Lưới (trạm, ngày, ca), nhãn, đặc trưng tại t0 chống rò rỉ | `03_dac_trung.py` |
| 4 | Dự báo xác suất (GBM, Logistic, baseline) và hiệu chỉnh | `04_du_bao_hieu_chinh.py` |
| 5 | Giải thích bằng PFI và SHAP ở mức nhóm đặc trưng | `05_pfi_shap.py` |
| 6 | Nhân quả: hợp đồng estimand và thiết kế pilot ngẫu nhiên | `06_nhan_qua_pilot.py` |
| 7 | Tối ưu hóa chọn trạm: SPO+, Greedy, MILP (HiGHS), độ nhạy, ROI | `07_toi_uu.py` |
| 8 | Action Router: ba lớp bộ lọc an toàn, nhóm CONTROL 10% | `08_router.py` |
| 9 | Đóng băng (SHA-256) và công bố tiêu chí chấp nhận | `09_dong_bang.py` |
| 10 | Kiểm tra locked (mở một lần) và OOT, giám sát | `10_locked_oot.py` |
| Bổ sung | Phân tích sau khi có kết quả (post-hoc) | `11_phan_tich_bo_sung.py` |
| Bổ sung | Vẽ sơ đồ quy trình | `12_ve_so_do_quy_trinh.py` |
| Bổ sung | Thử cải thiện đặc trưng và thời tiết (chỉ trên validation) | `audit_thu_cai_thien.py` |
| Bản gộp | Toàn bộ quy trình trong một tệp, tiện chạy nhanh | `pipeline.py` |

**Tệp chính thức và bản gộp.** Các tệp bước `01`–`10` cùng 5 tệp thư viện là bản chính thức, được băm trong `release_manifest.csv`. `pipeline.py` được gộp từ các tệp này sau thời điểm đóng băng, **không nằm trong manifest**; dùng nó để chạy nhanh, còn muốn đối chiếu bản đóng băng thì dùng các tệp bước.

Các tệp dùng chung: `config.py` (tham số, đường dẫn), `features_lib.py`, `solver_lib.py`, `router_lib.py`, `eval_lib.py`. Mỗi tệp bước mở đầu bằng khối chú thích "Mục tiêu".

**Chia thời gian theo ngày D:** huấn luyện 01/2024–10/2025; kiểm định 11–12/2025; hiệu chỉnh 01–02/2026; kiểm định chính sách 03–04/2026; **locked** 05–06/2026; **OOT** 07–09/2026.

## 3. Cấu trúc thư mục

```
Code/                  mã nguồn các bước và thư viện dùng chung
Results/               bảng CSV và hình PNG của từng bước
  release_manifest.*   bản kê SHA-256 của bản đóng băng (80 dòng) và tệp .sha256
  locked_access.lock   dấu thời điểm mở locked (23/10/2026)
  bo_sung/             phân tích post-hoc và sơ đồ quy trình
Data/models/           mô hình đã đóng băng (forecast_bundle, decision_bundle)
Data/weather_hourly.parquet   thời tiết Open-Meteo, chỉ dùng cho audit_thu_cai_thien.py
```

Không có trong kho: `Datasets/` (33 tệp CSV nguồn, 3,2 GB), `Data/trips_clean.parquet` (460 MB) và các bảng trung gian khác. Mã đọc/ghi theo cấu trúc `Datasets/`, `Data/`, `Results/` cùng cấp với `Code/`.

## 4. Cách chạy lại

1. Cài Python 3.14 và thư viện trong `requirements.txt`.
2. Tải 33 tệp `YYYYMM-capitalbikeshare-tripdata.csv` (202401–202609) từ trang dữ liệu công khai của Capital Bikeshare (https://capitalbikeshare.com/system-data) vào thư mục `Datasets/`.
3. Trong thư mục `Code/`, chạy lần lượt `py -3 01_gop_du_lieu.py` đến `py -3 08_router.py` để dựng lại dữ liệu trung gian, mô hình và kết quả giai đoạn phát triển.
4. **Không chạy lại bước 9 và 10 ở chế độ `real`.** Tệp `locked_access.lock` ghi nhận lần mở locked chính thức (23/10/2026, 14:31); bước 9 sẽ từ chối chạy khi tệp này tồn tại. Muốn xem lại cách chạy, dùng `py -3 10_locked_oot.py dry` (diễn tập trên giai đoạn kiểm định chính sách, không đụng locked).

Lưu ý minh bạch: sau lần mở chính thức, một bản sao của `pipeline.py` (chạy ở thư mục khác, không có tệp khóa) đã chạy chế độ `real` thêm một lần để kiểm tra tính tái lập; kết quả giống hệt lần chính thức và không dùng để chỉnh mô hình hay ngưỡng. Vì vậy locked không còn là "mở đúng một lần" theo nghĩa chặt chẽ, và nhóm khai điều này trong báo cáo.

Kết quả chính thức của bài là các tệp trong `Results/` đã sinh ra ở lần chạy đóng băng. Chạy lại sẽ ghi đè các tệp này, nên nếu chỉ cần đối chiếu, hãy chạy trên bản sao của thư mục.

## 5. Quy tắc đóng băng và kiểm chứng

- Trước khi mở locked, bước 9 công bố **25 tiêu chí chấp nhận** (`Results/acceptance_criteria.csv`) và băm SHA-256 mô hình, mã, tham số, dữ liệu trung gian và 33 tệp nguồn vào `release_manifest.csv`.
- Bước 10 kiểm tra lại toàn bộ bản băm trước khi chạy. Locked được mở chính thức một lần (xem lưu ý ở mục 4); OOT là cửa sổ căng thẳng sau locked.
- Kiểm tra bản băm của các tệp trong kho: so SHA-256 của `Code/*.py` (trừ `pipeline.py`), `Data/models/*` và các tệp tham số trong `Results/` với cột `sha256` của `release_manifest.csv`.

## 6. Kết quả chính

| Khối | Tiêu chí đạt | Lift@8 (thiếu / đầy) | GBM+MILP (sự kiện/ca) | Mức lý tưởng (oracle) |
|---|---|---|---|---|
| Locked (43 ngày, 86 ca) | 19/19 | 2,78 / 3,41 (lag5: 1,97 / 1,74) | 2,94 (38,9% oracle) | 7,57 |
| OOT (66 ngày, 132 ca) | 5/6 | 3,00 / 3,43 | 3,75 (48,4% oracle) | – |

Trên locked, GBM+MILP hơn chọn ngẫu nhiên +1,33 [1,04; 1,65] và hơn quy tắc "cùng ca tuần trước" +0,70 [0,45; 0,96]; MILP hơn Greedy +0,17 [0,013; 0,33] sự kiện mỗi ca (khoảng tin cậy 95%, bootstrap theo ngày). Router đạt 18/18 kiểm tra, nhóm CONTROL 11,0%. Trên OOT, tiêu chí O05 (tổn thất do bộ lọc an toàn ≤ 0,3) **không đạt** (0,4455); PSI trung bình tăng từ 0,194 (locked) lên 0,274 (OOT).

## 7. Hạn chế cần biết

- Nhãn thiếu/đầy xe là xấp xỉ từ dòng xe ròng, không phải số xe thực tế; khoảng 18% chuyến thiếu mã trạm xuất phát (toàn bộ là xe điện) không nằm trong phân tích.
- Chưa có thử nghiệm can thiệp, nên **không có kết luận nhân quả** về hiệu quả điều phối; bài chỉ thiết kế pilot ngẫu nhiên và tính số ngày cần thiết.
- Phân tích bổ sung (`Results/bo_sung/`) cho thấy GBM hơn "cùng ca tuần trước" nhưng gần như hòa với quy tắc "tần suất sự kiện 20 ngày gần nhất"; vì vậy kết luận đúng là mô hình hơn quy tắc hiện hành, không phải hơn mọi quy tắc đơn giản.
- Đặc trưng `city_*` tính tổng toàn thành phố có dùng 22:00–23:59 của D−1 (lệch nhẹ so với t0); ảnh hưởng được đo post-hoc và rất nhỏ. Mô hình đóng băng không bị sửa.
- ROI dựa trên giả định chi phí, không phải số liệu vận hành thật.

## 8. Thành viên Nhóm 2

Lê Nhật Thanh, Lê Ngọc Phú, Huỳnh Trúc Ngân, Phạm Thị Thúy Kiều, Trịnh Quang Tân.
