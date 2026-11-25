# Báo cáo triển khai dataset YOLO-OBB cho screw

## 1. Mục tiêu

Đã triển khai pipeline gộp hai nguồn dữ liệu WEEE Disassembly Screw và MVTec Screws để huấn luyện YOLO-OBB một lớp `screw`.

YOLO chỉ có nhiệm vụ phát hiện và crop screw. Nó không phân loại loại screw, không kết luận GOOD/DEFECT và không thay thế mô hình anomaly/classification phía sau.

## 2. Dữ liệu đầu vào

### WEEE Disassembly Screw

- Local path: `datasets/external/weee_screw`
- 945 ảnh, giữ nguyên train/validation/test.
- Annotation COCO có polygon segmentation.
- Polygon được chuyển thành rotated rectangle bằng minimum-area rectangle thật.

### MVTec Screws

- Local path: `datasets/external/mvtec_screws/extracted`
- 384 ảnh.
- Annotation gốc có dạng `[row, col, width, height, phi]`.
- Đã chuyển đúng hệ tọa độ MVTec sang 4 góc OBB.
- Chỉ giữ các nhóm screw-like:
  - `type_001` đến `type_006`
  - `type_008`
  - `type_013`
- Đã loại `type_007`, `type_009`, `type_010`, `type_011`, `type_012` vì đây là nut/washer hoặc nhóm không phù hợp với mục tiêu detector screw.

Việc chọn nhóm trên được kiểm tra bằng ảnh overlay. Các nut/washer còn xuất hiện trong ảnh gốc nhưng không bị gán nhãn `screw`, nên không làm phát sinh false-positive label trong ground truth.

## 3. Dataset đã tạo

Dataset gộp chính thức:

`datasets/screw_obb_combined_screw_only`

File cấu hình:

`datasets/screw_obb_combined_screw_only/data.yaml`

Thống kê:

| Split | Số ảnh | Số OBB |
|---|---:|---:|
| Train | 1.034 | 5.482 |
| Val | 145 | 811 |
| Test | 150 | 832 |
| Tổng | 1.329 | 7.125 |

Trong đó có 945 ảnh WEEE và 384 ảnh MVTec. Dataset chỉ có một class:

```yaml
names:
  0: screw
```

## 4. Kiểm tra chất lượng

Validator đã chạy trên toàn bộ dataset:

- Train: 1.034 ảnh, 1.034 label, 5.482 OBB, 0 lỗi.
- Val: 145 ảnh, 145 label, 811 OBB, 0 lỗi.
- Test: 150 ảnh, 150 label, 832 OBB, 0 lỗi.
- Tổng số lỗi format/toạ độ: **0**.

Unit test cho phần chuyển đổi OBB và xử lý annotation: **4 passed**.

Ảnh kiểm tra trực quan đã lưu tại:

`reports/screw_yolo_obb_combined_screw_only/labels/train_mvtec_0040fc5623_screws_265.png`

Kết quả overlay cho thấy các screw được khoanh theo đúng hướng; các nut/washer trong cùng ảnh không bị gán nhãn screw.

## 5. Các thay đổi mã nguồn

- `scripts/screw_yolo_obb.py`
  - Thêm chuyển đổi MVTec OBB `[row, col, width, height, phi]`.
  - Giữ polygon-to-OBB bằng minimum-area rectangle cho WEEE.
  - Không cho phép axis-aligned fallback mặc định.

- `scripts/prepare_screw_yolo_obb.py`
  - Gộp WEEE và MVTec.
  - Chọn nhóm MVTec bằng `--mvtec-types`.
  - Mặc định dùng `1,2,3,4,5,6,8,13`.
  - Tránh đọc trùng JSON tổng và JSON train/val/test.
  - Sinh `data.yaml` dùng đường dẫn tương đối, thuận tiện chạy trên Windows.

- `scripts/validate_screw_yolo_obb.py`
  - Kiểm tra ảnh, label, số điểm OBB, khoảng tọa độ và có thể render overlay.

- `docs/Screw_YOLO.md`
  - Bổ sung hướng dẫn chuyển đổi, validate, train và evaluate.

## 6. Lệnh train trên máy Windows có GPU

```bash
python scripts/train_screw_yolo_obb.py \
  --data datasets/screw_obb_combined_screw_only/data.yaml \
  --model yolo11n-obb.pt \
  --epochs 100 \
  --imgsz 640 \
  --device 0
```

Vì screw nhỏ, nên chạy thêm một lần với `--imgsz 1024` và chọn checkpoint theo recall/precision trên validation. Metric quan trọng nhất ở bước detector là recall/missed-screw rate: bỏ sót screw sẽ làm downstream anomaly không có crop để xử lý.

## 7. Lưu ý dữ liệu

- JSON WEEE local có 4.398 annotation, trong khi con số công bố của dataset là 4.414; cần ghi nhận đây là sai khác giữa bản tải và thống kê công bố.
- JSON MVTec local có 4.427 annotation, trong khi trang dataset ghi 4.426; đây là sai khác một annotation trong bản local.
- MVTec có nhiều vật thể trong cùng ảnh. Các nut/washer vẫn được giữ trong ảnh nhưng không được gán nhãn, phù hợp với mục tiêu chỉ detect screw; khi train cần theo dõi false positive trên các vật thể này.
- Đã chạy smoke test YOLO-OBB 5 epoch trên Apple M4 bằng MPS ở `imgsz=640`, batch 4. Đây là kết quả kiểm tra pipeline ban đầu, chưa phải checkpoint production.

## 8. Trạng thái và bước tiếp theo

Smoke test 5 epoch hoàn tất trong khoảng 6,6 phút. Kết quả validation tốt nhất/cuối cùng:

| Tập | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Val | 0,846 | 0,788 | 0,852 | 0,618 |
| Test | 0,905 | 0,612 | 0,751 | 0,529 |

Checkpoint tốt nhất:

`runs/obb/runs/screw_yolo_obb/m4_smoke5_final/weights/best.pt`

Bảng metric theo epoch:

`runs/obb/runs/screw_yolo_obb/m4_smoke5_final/results.csv`

Trạng thái tại thời điểm smoke test: **pipeline train/evaluate đã chạy thành công trên Mac M4**. Sau đó phát hiện một phần ảnh WEEE của dataset cũ bị trỏ nhầm sang mask phân đoạn, nên các metric cũ chỉ giữ lại để tham khảo lịch sử.

### Fine-tune thêm 10 epoch từ `best.pt`

Đã fine-tune thêm 10 epoch từ checkpoint 5 epoch bằng MPS trên Apple M4, không push lên GitHub. Kết quả validation ở epoch 10:

| Tập | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Val | 0,901 | 0,840 | 0,909 | 0,694 |
| Test | 0,925 | 0,650 | 0,706 | 0,559 |

So với baseline 5 epoch trên test: precision tăng `+0,019`, recall tăng `+0,038`, mAP50 giảm `-0,045`, còn mAP50-95 tăng `+0,029`. Vì vậy bản 10 epoch bắt được nhiều screw hơn và định vị ở ngưỡng IoU cao tốt hơn, nhưng mAP50 giảm trên test; cần giữ cả hai checkpoint để chọn theo mục tiêu recall hay độ chính xác tổng thể.

Checkpoint fine-tune:

`runs/obb/runs/screw_yolo_obb/m4_finetune10_from_best/weights/best.pt`

Kết quả theo epoch:

`runs/obb/runs/screw_yolo_obb/m4_finetune10_from_best/results.csv`

Kết quả evaluate riêng:

`runs/obb/runs/screw_yolo_obb_eval_m4_finetune10/val`

`runs/obb/runs/screw_yolo_obb_eval_m4_finetune10/test`

### Dataset corrected và train mới ở `imgsz=1024`

Đã sửa resolver ảnh WEEE để ưu tiên ảnh RGB trong `images/train|val|test`, build lại dataset tại:

`datasets/screw_obb_combined_screw_only_fixed`

Kiểm tra dataset corrected: 945/945 ảnh WEEE là RGB, không có file mask nhỏ hơn 10 KB; toàn bộ nhãn train/val/test hợp lệ, không có lỗi hình học. Sau đó train mới từ `yolo11n-obb.pt`, 10 epoch, `imgsz=1024`, batch 2 trên Apple M4/MPS.

| Tập | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Val | 0,902 | 0,863 | 0,927 | 0,717 |
| Test | 0,910 | 0,862 | 0,942 | 0,703 |

Checkpoint mới:

`runs/obb/runs/screw_yolo_obb/m4_fixed1024_10/weights/best.pt`

Kết quả evaluate:

`runs/obb/runs/screw_yolo_obb_eval_m4_fixed1024_10/val`

`runs/obb/runs/screw_yolo_obb_eval_m4_fixed1024_10/test`

Bước tiếp theo là evaluate riêng trên WEEE/MVTec, kiểm tra recall trên ảnh có nhiều screw, rồi dùng checkpoint corrected để crop screw và chuyển crop sang mô hình GOOD/DEFECT.

### Smoke test synthetic augmentation trên OBB

Đã thử tăng cường trực tiếp từ các OBB của tập train corrected. Mỗi mẫu synthetic được tạo trên **toàn bộ ảnh**, không lưu crop rời: scene được xoay/scale/translate, các OBB được biến đổi theo ảnh, sau đó áp dụng một số biến đổi cục bộ/toàn ảnh như blur, thay đổi sáng, noise và grayscale. Vì vậy label vẫn là OBB đúng với ảnh mới.

Script:

`scripts/augment_screw_obb.py`

Dataset synthetic:

`datasets/screw_obb_combined_screw_only_fixed_aug`

Thống kê:

| Split | Số ảnh | Số OBB | Ghi chú |
|---|---:|---:|---|
| Train | 1.540 | 8.149 | 1.034 ảnh gốc + 506 ảnh synthetic |
| Val | 145 | 811 | Giữ nguyên, không augment |
| Test | 150 | 832 | Giữ nguyên, không augment |

Validator: **0 lỗi**. Smoke train dùng YOLO11n-OBB, `imgsz=1024`, 5 epoch, batch 2 trên Apple M4/MPS.

Checkpoint:

`runs/obb/runs/screw_yolo_obb/m4_fixed1024_aug5/weights/best.pt`

Kết quả đo trên test gốc, không phải test synthetic:

| Tập | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Val | 0,891 | 0,846 | 0,923 | 0,672 |
| Test | 0,916 | 0,870 | 0,936 | 0,673 |

Kết quả evaluate:

`runs/obb/runs/screw_yolo_obb_eval_m4_fixed1024_aug5/val`

`runs/obb/runs/screw_yolo_obb_eval_m4_fixed1024_aug5/test`

So với baseline corrected `imgsz=1024`, 10 epoch trên test (`P=0,910`, `R=0,862`, `mAP50=0,942`, `mAP50-95=0,703`), smoke augment 5 epoch tăng nhẹ precision/recall nhưng giảm nhẹ mAP50 và giảm mAP50-95. Đây **chưa đủ để kết luận augmentation tốt hơn**, vì số epoch khác nhau. Cần train cùng 10 epoch hoặc dùng cùng tiêu chí chọn checkpoint trước khi quyết định giữ augmentation.

Kết luận hiện tại: pipeline augmentation chạy đúng và nhãn OBB hợp lệ, nhưng chưa chứng minh được cải thiện. MVTec AD vẫn chưa đưa vào train vì chưa có bounding box object-level cho screw; ảnh anomaly mask không thể tự động thay thế nhãn OBB chính xác.
