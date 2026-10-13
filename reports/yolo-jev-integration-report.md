# Báo cáo đánh giá tích hợp YOLO và Jev AI

Ngày: 2026-10-10
Dự án: Industrial Vision Anomaly Detection

## 1. Tóm tắt quyết định

| Hướng | Quyết định | Vai trò đề xuất |
|---|---|---|
| YOLO | **GO PoC** | Phát hiện nhiều sản phẩm trong ảnh toàn cảnh |
| Pipeline hiện tại | **Giữ lại** | Phân loại từng sản phẩm và phát hiện anomaly |
| Jev AI | **GO shadow-mode** | Triage/manual review cho ca mơ hồ |
| Jev AI làm detector chính | **NO-GO** | Jev không thay thế YOLO, DINOv2, PatchCore/kNN/PaDiM |

Kiến trúc phù hợp nhất:

```text
Ảnh toàn cảnh
  → YOLO phát hiện N sản phẩm
  → crop + pad từng bounding box
  → batch DINOv2 hiện có
  → route category từng crop
  → exact kNN / PatchCore / PaDiM
  → anomaly score + heatmap từng instance
  → luật local tổng hợp kết quả toàn ảnh
  → Jev tùy chọn cho ca mơ hồ/manual review
```

## 2. Hiện trạng dự án

Repository hiện đang xử lý theo giả định một ảnh tương ứng một sản phẩm:

- DINOv2 ViT-B/14 frozen tạo global embedding và patch embedding.
- Classifier/router chọn product category.
- Anomaly detector gồm `exact_knn`, `patchcore` và `padim`.
- Input contract chính là ảnh RGB `256×256`.
- CLI hiện tại là inference một ảnh; `requirements.txt` chưa có `ultralytics` hoặc `typesafe-sdk`.

Vì vậy, YOLO cần được thêm ở trước pipeline, còn logic anomaly hiện tại nên được tái sử dụng theo từng crop. Dataset MVTec hiện tại là dữ liệu chủ yếu cho ảnh đơn sản phẩm; không nên dùng nó làm bằng chứng rằng detector đã hoạt động tốt trên ảnh nhiều sản phẩm.

## 3. Hướng YOLO cho một ảnh nhiều sản phẩm

### Cách dùng

Giai đoạn đầu nên huấn luyện YOLO với một class duy nhất là `product`. YOLO chỉ trả lời “có bao nhiêu sản phẩm và ở đâu”; category cụ thể vẫn do classifier DINOv2 hiện có đảm nhiệm. Cách tách này giúp tái sử dụng category-specific feature banks và dễ mở rộng sản phẩm mới.

YOLO26 hiện hỗ trợ detection, segmentation, nhiều kích thước model và tùy chọn inference không cần NMS. Tài liệu chính thức công bố ở input 640: YOLO26n khoảng 38.9 ms CPU ONNX, YOLO26s khoảng 87.2 ms, nhưng cần benchmark lại trên phần cứng triển khai. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)

Khuyến nghị PoC: bắt đầu với `YOLO26s`; nếu ưu tiên CPU/edge thì thử `YOLO26n`, nếu sản phẩm nhỏ hoặc bị che nhiều thì cân nhắc model lớn hơn hoặc segmentation. Không nên chọn model chỉ theo benchmark COCO.

### Dữ liệu cần bổ sung

```text
datasets/multi_product/
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
  data.yaml
```

Mỗi sản phẩm cần bounding box theo format YOLO:

```text
class_id x_center y_center width height
```

Nên lưu thêm manifest per-instance gồm `image_id`, `instance_id`, `bbox`, category, trạng thái lỗi và mask nếu có. Chia train/validation/test theo ca chụp, batch hoặc session camera; không chia ngẫu nhiên các frame gần như giống nhau.

Dataset phải bao phủ sản phẩm nhỏ, chồng lấn, bị che, bị cắt khỏi ảnh, nền trống, phản chiếu và thay đổi ánh sáng. Format dataset và cấu trúc YAML tham khảo [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect).

### Crop contract

Với mỗi detection:

1. Mở rộng bounding box khoảng 5–15%.
2. Clip vào biên ảnh.
3. Pad thành ảnh vuông, không kéo méo sản phẩm.
4. Resize về `256×256` theo contract hiện tại.
5. Gom các crop thành batch để chạy một forward DINOv2.
6. Lưu phép biến đổi crop → ảnh gốc để reproject anomaly heatmap.

Kết quả mỗi instance nên có `bbox`, detection confidence, category, routing status, anomaly score, threshold, label và đường dẫn heatmap.

Scene-level policy nên cố định bằng code:

```text
Có instance DEFECT rõ ràng       → DEFECT
Thiếu/không rõ/che khuất nhiều   → MANUAL_REVIEW
Category unknown                 → MANUAL_REVIEW
Tất cả GOOD và đủ confidence      → GOOD
```

Không tái sử dụng trực tiếp feature bank hoặc threshold được xây dựng từ ảnh full-frame; chúng phải được xây dựng/calibrate lại trên crop sản phẩm normal.

### Đánh giá

- YOLO: mAP50-95, precision, recall, missed-object rate, count error, AP theo kích thước.
- Anomaly crop-level: AUROC/AP, F1, false-accept, false-reject, pixel AUROC/AP, Dice/IoU.
- End-to-end: per-instance recall, scene-level false accept/false reject, p50/p95 latency.
- Báo cáo riêng hai chế độ: ground-truth boxes (đo anomaly thuần) và predicted boxes (đo hệ thống thực tế).

## 4. Jev AI của TypeSafe

Jev là model System One: nhận `state` và các câu hỏi typed như `Choice`, `Score`, `Noul`, sau đó trả về kết quả có cấu trúc và probability/confidence để code sử dụng. [TypeSafe Introduction](https://docs.typesafe.ai/introduction/)

Jev **không nhận ảnh, audio hoặc video**; `state` là string, JSON object hoặc array của text. Vì vậy Jev không phù hợp để phát hiện object, tính anomaly score, tạo heatmap hoặc thay thế DINOv2. [State](https://docs.typesafe.ai/concepts/state)

API hiện dùng endpoint:

```text
POST https://api.typesafe.ai/v1/systemone
```

Python SDK chính thức là `typesafe-sdk`, hỗ trợ client sync/async. [API reference](https://docs.typesafe.ai/api) · [Python SDK](https://docs.typesafe.ai/sdk/python)

### Use case phù hợp

Chỉ gửi structured evidence đã rút gọn, không gửi ảnh/crop/heatmap/embedding:

```json
{
  "category": "bottle",
  "local_label": "DEFECT",
  "score_bucket": "near_threshold",
  "image_quality": "good",
  "quality_flags": ["possible_occlusion"],
  "operator_note": "possible contamination near cap"
}
```

Jev có thể hỗ trợ:

- quyết định có cần chuyển human review hay không;
- xếp mức ưu tiên review;
- phân loại ghi chú lỗi bằng text;
- kiểm tra inspection record có đủ thông tin hay chưa.

Jev không được override `DEFECT` cứng. Khi timeout, 429, 5xx hoặc mất mạng, hệ thống phải giữ quyết định local hoặc chuyển `MANUAL_REVIEW`, tuyệt đối không tự động đổi thành `GOOD`.

Nên pin model version trong production thay vì dùng alias không cố định. Tài liệu TypeSafe ghi rõ alias có thể đổi khi phát hành model mới; model hiện được nêu là `jev-1.13.0`. [TypeSafe Models](https://docs.typesafe.ai/models)

### Rủi ro

- Network latency, rate limit và outage.
- Dữ liệu sản xuất rời khỏi máy nội bộ.
- Model/version drift nếu dùng alias.
- Độ chính xác tiếng Việt cần kiểm thử riêng; TypeSafe ghi tiếng Anh là ngôn ngữ chính.
- Tăng dependency và chi phí vận hành nhưng không cải thiện trực tiếp vision anomaly.

TypeSafe công bố không dùng input để train/fine-tune model, nhưng chính sách vẫn mô tả việc lưu giữ dữ liệu trong thời gian cần thiết cho dịch vụ và chia sẻ với service providers. Cần review DPA, retention, vùng xử lý dữ liệu và yêu cầu enterprise trước khi gửi thông tin nhạy cảm. [TypeSafe Privacy Policy](https://typesafe.ai/legal/privacy-policy)

## 5. Kiến trúc tích hợp đề xuất

### Ranh giới local/cloud

| Thành phần | Local | Jev cloud |
|---|---:|---:|
| Ảnh gốc, YOLO, DINOv2 | Có | Không |
| Feature bank, threshold, anomaly map | Có | Không |
| Luật an toàn và quyết định defect cứng | Có | Không |
| Structured evidence | Tạo | Nhận |
| Triage/manual-review | Có thể | Có thể hỗ trợ |
| Fallback khi mất mạng | Có | Không phụ thuộc |

### Module tối thiểu

```text
src/mvtec_system/detection/yolo.py
src/mvtec_system/production/multi_instance.py
src/mvtec_system/integration/jev.py
```

Chưa cần thêm FastAPI, Kafka, Redis, database hoặc dashboard cho PoC.

### Kế hoạch theo giai đoạn

**P0 — Đóng băng baseline**

- Lưu metric, threshold, heatmap và latency của pipeline hiện tại.
- Chốt trạng thái `GOOD / DEFECT / UNKNOWN / MANUAL_REVIEW`.

**P1 — YOLO local**

- Thu thập ảnh toàn cảnh thực tế.
- Gắn bounding box từng sản phẩm.
- Train/validate model một class `product`.
- Đo recall sản phẩm trước khi nối anomaly detector.

**P2 — Multi-instance anomaly**

- YOLO → crop adapter → batch DINOv2.
- Route category từng crop.
- Chạy exact kNN trước, sau đó so sánh PatchCore/PaDiM.
- Reproject heatmap và aggregate scene-level.

**P3 — Jev shadow mode**

- Gọi Jev chỉ cho ca borderline/conflicting.
- Không cho output Jev ảnh hưởng quyết định.
- Ghi latency, lỗi, token usage, model version, confidence và mức đồng thuận với rule local.

**P4 — Pilot có feature flag**

- Bật/tắt Jev độc lập.
- Không có API key vẫn chạy được local pipeline.
- Cloud lỗi → `MANUAL_REVIEW`.
- Canary trên một camera/line trước khi mở rộng.

## 6. Go/no-go và khuyến nghị cuối

**Nên làm ngay:** YOLO local + crop adapter + batch DINOv2 + anomaly theo instance. Đây là hướng giải quyết trực tiếp bài toán một ảnh nhiều sản phẩm và tận dụng được phần lớn code hiện tại.

**Chỉ nên thử nghiệm:** Jev ở shadow-mode cho triage văn bản/manual review. Chỉ giữ lại nếu chứng minh được giảm review thủ công hoặc giảm false-negative mà không tăng false-accept.

**Không nên làm:** gửi ảnh/heatmap lên Jev, dùng Jev thay detector, cho Jev override defect cứng, hoặc dùng MVTec single-product làm dataset train YOLO production.

## 7. Nguồn chính

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics object detection datasets](https://docs.ultralytics.com/datasets/detect)
- [TypeSafe Introduction](https://docs.typesafe.ai/introduction/)
- [TypeSafe State](https://docs.typesafe.ai/concepts/state)
- [TypeSafe Models](https://docs.typesafe.ai/models)
- [TypeSafe API reference](https://docs.typesafe.ai/api)
- [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python)
- [TypeSafe Privacy Policy](https://typesafe.ai/legal/privacy-policy)
