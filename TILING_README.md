# Hướng Dẫn Áp Dụng Kỹ Thuật Tiling cho RetinaNet

## Mục Lục

1. [Tiling là gì và tại sao cần dùng?](#1-tiling-là-gì-và-tại-sao-cần-dùng)
2. [Phân tích vấn đề trong dataset hiện tại](#2-phân-tích-vấn-đề-trong-dataset-hiện-tại)
3. [Tổng quan kiến trúc thay đổi](#3-tổng-quan-kiến-trúc-thay-đổi)
4. [Các file cần thay đổi](#4-các-file-cần-thay-đổi)
5. [Chi tiết từng thay đổi](#5-chi-tiết-từng-thay-đổi)
6. [File mới cần tạo](#6-file-mới-cần-tạo)
7. [Pipeline tổng thể sau khi áp dụng Tiling](#7-pipeline-tổng-thể-sau-khi-áp-dụng-tiling)
8. [Các thông số quan trọng cần tinh chỉnh](#8-các-thông-số-quan-trọng-cần-tinh-chỉnh)
9. [Lưu ý và rủi ro](#9-lưu-ý-và-rủi-ro)

---

## 1. Tiling là gì và tại sao cần dùng?

### Khái niệm

**Tiling** (hay còn gọi là **Sliding Window** hoặc **Image Chipping**) là kỹ thuật chia một ảnh lớn thành nhiều ảnh con nhỏ hơn (gọi là **tile**) có thể chồng lên nhau (**overlap**), sau đó xử lý từng tile riêng lẻ.

**Minh họa lưới tile trượt trên ảnh gốc 1280×1024 (overlap 20%):**

```
Ảnh gốc 1280×1024 pixels
┌──────────────────────────────────────────────────┐
│←──────── 640px ────────→│                        │
│ ┌────────────────────┐  │                        │
│ │     tile_00        │  │                        │
│ │                    ├──┤←── Vùng OVERLAP 128px  │
│ │                    │  │  (2 tile chồng nhau)   │
│ └────────────────────┘  │                        │  ↑ 512px
│         │←──── 512px ───────→│                   │
│         │ ┌────────────────────┐                 │
│         │ │     tile_01        │                 │
│         │ │                    │                 │
│         │ └────────────────────┘                 │
├─────────────────────────────────────────────── ←─┤ Vùng OVERLAP dọc (102px)
│ ┌────────────────────┐  ┌────────────────────┐   │  ↓
│ │     tile_10        │  │     tile_11        │   │
│ │                    │  │                    │   │
│ └────────────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────┘

  Kết quả: 1 ảnh gốc → 4 tiles (có thể tới 6 do overlap ở cạnh)
  Người trong ảnh gốc: ~15px → Người trong tile: ~30px ✓
```

> **Tại sao cần Overlap?** Không có overlap, người đứng ở ranh giới giữa 2 tile sẽ bị cắt đôi — một nửa nằm trong tile trái, một nửa trong tile phải. Model sẽ không nhận ra. Overlap đảm bảo mỗi người luôn xuất hiện **nguyên vẹn trong ít nhất 1 tile**.

### Tại sao dataset này cần Tiling?

Dataset drone thermal của bạn có đặc điểm:
- **Kích thước ảnh gốc:** 1280×1024 pixels
- **Đối tượng cần phát hiện:** người (rất nhỏ trong ảnh drone)
- **Kích thước training hiện tại:** resize về 640px

Khi resize 1280×1024 → 640px, tỷ lệ resize là **0.5x**. Điều này có nghĩa:
- Một người cao 30px trong ảnh gốc → còn **15px** sau khi resize
- Anchor box nhỏ nhất của bạn (area=4, stride=8) → kích thước ~16px

**Vấn đề:** Các đối tượng quá nhỏ sau khi resize sẽ bị mất thông tin và khó phát hiện.

**Giải pháp Tiling:** Thay vì resize toàn bộ ảnh, chia ảnh thành 4 tile (640×512), mỗi tile chứa ít đối tượng hơn nhưng mỗi đối tượng **lớn gấp đôi** so với khi resize toàn ảnh.

---

## 2. Phân tích vấn đề trong dataset hiện tại

### Cấu trúc data hiện tại

```
Drone Thermal.v4i.voc/
├── imgs/           # 2,866 ảnh JPEG (1280×1024)
├── anno/           # 2,866 file XML (Pascal VOC)
├── csv_file/
│   ├── train_data.csv   # 2,006 ảnh
│   ├── valid_data.csv   # 573 ảnh
│   └── test_data.csv    # 287 ảnh
└── weight_store/   # Model weights đã lưu
```

### Format CSV hiện tại (quan trọng để hiểu)

```
path_img,bbox,class_id
/path/to/img.jpg,"[[x_c, y_c, w, h], [x_c, y_c, w, h]]","[1, 1]"
```

Trong đó tọa độ bbox là **[x_center, y_center, width, height]** tính bằng pixel trong ảnh gốc.

### Pipeline hiện tại (tóm tắt)

```
CSV → read_img_and_label() → resize_and_pad_img(target=640) → pack_targets() → model
```

---

## 3. Tổng quan kiến trúc thay đổi

Tiling ảnh hưởng đến **2 giai đoạn**:

### Giai đoạn 1: Preprocessing (tạo data)

```
TRƯỚC:  1 ảnh → resize → 1 sample training
SAU:    1 ảnh → tiling → N tiles → N samples training
```

### Giai đoạn 2: Inference (dự đoán)

```
TRƯỚC:  1 ảnh → resize → model → N boxes (trong ảnh nhỏ) → scale lại
SAU:    1 ảnh → tiling → N tiles → model × N → merge boxes → Global NMS
```

---

## 4. Các file cần thay đổi

### Tóm tắt nhanh

| File | Loại thay đổi | Mức độ thay đổi |
|------|--------------|----------------|
| `Xu_ly_du_lieu/main_step_123.py` | Thêm logic tiling khi tạo CSV | **Trung bình** |
| `Xu_ly_du_lieu/preprocessing_data_before_training.py` | Thay `resize_and_pad_img` | **Nhỏ** |
| `predict.py` / `predict_new.py` | Thêm tiled inference + merge NMS | **Lớn** |
| `evaluate_model.py` | Dùng tiled inference để evaluate | **Trung bình** |
| `train_adam.py` | Điều chỉnh batch_size, target_size | **Nhỏ** |

### Các file KHÔNG cần thay đổi

- `RetinaNet/main.py` — Model architecture không đổi
- `resnet_50/main.py` — Backbone không đổi
- `FPN/main.py` — FPN không đổi
- `Anchor_box/main.py` — **Cần xem lại** (xem Phần 8 — mục Anchor Boxes)
- `Label_encode/main.py` — Label encoding không đổi

---

## 5. Chi tiết từng thay đổi

### 5.1. `Xu_ly_du_lieu/main_step_123.py` — Tạo CSV với tiling

> **File thực tế:** `Xu_ly_du_lieu/main_step_123.py`
>
> **Thêm ở đầu file (sau các import hiện có):**
> ```python
> from tiling_utils import generate_tile_coords, get_bboxes_for_tile
> import cv2  # hoặc PIL để crop và lưu tile
> ```
>
> **Sửa logic chính trong vòng lặp `for xml_file in glob.glob(...)` (dòng 17–64):**
>
> Hiện tại: sau khi đọc xong tất cả object của 1 ảnh, code làm:
> ```python
> value = {'path_img': path_img, 'bbox': bboxes_of_img, 'class_id': class_ids_of_img}
> xml_list.append(value)   # ← 1 ảnh = 1 dòng
> ```
>
> Cần thay thành: sau khi có `bboxes_of_img` và `class_ids_of_img`, **không append ngay** mà:
> ```python
> tile_coords = generate_tile_coords(width_img, height_img, tile_w=640, tile_h=512, overlap=0.2)
> for (tx, ty, tw, th) in tile_coords:
>     tile_bboxes, tile_cls = get_bboxes_for_tile(bboxes_of_img, class_ids_of_img, tx, ty, tw, th)
>     tile_img_path = crop_and_save_tile(path_img, tx, ty, tw, th)  # lưu ra disk
>     xml_list.append({'path_img': tile_img_path, 'bbox': tile_bboxes, 'class_id': tile_cls})
> ```
>
> **Sửa tên file output CSV (dòng 71):** đổi tên từ `data_information_grouped_thermal_drone.csv`
> sang `data_information_tiled.csv` để phân biệt với CSV gốc.

**Mục tiêu:** Thay vì mỗi ảnh gốc tạo ra 1 dòng CSV, giờ tạo ra N dòng (N = số tiles).

**Logic cần thêm:**

```
Với mỗi ảnh XML:
  1. Đọc tất cả bbox gốc
  2. Gọi hàm generate_tiles(img_w=1280, img_h=1024, tile_w=640, tile_h=512, overlap=0.2)
     → trả về list các (tile_x_offset, tile_y_offset, tile_w, tile_h)
  3. Với mỗi tile:
     a. Lọc bbox nào nằm TRONG tile (có phần chồng lên tile đủ lớn)
     b. Chuyển tọa độ bbox từ hệ ảnh gốc → hệ tile
     c. Clip bbox nếu vượt biên tile
     d. Ghi 1 dòng CSV: (tile_crop_info, bbox_trong_tile, class_id)
```

**Thông số gợi ý cho ảnh 1280×1024:**
- `tile_w = 640`, `tile_h = 512` (đúng 1/4 ảnh gốc)
- `overlap = 0.2` (20% overlap = 128px theo chiều ngang, 102px theo chiều dọc)
- Kết quả: mỗi ảnh → khoảng 4–6 tiles (do overlap)

**Xử lý bbox khi tiling:**

```
bbox_gốc = [x_c, y_c, w, h] (pixel, hệ ảnh gốc)

Chuyển sang corners:
  xmin = x_c - w/2
  ymin = y_c - h/2
  xmax = x_c + w/2
  ymax = y_c + h/2

Kiểm tra overlap với tile [tx, ty, tx+tw, ty+th]:
  x_overlap = max(0, min(xmax, tx+tw) - max(xmin, tx))
  y_overlap = max(0, min(ymax, ty+th) - max(ymin, ty))
  iou_with_tile = (x_overlap * y_overlap) / (w * h)

Nếu iou_with_tile >= 0.3: giữ lại bbox này trong tile

Chuyển tọa độ sang hệ tile:
  new_xmin = max(xmin - tx, 0)
  new_ymin = max(ymin - ty, 0)
  new_xmax = min(xmax - tx, tw)
  new_ymax = min(ymax - ty, th)
  new_x_c = (new_xmin + new_xmax) / 2
  new_y_c = (new_ymin + new_ymax) / 2
  new_w   = new_xmax - new_xmin
  new_h   = new_ymax - new_ymin
```

**Lưu tile như thế nào — 2 lựa chọn:**

- **Option A (đơn giản):** Lưu ảnh tile thực ra đĩa vào thư mục `tiles/`. CSV lưu path đến file tile.
  - Ưu điểm: load nhanh trong lúc train
  - Nhược điểm: tốn thêm ~10–15 GB disk

- **Option B (tiết kiệm disk):** CSV lưu path ảnh gốc + thông tin crop `(tx, ty, tw, th)`. Lúc load sẽ crop on-the-fly.
  - Ưu điểm: không tốn disk thêm
  - Nhược điểm: load chậm hơn một chút

**Gợi ý: dùng Option A** vì dataset chỉ ~2,866 ảnh, tile output ~8,000–15,000 ảnh, dung lượng chấp nhận được.

---

### 5.2. `Xu_ly_du_lieu/preprocessing_data_before_training.py` — Load tile thay vì ảnh gốc

> **File thực tế:** `Xu_ly_du_lieu/preprocessing_data_before_training.py`
>
> **Không cần sửa bất kỳ hàm nào** trong file này nếu dùng Option A (lưu tile ra disk).
> Tất cả 4 hàm (`change_string2number`, `create_dataset_from_dataframe`, `read_img_and_label`,
> `resize_and_pad_img`) đều hoạt động đúng với tile image, vì tile cũng chỉ là file JPEG bình thường.
>
> **Điều duy nhất cần chú ý:** `target_size` được truyền vào `resize_and_pad_img` từ bên ngoài
> (ở `train_adam.py` dòng 69 và 76), nên không cần sửa file này mà chỉ sửa giá trị ở `train_adam.py`.

**Thay đổi nhỏ:** Hàm `resize_and_pad_img` vẫn dùng được, chỉ cần điều chỉnh `target_size`.

- Tile size = 640×512, sau khi pad thành 640×640 → dùng `target_size = 640`
- Hoặc nếu tile 640×512 → resize về 512×512 → dùng `target_size = 512`

Không cần thay đổi nhiều nếu lưu tile ra file (Option A).

---

### 5.3. `train_adam.py` — Điều chỉnh tham số training

> **File thực tế:** `train_adam.py`
>
> **Sửa dòng 45–46:**
> ```python
> # DÒNG 45 - ĐỔI:
> target_size = 640  →  target_size = 512
>
> # DÒNG 46 - ĐỔI:
> batch_size = 4     →  batch_size = 8
> ```
>
> **Sửa dòng 57–58** (đường dẫn CSV) — trỏ sang CSV tile mới:
> ```python
> # DÒNG 57 - ĐỔI:
> train_csv_path = '.../csv_file/train_data.csv'
> →  train_csv_path = '.../csv_file/train_tiled_data.csv'
>
> # DÒNG 58 - ĐỔI:
> val_csv_path = '.../csv_file/valid_data.csv'
> →  val_csv_path = '.../csv_file/valid_tiled_data.csv'
> ```
>
> **Sửa dòng 166** (tên file weight lưu ra) — đổi tên để không ghi đè model cũ:
> ```python
> filepath=weight_folder + "/retinanet_gamma_3.0.weights.h5"
> →  filepath=weight_folder + "/retinanet_tiled_512.weights.h5"
> ```
>
> **Sửa dòng 194** (tên file history CSV) — tương tự:
> ```python
> hist_df.to_csv(hist_store_folder + '/retinanet_gamma_3.0.csv')
> →  hist_df.to_csv(hist_store_folder + '/retinanet_tiled_512.csv')
> ```

**Thay đổi nhỏ:**

```python
# TRƯỚC
target_size = 640
batch_size = 4

# SAU (tile nhỏ hơn → có thể tăng batch_size)
target_size = 512   # hoặc 640 tùy tile size bạn chọn
batch_size = 8      # tăng được vì mỗi ảnh nhỏ hơn
```

**Lưu ý:** Số lượng training samples tăng ~4x (vì mỗi ảnh → ~4 tiles), nhưng mỗi tile ít object hơn → cần điều chỉnh `epochs` cho phù hợp.

---

### 5.4. `predict.py` — Tiled Inference (thay đổi lớn nhất)

> **File thực tế:** `predict.py`
>
> **Sửa dòng 12–15** (config):
> ```python
> # DÒNG 12 - ĐỔI:
> TARGET_SIZE = 512   # giữ nguyên, đây là kích thước tile sau khi resize
>
> # DÒNG 15 - ĐỔI đường dẫn weight sang model mới train với tile:
> WEIGHT_PATH = '.../weight_store/retinanet_tiled_512.weights.h5'
> ```
>
> **Thêm import ở đầu file** (sau các import hiện có):
> ```python
> from Xu_ly_du_lieu.tiling_utils import generate_tile_coords, merge_tile_detections, apply_global_nms
> ```
>
> **Giữ nguyên 2 hàm hiện có:**
> - `decode_box_predictions()` (dòng 19–49) — **không đổi gì**
> - `get_inference_model()` (dòng 52–58) — **không đổi gì**
> - `run_inference()` (dòng 61–103) — **không đổi gì**, vẫn dùng để inference 1 tile
>
> **Thêm hàm mới `run_tiled_inference()`** (thêm sau `run_inference()`):
> Hàm này gọi `run_inference()` nhiều lần (1 lần/tile) rồi gộp lại bằng Global NMS.
> Xem chi tiết logic ở Phần 5 — Bước 1 đến Bước 4 bên dưới.
>
> **Sửa phần `if __name__ == '__main__'` (dòng 106 trở đi):**
> Thay `run_inference(...)` thành `run_tiled_inference(...)`. Phần vẽ kết quả **không cần đổi**
> vì output của `run_tiled_inference` trả về cùng format với `run_inference`.
>
> **Lưu ý quan trọng về tọa độ:** `run_inference()` hiện trả về `ratio` (tỷ lệ resize).
> Trong tiled inference, mỗi tile có ratio riêng → `run_tiled_inference()` sẽ trả về
> boxes đã ở hệ tọa độ ảnh gốc, **không cần** scale lại bằng ratio nữa.
> Cần sửa phần vẽ Ground Truth (dòng 146–153) để không nhân `* ratio` nữa.

Đây là phần phức tạp nhất. Thay vì resize ảnh và inference 1 lần, cần:

**Bước 1: Chia ảnh thành tiles**
```
input_image (1280×1024) → [tile_00, tile_01, tile_10, tile_11, ...]
```

**Bước 2: Inference từng tile**
```
Với mỗi tile:
  - Resize tile về target_size (vd 512×512)
  - Chạy model → boxes (trong hệ tile đã resize)
  - Scale boxes về hệ tile gốc (×(tile_w/target_size))
  - Dịch chuyển boxes về hệ ảnh gốc (+ tile_offset)
```

**Bước 3: Gộp tất cả boxes từ các tiles**
```
all_boxes = concat([boxes_tile_00, boxes_tile_01, boxes_tile_10, boxes_tile_11])
```

> ⚠️ **QUAN TRỌNG — Tại sao phải có Bước 4?**
> Ở vùng overlap, 2 tile cùng chứa 1 người → cùng dự đoán ra 2 box cho cùng 1 người.
> Nếu bỏ qua Bước 4, mỗi người ở vùng ranh giới sẽ bị **2 box chồng đè** lên nhau.

**Bước 4: Global NMS — BẮT BUỘC, KHÔNG ĐƯỢC BỎ QUA**
```
final_boxes = global_nms(all_boxes, iou_threshold=0.5, score_threshold=0.3)

Minh họa trước/sau Global NMS:
  Trước NMS:                       Sau NMS:
  ┌──────────┐                     ┌──────────┐
  │  box A   │  ← từ tile_00       │  box A   │  ← giữ lại (score cao hơn)
  │ ┌──────────┐                   └──────────┘
  └─│  box B   │ ← từ tile_01
    └──────────┘
  (cùng 1 người, 2 box chồng nhau)
```

**Xử lý chuyển đổi tọa độ (quan trọng):**

```
box_trong_resized_tile → box_trong_tile_gốc:
  scale = tile_w / target_size  (hoặc tile_h / target_size)
  box_x = box_x_resized × scale_x
  box_y = box_y_resized × scale_y

box_trong_tile_gốc → box_trong_ảnh_gốc:
  box_xmin_global = box_xmin_tile + tile_x_offset
  box_ymin_global = box_ymin_tile + tile_y_offset
  box_xmax_global = box_xmax_tile + tile_x_offset
  box_ymax_global = box_ymax_tile + tile_y_offset
```

**Vấn đề box nằm ở ranh giới 2 tile (do overlap):**
- Box ở vùng overlap có thể được phát hiện 2 lần (từ 2 tile khác nhau)
- **Giải pháp:** Global NMS sau khi gộp sẽ xử lý tự động — đây là lý do cần overlap

---

### 5.5. `evaluate_model.py` — Dùng tiled inference

> **File thực tế:** `evaluate_model.py`
>
> **Sửa dòng 7** (import):
> ```python
> # DÒNG 7 - ĐỔI:
> from predict import get_inference_model, run_inference, TARGET_SIZE, WEIGHT_PATH, TEST_CSV
> →  from predict import get_inference_model, run_tiled_inference, TARGET_SIZE, WEIGHT_PATH, TEST_CSV
> ```
>
> **Sửa dòng 40** — anchors cho tile, không phải full ảnh:
> ```python
> # DÒNG 40 - KIỂM TRA: TARGET_SIZE phải = tile size (512), không phải size ảnh gốc
> all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
> # → giữ nguyên nếu TARGET_SIZE = 512 (= tile size) ✓
> ```
>
> **Sửa dòng 52–53** (gọi inference):
> ```python
> # DÒNG 52-53 - ĐỔI:
> _, pred_boxes, pred_scores, _, num_det, ratio = run_inference(
>     model, img_path, all_anchors, score_threshold=0.5
> )
> →  pred_boxes, pred_scores, num_det = run_tiled_inference(
>     model, img_path, all_anchors, score_threshold=0.05  # thấp hơn để vẽ PR curve
> )
> ```
>
> **Sửa dòng 57–65** (xử lý Ground Truth):
> ```python
> # DÒNG 59-60 - XÓA dòng nhân ratio vì boxes đã ở hệ ảnh gốc:
> x_c, y_c, w, h = x_c * ratio, y_c * ratio, w * ratio, h * ratio  ← XÓA dòng này
> ```
>
> Phần còn lại của `evaluate_model()` (tính TP/FP, AP, vẽ PR curve) **không cần đổi gì**.

Thay hàm `run_inference()` đơn giản bằng `run_tiled_inference()` mới.
Metrics (precision, recall, AP) tính như cũ, chỉ thay cách lấy predictions.

---

## 6. File mới cần tạo

### `Xu_ly_du_lieu/tiling_utils.py` — Module Tiling chính

File này tập trung toàn bộ logic tiling để tái sử dụng ở preprocessing VÀ inference:

```python
# Các hàm cần implement:

def generate_tile_coords(img_w, img_h, tile_w, tile_h, overlap):
    """
    Trả về list [(x_offset, y_offset, actual_tile_w, actual_tile_h), ...]
    Xử lý cả edge tiles (tile ở góc/cạnh ảnh có thể nhỏ hơn)
    """

def get_bboxes_for_tile(bboxes, class_ids, tile_x, tile_y, tile_w, tile_h, min_visibility=0.3):
    """
    Lọc và chuyển đổi bbox từ hệ ảnh gốc sang hệ tile.
    min_visibility: % diện tích bbox phải nằm trong tile để giữ lại
    Trả về (new_bboxes, new_class_ids) trong hệ tọa độ tile
    """

def crop_tile_from_image(image_path, tile_x, tile_y, tile_w, tile_h):
    """
    Crop tile từ ảnh gốc và trả về numpy array
    """

def merge_tile_detections(tile_detections, tile_coords, orig_w, orig_h):
    """
    Gộp boxes từ tất cả tiles về hệ tọa độ ảnh gốc.
    tile_detections: [(boxes, scores, classes), ...] cho mỗi tile
    tile_coords: [(tx, ty, tw, th), ...] tương ứng
    Trả về (all_boxes, all_scores, all_classes) trong hệ ảnh gốc
    """

def apply_global_nms(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.3):
    """
    NMS trên toàn bộ predictions sau khi gộp từ các tiles.
    Trả về (final_boxes, final_scores, final_classes)
    """
```

---

## 7. Pipeline tổng thể sau khi áp dụng Tiling

### Training Pipeline mới

```
Ảnh gốc (1280×1024) + XML annotations
            ↓
  [main_step_123.py với tiling]
            ↓
  ~8,000–15,000 tile images (640×512) + tile CSVs
  (mỗi tile chứa bbox đã chuyển về hệ tọa độ tile)
            ↓
  preprocessing_data_before_training.py
  (resize tile 640×512 → 512×512 hoặc 640×640)
            ↓
  pack_targets() → anchor matching
  (anchor boxes khớp tốt hơn vì object lớn hơn tương đối)
            ↓
  model.fit() — train như bình thường
```

### Inference Pipeline mới

```
Input: ảnh drone mới (1280×1024)
            ↓
  generate_tile_coords(1280, 1024, 640, 512, overlap=0.2)
  → [(0,0,640,512), (512,0,640,512), (0,461,640,512), (512,461,640,512), ...]
            ↓
  Với mỗi tile:
    crop_tile → resize → model inference → decode boxes → scale về hệ tile gốc
            ↓
  merge_tile_detections() → chuyển tất cả boxes về hệ tọa độ ảnh gốc
            ↓
  ╔══════════════════════════════════════════════════╗
  ║  apply_global_nms()  ← BẮT BUỘC                 ║
  ║  Loại bỏ duplicate boxes ở vùng overlap          ║
  ║  (không có bước này → 1 người = 2 box chồng)     ║
  ╚══════════════════════════════════════════════════╝
            ↓
  Output: final boxes trên ảnh gốc 1280×1024
```

---

## 8. Các thông số quan trọng cần tinh chỉnh

### Tile size

| Tile Size | Số tiles/ảnh | Object size tăng | Gợi ý dùng khi |
|-----------|-------------|-----------------|----------------|
| 640×512   | ~4–6 tiles  | 2× so với resize toàn ảnh | Object trung bình |
| 480×384   | ~9–12 tiles | 2.67× | Object nhỏ |
| 320×256   | ~20–25 tiles| 4×    | Object rất nhỏ |

**Gợi ý bắt đầu:** tile 640×512 (2 cột × 2 hàng)

### Overlap

| Overlap | % ảnh được cover thêm | Tránh miss object ở ranh giới |
|---------|----------------------|------------------------------|
| 0.1 (10%) | Nhỏ | Có thể miss |
| 0.2 (20%) | Vừa | Tốt cho hầu hết trường hợp |
| 0.3 (30%) | Lớn | An toàn nhất, nhiều tiles hơn |

**Gợi ý bắt đầu:** overlap = 0.2

### NMS thresholds (sau khi merge tiles)

```python
score_threshold = 0.05   # Thấp hơn bình thường để không bỏ sót
iou_threshold   = 0.45   # Standard NMS
```

### min_visibility (tỷ lệ bbox tối thiểu phải nằm trong tile)

```python
min_visibility = 0.3  # Giữ lại bbox nếu ít nhất 30% diện tích nằm trong tile
```

### Anchor Boxes — CẦN TÍNH LẠI sau khi tiling

> **File thực tế:** `Anchor_box/main.py` — **dòng 30**
>
> Dòng hiện tại:
> ```python
> self._areas = [x ** 2 for x in [2.0, 4.0, 8.0, 10.0, 12.0]]
> # → areas = [4, 16, 64, 100, 144]
> ```
> Đây là giá trị được tinh chỉnh cho ảnh resize 1280→512 (object ~2–12px trong input model).
> Sau khi tiling (crop 640→512), object lớn hơn ~2× → cần areas lớn hơn tương ứng.
>
> **Quy trình:**
> 1. Tạo xong tiled CSV → chạy `Xu_ly_du_lieu/analize_histogram_data.py` trên CSV mới
> 2. Đọc phân phối `w` và `h` của object trong tile (đơn vị pixel trong tile)
> 3. Scale theo `target_size/tile_size` để ra pixel trong input model
> 4. Sửa **dòng 30** với giá trị `_areas` mới phù hợp

Đây là bước **dễ bị quên nhất** nhưng ảnh hưởng lớn đến kết quả.

**Vấn đề:** Anchor box hiện tại trong `Anchor_box/main.py` được căn chỉnh theo kích thước object trong ảnh đã resize toàn bộ (1280→512px). Sau khi tiling, object trong tile **lớn hơn** tương đối → distribution kích thước object thay đổi hoàn toàn.

**Ví dụ cụ thể:**
```
Trước tiling: resize 1280→512, object cao 30px → còn ~12px trong input model
Sau tiling:   crop 640→512,    object cao 30px → còn ~24px trong input model
              (gấp đôi!) → cần anchor area lớn hơn tương ứng
```

**Việc cần làm:**

```
Bước 1: Sau khi tạo xong tile CSV (train_tiled_data.csv),
        chạy lại script phân tích histogram:

        python Xu_ly_du_lieu/analize_histogram_data.py \
               --csv csv_file/train_tiled_data.csv

Bước 2: Nhìn vào phân phối width/height của object trong tile
        → Xác định lại các giá trị _areas phù hợp

Bước 3: Cập nhật DÒNG 30 trong Anchor_box/main.py:
        # Cũ (cho ảnh resize toàn bộ):
        self._areas = [x ** 2 for x in [2.0, 4.0, 8.0, 10.0, 12.0]]

        # Mới (ước tính sau tiling — cần đo thực tế từ histogram):
        self._areas = [x ** 2 for x in [4.0, 8.0, 16.0, 20.0, 24.0]]  # ← ~2x
```

> **Tóm lại:** Tiling thay đổi "kích thước hiệu dụng" của object trong input model. Anchor box không điều chỉnh theo = model sẽ dùng anchor không phù hợp = giảm recall. **Luôn chạy lại histogram analysis trên tiled data trước khi train.**

---

## 9. Lưu ý và rủi ro

### Tăng thời gian preprocessing

- 2,866 ảnh gốc → ~12,000 tiles (estimate)
- Cần chạy preprocessing một lần để tạo tile CSV và lưu ảnh tile

### Tăng thời gian inference

- Trước: 1 forward pass/ảnh
- Sau: 4–6 forward passes/ảnh
- Có thể tối ưu bằng cách batch nhiều tiles của cùng 1 ảnh

### Tiles không có object

- Nhiều tiles sẽ không chứa person nào (background tiles)
- Nên lọc bỏ các tiles này để cân bằng dataset (negative mining)
- Tỷ lệ gợi ý: giữ lại tối đa 3× số tiles có object

### Kiểm tra tính đúng đắn

Sau khi implement, kiểm tra:
1. Vẽ tile lên ảnh gốc → xem tile crops đúng vị trí không
2. Vẽ bbox đã chuyển đổi lên tile → xem có đúng không
3. Chạy tiled inference trên 1 ảnh test → visualize kết quả

### Khi drone bay thấp — object to lên: Tiling có còn dùng được không?

**Câu trả lời: Có vấn đề, nhưng giải quyết được.**

Khi drone bay thấp, người trong ảnh **lớn** (ví dụ 200×300px trong ảnh gốc). Lúc này Tiling gặp rủi ro:

```
Người cao 300px, tile_h = 512px
→ Người chiếm 58% chiều cao tile → VẪN VỪA, không bị cắt ✓

Người cao 600px (drone rất thấp), tile_h = 512px
→ Người BỊ CẮT ĐÔI ngay cả với overlap 20% (102px) ✗
  (phần thân trên ở tile_10, phần chân ở tile_11)
```

**Vùng an toàn của tiling hiện tại (tile 640×512, overlap 20%):**

| Chiều cao người (px, ảnh gốc) | Kết quả |
|-------------------------------|---------|
| < 100px (drone rất cao)       | Quá nhỏ, khó detect dù có tiling |
| 30–200px (drone tầm trung)    | Tiling hiệu quả nhất ✓ |
| 200–410px (drone thấp)        | Vẫn ổn, nằm gọn trong 1 tile ✓ |
| > 410px (drone rất thấp)      | Bị cắt, cần xử lý thêm |

**3 cách xử lý khi gặp object to (drone bay thấp):**

1. **Tăng tile size** — dùng tile 960×768 thay vì 640×512. Object to hơn vẫn lọt vào 1 tile. Nhược điểm: ít tile hơn = ít augmentation = object nhỏ khó detect hơn.

2. **Multi-scale tiling** — kết hợp cả 2: tile nhỏ (640×512) để bắt người xa + tile lớn (1280×1024 = toàn ảnh) để bắt người gần. Inference chạy cả 2 scale rồi gộp NMS.
   ```
   scale 1: 4 tiles 640×512   → bắt người nhỏ (xa)
   scale 2: 1 tile  1280×1024 → bắt người to (gần)
   → merge all boxes → Global NMS
   ```

3. **Adaptive tiling** (phức tạp nhất) — phân tích histogram kích thước object theo từng ảnh, tự động chọn tile size phù hợp.

**Gợi ý cho dataset này:** Dataset drone thermal của bạn chủ yếu là ảnh từ xa (người rất nhỏ). Dùng **Option 2 (Multi-scale)** sẽ bao phủ được cả 2 trường hợp drone cao và thấp mà không cần thay đổi nhiều.

---

### Thứ tự thực hiện gợi ý

```
Bước 1: TẠO FILE MỚI Xu_ly_du_lieu/tiling_utils.py
        → implement generate_tile_coords() và get_bboxes_for_tile()

Bước 2: TEST VISUAL trên 1 ảnh bất kỳ
        → crop tile, vẽ bbox lên tile để xác nhận tọa độ đúng

Bước 3: SỬA Xu_ly_du_lieu/main_step_123.py
        → thêm import tiling_utils, thêm vòng lặp tile, đổi tên CSV output

Bước 4: CHẠY main_step_123.py
        → sinh ra train_tiled_data.csv, valid_tiled_data.csv + thư mục tiles/

Bước 5: CHẠY Xu_ly_du_lieu/analize_histogram_data.py trên CSV tiled
        → đọc phân phối kích thước object → xác định _areas mới

Bước 6: SỬA Anchor_box/main.py dòng 30
        → cập nhật self._areas với giá trị từ bước 5

Bước 7: SỬA train_adam.py (dòng 45, 46, 57, 58, 166, 194)
        → đổi target_size, batch_size, CSV paths, tên file lưu

Bước 8: CHẠY train_adam.py
        → train model mới với tiled data

Bước 9: SỬA predict.py
        → thêm run_tiled_inference(), đổi WEIGHT_PATH

Bước 10: SỬA evaluate_model.py (dòng 7, 52-53, 59-60)
         → đổi sang run_tiled_inference, bỏ nhân ratio GT

Bước 11: ĐÁNH GIÁ và so sánh AP với baseline (không tiling)
```

---

*Tài liệu này được tạo để hướng dẫn implement kỹ thuật Tiling cho dự án RetinaNet phát hiện người trong ảnh drone thermal.*
*Dataset: Drone Thermal.v4i.voc — 2,866 ảnh 1280×1024 — Framework: TensorFlow/Keras*
