# TÀI LIỆU DỰ ÁN: PHÁT HIỆN ĐỐI TƯỢNG TRONG ẢNH NHIỆT DRONE SỬ DỤNG RETINANET

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Bộ dữ liệu Drone Thermal](#3-bộ-dữ-liệu-drone-thermal)
4. [Kiến trúc mô hình](#4-kiến-trúc-mô-hình)
5. [Pipeline tiền xử lý dữ liệu](#5-pipeline-tiền-xử-lý-dữ-liệu)
6. [Pipeline huấn luyện](#6-pipeline-huấn-luyện)
7. [Pipeline suy luận (Inference)](#7-pipeline-suy-luận-inference)
8. [Pipeline đánh giá mô hình](#8-pipeline-đánh-giá-mô-hình)
9. [Siêu tham số & Cấu hình](#9-siêu-tham-số--cấu-hình)
10. [Cách chạy dự án](#10-cách-chạy-dự-án)
11. [Phụ lục: Các vấn đề kỹ thuật](#11-phụ-lục-các-vấn-đề-kỹ-thuật-và-cách-xử-lý)

---

## 1. TỔNG QUAN DỰ ÁN

### Mục tiêu
Xây dựng hệ thống phát hiện người trong ảnh nhiệt (thermal image) chụp từ drone, sử dụng kiến trúc **RetinaNet** kết hợp với kỹ thuật **tiling** để tăng cường khả năng phát hiện đối tượng nhỏ.

### Vấn đề đặt ra
Ảnh drone nhiệt thường có độ phân giải cao (1280×1024), nhưng đối tượng người nhìn từ trên xuống rất nhỏ (chỉ ~15-30 pixel). Nếu resize toàn bộ ảnh về 512×512 để đưa vào mô hình, các đối tượng nhỏ sẽ bị thu nhỏ thêm và rất khó phát hiện.

### Giải pháp
Áp dụng kỹ thuật **tiling** (chia ảnh thành các tile nhỏ có chồng lấp 20%): thay vì resize cả ảnh 1280×1024 → 512×512, ta cắt ra các tile 640×512 từ ảnh gốc, sau đó resize từng tile về 512×512. Điều này giúp các đối tượng người to hơn ~2 lần trong mỗi tile, cải thiện đáng kể khả năng phát hiện.

### Công nghệ sử dụng
| Thành phần | Phiên bản |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.20.0 |
| Keras | 3.12.0 |
| NumPy | 2.2.6 |
| Pandas | mới nhất |
| Matplotlib | 3.10.8 |
| Pillow (PIL) | mới nhất |

---

## 2. CẤU TRÚC THƯ MỤC

```
de_tai_tot_nghiep/
│
├── Anchor_box/                     # Module sinh Anchor Box
│   └── main.py                     # Tạo 9 anchor/vị trí, 5 pyramid level
│
├── FPN/                            # Feature Pyramid Network
│   └── main.py                     # Xây dựng feature map đa tỉ lệ P3-P7
│
├── Label_encode/                   # Bộ mã hóa nhãn cho anchor matching
│   └── main.py                     # Khớp ground truth với anchor, mã hóa delta
│
├── RetinaNet/                      # Kiến trúc RetinaNet chính
│   └── main.py                     # Model + Focal Loss + Smooth L1 Loss
│
├── resnet_50/                      # Backbone ResNet-50
│   └── main.py                     # Tạo feature map C3, C4, C5
│
├── Xu_ly_du_lieu/                  # Module xử lý dữ liệu
│   ├── tiling_utils.py             # Hàm tiling: cắt tile, filter bbox, global NMS
│   ├── main_step_123.py            # Tạo CSV từ XML annotations + tiling
│   ├── chia_data.py                # Chia train/val/test 70/20/10
│   └── preprocessing_data_before_training.py  # Pipeline đọc ảnh cho training
│
├── Drone Thermal.v4i.voc/          # Thư mục chứa dataset và file chạy chính
│   ├── train/                      # Ảnh + XML annotations (train)
│   ├── valid/                      # Ảnh + XML annotations (valid)
│   ├── test/                       # Ảnh + XML annotations (test)
│   ├── tile_data/                  # Ảnh tile đã được cắt ra
│   ├── csv_file/                   # Các file CSV
│   │   ├── data_information_tiled.csv  # Toàn bộ ~18,000 tile samples
│   │   ├── train_data.csv          # ~12,600 tile samples
│   │   ├── valid_data.csv          # ~3,600 tile samples
│   │   └── test_data.csv           # ~1,800 tile samples
│   ├── weight_store/               # Lưu weights model
│   │   ├── retinanet_tiled_512.weights.h5  # Weights tốt nhất (695 MB)
│   │   ├── 512_finetune.weights.h5
│   │   └── 512_.weights.h5
│   ├── predict_drone.py            # Inference cho từng ảnh (dùng tiling)
│   ├── evaluate_drone_model.py     # Tính AP, vẽ đường cong Precision-Recall
│   └── result/                     # Kết quả đánh giá (CSV, PNG)
│
├── train_adam.py                   # Script huấn luyện chính (Adam + tiling)
├── train_tiep.py                   # Tiếp tục huấn luyện từ checkpoint
├── fine_tune_adam.py               # Fine-tuning model
├── predict.py                      # Inference + trực quan hóa kết quả
├── evaluate_model.py               # Đánh giá trên tập test
├── plot_history.py                 # Vẽ biểu đồ loss/metric theo epoch
└── check.py                        # Tiện ích kiểm tra
```

---

## 3. BỘ DỮ LIỆU DRONE THERMAL

### Thông tin chung
- **Nguồn:** Roboflow — Drone Thermal v4i (định dạng Pascal VOC)
- **Tổng số ảnh gốc:** 2,866 ảnh
- **Kích thước ảnh gốc:** 1280×1024 pixel
- **Số lớp:** 2 (0: background, 1: person)
- **Định dạng annotation:** XML (Pascal VOC) — tọa độ `[xmin, ymin, xmax, ymax]`

### Phân chia dữ liệu gốc
| Tập | Số ảnh gốc | Tỉ lệ |
|---|---|---|
| Train | ~2,006 | 70% |
| Valid | ~573 | 20% |
| Test | ~287 | 10% |

### Sau khi tiling
| Tập | Số tile samples | Ghi chú |
|---|---|---|
| Train | ~12,600 | Từ ~2,006 ảnh gốc |
| Valid | ~3,600 | Từ ~573 ảnh gốc |
| Test | ~1,800 | Từ ~287 ảnh gốc |
| **Tổng** | **~18,000** | |

### Cấu trúc file CSV
Mỗi dòng trong CSV tương ứng với **một bounding box** trong một tile:

```
path_img, xmin, ymin, xmax, ymax, class_id
```

Ví dụ:
```
/home/huy/.../tile_data/train_img001_tile_0.jpg, 120, 45, 180, 110, 1
/home/huy/.../tile_data/train_img001_tile_0.jpg, 300, 200, 360, 280, 1
```

---

## 4. KIẾN TRÚC MÔ HÌNH

### 4.1 Tổng quan kiến trúc RetinaNet

```
Ảnh đầu vào (512×512×3)
         │
    ┌────▼────┐
    │ResNet-50│  <- Backbone
    └────┬────┘
         │ [C3, C4, C5]
    ┌────▼────┐
    │   FPN   │  <- Feature Pyramid Network
    └────┬────┘
         │ [P3, P4, P5, P6, P7]
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──┐
│Class  │ │ Box │  <- Hai prediction head
│ Head  │ │Head │
└───┬───┘ └──┬──┘
    └────┬───┘
    Concatenate -> (batch, num_anchors, num_classes + 4)
```

### 4.2 ResNet-50 Backbone

**File:** `resnet_50/main.py`

Backbone trích xuất đặc trưng từ ảnh đầu vào, trả về 3 feature map ở các stride khác nhau:

| Output | Stride | Kích thước (với input 512×512) |
|---|---|---|
| C3 | 8 | 64×64 |
| C4 | 16 | 32×32 |
| C5 | 32 | 16×16 |

**Cấu trúc:**
- Lớp đầu: Conv 7×7 + MaxPool (stride=2)
- 4 stage Residual blocks: 3 + 4 + 6 + 3 bottleneck blocks
- Mỗi Bottleneck block: Conv 1×1 → Conv 3×3 → Conv 1×1 + Skip connection

**Hai loại block:**
- `Conv_block`: Có stride, thay đổi kích thước spatial
- `Res_id_block`: Stride=1, giữ nguyên kích thước spatial

### 4.3 Feature Pyramid Network (FPN)

**File:** `FPN/main.py`

FPN tạo ra các feature map đa tỉ lệ từ backbone, giúp phát hiện đối tượng ở nhiều kích thước khác nhau.

```
C5 --1x1--> P5
             |
C4 --1x1-->(+)--Upsample--> P4
             |
C3 --1x1-->(+)--Upsample--> P3
             |
            P5--stride-2--> P6--stride-2--> P7
```

| Feature Map | Stride | Kích thước (512×512) | Mục đích |
|---|---|---|---|
| P3 | 8 | 64×64 | Phát hiện đối tượng nhỏ |
| P4 | 16 | 32×32 | Phát hiện đối tượng vừa |
| P5 | 32 | 16×16 | Phát hiện đối tượng lớn |
| P6 | 64 | 8×8 | Phát hiện đối tượng rất lớn |
| P7 | 128 | 4×4 | Phát hiện đối tượng cực lớn |

Tất cả P3-P7 đều có 256 channels sau các lớp Conv 1×1.

### 4.4 Anchor Boxes

**File:** `Anchor_box/main.py`

Mỗi vị trí trên mỗi feature map sinh ra **9 anchor boxes** (3 tỉ lệ × 3 scale).

**Cấu hình anchor:**
```python
aspect_ratios = [0.4, 0.7, 1.1]          # 3 tỉ lệ chiều cao/chiều rộng
scales = [2^0, 2^(1/3), 2^(2/3)]         # 3 scale: 1.0, 1.26, 1.587
areas = [12^2, 24^2, 48^2, 96^2, 192^2]  # Diện tích cơ bản cho P3->P7
```

**Công thức sinh anchor:**
```
w = sqrt(area / ratio)
h = area / w
w_scaled = w * scale
h_scaled = h * scale
```

**Tổng số anchor** với ảnh 512×512: ~49,104 anchors (tổng hợp từ P3 đến P7).

### 4.5 Classification Head và Box Regression Head

Cả hai head đều có cùng cấu trúc nhưng output khác nhau:

```
Input: Feature map P_i (H_i x W_i x 256)
  -> Conv 3x3 (256) + ReLU  x4
  -> Conv 3x3 (9 x num_classes)   <- Classification Head
  hoac
  -> Conv 3x3 (9 x 4)             <- Box Regression Head
```

- **Classification Head output:** Logits cho từng lớp, sau đó dùng sigmoid
- **Box Regression Head output:** 4 delta `[Δx, Δy, Δw, Δh]` tương đối so với anchor

### 4.6 Hàm mất mát (Loss Functions)

**Focal Loss** (cho Classification):
```
FL(pt) = -alpha x (1 - pt)^gamma x log(pt)

alpha = 0.25   (giảm trọng số mẫu âm dễ)
gamma = 2.0    (tập trung vào mẫu khó)
```
Focal Loss giải quyết bất cân bằng lớp nghiêm trọng (hàng nghìn anchor âm vs. vài chục anchor dương).

**Smooth L1 Loss** (cho Box Regression):
```
L(x) = 0.5 x x^2       nếu |x| < delta (delta = 1.0)
       |x| - 0.5        nếu |x| >= delta
```
Smooth L1 ổn định hơn L2 khi có outlier lớn.

**Tổng loss:**
```
Loss = (Focal Loss) + (Smooth L1 Loss)
       Cả hai đều normalize theo số lượng anchor dương
```

### 4.7 Label Encoder

**File:** `Label_encode/main.py`

Khớp ground truth với anchor và tạo target để huấn luyện:

| IoU với GT tốt nhất | Nhãn anchor |
|---|---|
| >= 0.4 | **Positive** — học regression và classification |
| 0.2 -> 0.4 | **Ignore** — không tính vào loss |
| < 0.2 | **Negative** — chỉ học classification = 0 |

**Mã hóa box delta (có variance):**
```
Δx = (GT_cx - anchor_cx) / anchor_w / 0.1
Δy = (GT_cy - anchor_cy) / anchor_h / 0.1
Δw = log(GT_w / anchor_w) / 0.2
Δh = log(GT_h / anchor_h) / 0.2
```

---

## 5. PIPELINE TIỀN XỬ LÝ DỮ LIỆU

### 5.1 Tạo dữ liệu tiled (Bước 1)

**File:** `Xu_ly_du_lieu/main_step_123.py`

```
Ảnh gốc (1280×1024) + XML annotations
            |
    Parse XML -> [xmin, ymin, xmax, ymax, class]
            |
    generate_tile_coords() -> [(x0,y0,640,512), ...]
                               Với overlap 20%: ~6 tile/ảnh
            |
    get_bboxes_for_tile() -> Lọc bbox có >=30% diện tích trong tile
                              Chuyển về tọa độ tile-local
            |
    crop_and_save_tile() -> Lưu tile ra disk (JPEG quality=95)
            |
    Ghi vào data_information_tiled.csv
```

### 5.2 Chia train/val/test (Bước 2)

**File:** `Xu_ly_du_lieu/chia_data.py`

```
data_information_tiled.csv
        |
unique_images = danh sách file ảnh gốc
        |
shuffle -> split 70/20/10
        |
train_data.csv | valid_data.csv | test_data.csv
```

**Lưu ý quan trọng:** Split theo ảnh gốc, không phải theo tile — đảm bảo tiles từ cùng một ảnh gốc không bị rải sang cả train lẫn test (tránh data leakage).

### 5.3 Pipeline đọc dữ liệu khi training

**File:** `Xu_ly_du_lieu/preprocessing_data_before_training.py`

```
CSV file
  |
  read_img_and_label(path, bbox, class_id)
      -> Đọc JPEG, cast to float32
  |
  resize_and_pad_img(img, bbox, class_id, target=512)
      -> ratio = 512 / max(H, W)
      -> Resize giữ tỉ lệ
      -> Pad đến 512×512
      -> Scale bbox theo ratio
  |
  pack_targets(img, bbox, cls)
      -> LabelEncoder.encode_batch() -> target_boxes, target_classes
  |
  tf.data.Dataset
      -> from_generator()
      -> map(read_img_and_label, num_parallel_calls=AUTOTUNE)
      -> map(resize_and_pad_img)
      -> map(pack_targets)
      -> padded_batch(8)
      -> prefetch(buffer_size=2)
```

---

## 6. PIPELINE HUẤN LUYỆN

### 6.1 Cấu hình huấn luyện

**File:** `train_adam.py`

```python
target_size = 512
batch_size = 8
optimizer = Adam(learning_rate=5e-5)
epochs = 200
mixed_precision = 'mixed_float16'    # GPU inference ở float16, loss ở float32
```

### 6.2 Khởi tạo model

```python
backbone = resnet_50_backbone()
model = RetinaNet(num_classes=1, backbone=backbone)  # 1 lớp: person
model.compile(optimizer=Adam(5e-5))
```

### 6.3 Callbacks

| Callback | Cấu hình | Mục đích |
|---|---|---|
| `ReduceLROnPlateau` | factor=0.5, patience=5 | Giảm LR khi loss không cải thiện |
| `ModelCheckpoint` | monitor=val_loss | Lưu weights tốt nhất |
| `EarlyStopping` | patience=10 | Dừng sớm tránh overfitting |

### 6.4 Mixed Precision Training

Model được cấu hình sử dụng `mixed_float16`:
- **Tính toán forward/backward:** float16 — nhanh hơn trên GPU, ít VRAM hơn (~50%)
- **Weights và loss:** float32 — đảm bảo độ chính xác
- **Lưu ý:** Cần cast tensor về float32 trước khi đưa vào `tf.image.combined_non_max_suppression`

### 6.5 Vị trí lưu weights

```
Drone Thermal.v4i.voc/weight_store/
├── retinanet_tiled_512.weights.h5   <- Model tốt nhất (~695 MB)
├── 512_finetune.weights.h5
└── 512_.weights.h5
```

---

## 7. PIPELINE SUY LUẬN (INFERENCE)

### 7.1 Inference một ảnh đơn

**File:** `Drone Thermal.v4i.voc/predict_drone.py` — hàm `run_inference()`

```
Ảnh gốc (bất kỳ kích thước)
        |
Resize & Pad về 512×512 (giữ tỉ lệ)
        |
Model forward pass
  predictions: (1, num_anchors, 5)
        |
box_predictions = predictions[:, :, :4]
class_predictions = predictions[:, :, 4:]
        |
decode_box_predictions(anchors, box_predictions)
  -> tọa độ tuyệt đối [ymin, xmin, ymax, xmax]
        |
class_probs = sigmoid(class_predictions)
        |
Cast về float32 (do mixed precision)
        |
combined_non_max_suppression(
    boxes, scores,
    max_output_size_per_class=10,
    iou_threshold=0.4,
    score_threshold=0.4
)
        |
Output: nms_boxes, nms_scores, nms_classes, valid_detections
```

### 7.2 Giải mã tọa độ box (Box Decoding)

Inverse của quá trình mã hóa delta khi training:

```python
# Nhân ngược lại variance
t_x = t_x * 0.1;  t_y = t_y * 0.1
t_w = t_w * 0.2;  t_h = t_h * 0.2

# Giải mã center
x = t_x * anchor_w + anchor_cx
y = t_y * anchor_h + anchor_cy

# Giải mã kích thước
w = exp(t_w) * anchor_w
h = exp(t_h) * anchor_h

# Chuyển về [ymin, xmin, ymax, xmax] (format TF NMS)
ymin = y - h/2;  xmin = x - w/2
ymax = y + h/2;  xmax = x + w/2
```

### 7.3 Tiled Inference (cho ảnh lớn 1280×1024)

**File:** `predict.py` — hàm `run_tiled_inference()`

```
Ảnh gốc (1280×1024)
        |
generate_tile_coords(overlap=0.2)
  -> [(x0,y0,640,512), (x1,y0,640,512), ...]  (~6 tiles)
        |
For each tile:
    Crop tile từ ảnh gốc
    Resize về 512×512
    run_inference() -> boxes trong tile coordinates
    Scale boxes về ảnh gốc coordinates
    Per-tile NMS (iou=0.4, score=0.05)
        |
Gộp tất cả boxes từ các tiles
        |
apply_global_nms(iou=0.45, score=0.05)
  -> Loại bỏ duplicate detections ở vùng chồng lấp
        |
Output: Final boxes trong ảnh gốc coordinates
```

---

## 8. PIPELINE ĐÁNH GIÁ MÔ HÌNH

### 8.1 Tính Average Precision (AP)

**File:** `Drone Thermal.v4i.voc/evaluate_drone_model.py`

**Thuật toán:**
```
1. Load test_data.csv
2. Load model với trained weights
3. For each test image:
   a. GT boxes = parse từ CSV (chuyển [x_c, y_c, w, h] -> [xmin, ymin, xmax, ymax])
   b. Predictions = run_tiled_inference(score_threshold=0.05)
   c. Với mỗi prediction (sắp xếp theo score giảm dần):
      - Tính IoU với tất cả GT boxes chưa được match
      - Nếu IoU >= 0.5 và GT chưa match: TP
      - Ngược lại: FP

4. Precision = TP / (TP + FP)
5. Recall = TP / total_GT

6. Average Precision = Area Under PR Curve
```

### 8.2 Output đánh giá

```
Drone Thermal.v4i.voc/result/
├── recall_precision.csv    <- Các cặp (Recall, Precision) trên toàn test set
└── pr_curve.png            <- Biểu đồ đường cong Precision-Recall
```

Console output:
```
AP = XX.XX%
Total GT boxes: XXXX
Total predictions: XXXX
```

---

## 9. SIÊU THAM SỐ & CẤU HÌNH

### Tham số kiến trúc

| Thành phần | Tham số | Giá trị |
|---|---|---|
| Input | Kích thước | 512×512 |
| FPN | Channels | 256 |
| Anchor | Số anchor/vị trí | 9 |
| Anchor | Aspect ratios | [0.4, 0.7, 1.1] |
| Anchor | Scales | [1.0, 1.26, 1.587] |
| Anchor | Base areas (P3->P7) | [144, 576, 2304, 9216, 36864] |

### Tham số huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer | Adam |
| Learning rate ban đầu | 5e-5 |
| Batch size | 8 |
| Epochs tối đa | 200 |
| Mixed precision | mixed_float16 |
| LR reduce factor | 0.5 (khi val_loss không giảm 5 epoch) |
| Early stopping patience | 10 epochs |

### Tham số Label Encoding

| Tham số | Giá trị |
|---|---|
| Positive IoU threshold | 0.4 |
| Ignore IoU threshold | 0.2 |
| Box variances | [0.1, 0.1, 0.2, 0.2] |

### Tham số Focal Loss

| Tham số | Giá trị |
|---|---|
| Alpha (α) | 0.25 |
| Gamma (γ) | 2.0 |

### Tham số Tiling

| Tham số | Giá trị |
|---|---|
| Tile width (tọa độ gốc) | 640 px |
| Tile height (tọa độ gốc) | 512 px |
| Overlap | 20% |
| Min bbox overlap để giữ | 30% |

### Tham số NMS

| Ngữ cảnh | IoU threshold | Score threshold |
|---|---|---|
| Per-tile NMS (training eval) | 0.4 | 0.4 |
| Per-tile NMS (evaluation) | 0.4 | 0.05 |
| Global NMS | 0.45 | 0.05 |

---

## 10. CÁCH CHẠY DỰ ÁN

### Bước 0: Cài đặt môi trường

```bash
cd /home/huy/Documents/de_tai_tot_nghiep
python -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.20.0 pandas numpy matplotlib pillow
```

### Bước 1: Tạo dữ liệu tiled

```bash
python Xu_ly_du_lieu/main_step_123.py
# Output: Drone Thermal.v4i.voc/csv_file/data_information_tiled.csv
#         Drone Thermal.v4i.voc/tile_data/*.jpg
```

### Bước 2: Chia train/val/test

```bash
python Xu_ly_du_lieu/chia_data.py
# Output: train_data.csv, valid_data.csv, test_data.csv
```

### Bước 3: Huấn luyện mô hình

```bash
python train_adam.py
# Weights lưu tại: Drone Thermal.v4i.voc/weight_store/retinanet_tiled_512.weights.h5
```

### Bước 4: Đánh giá mô hình

```bash
cd "Drone Thermal.v4i.voc"
python evaluate_drone_model.py
# Output: result/recall_precision.csv, result/pr_curve.png
# Console: AP = XX.XX%
```

### Bước 5: Chạy inference trên ảnh mới

```bash
python predict.py
# Chỉnh sửa image_path trong script để dự đoán ảnh mới
```

### Tiếp tục training từ checkpoint

```bash
python train_tiep.py
```

### Fine-tuning

```bash
python fine_tune_adam.py
```

---

## 11. PHỤ LỤC: CÁC VẤN ĐỀ KỸ THUẬT VÀ CÁCH XỬ LÝ

### Mixed Precision và NMS

**Vấn đề:** `tf.image.combined_non_max_suppression` yêu cầu tensor float32, nhưng mixed_float16 khiến model output ra float16.

**Giải pháp:**
```python
nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
    tf.cast(tf.expand_dims(decoded_boxes, axis=2), tf.float32),  # cast về float32
    tf.cast(class_probs, tf.float32),                            # cast về float32
    ...
)
```

### Xử lý đối tượng nhỏ trong ảnh drone

**Vấn đề:** Người nhìn từ drone chỉ ~15-30 pixel, resize ảnh toàn bộ làm đối tượng bị quá nhỏ.

**Giải pháp:** Kỹ thuật tiling — cắt ảnh thành tile nhỏ có overlap 20%, mỗi tile chứa đối tượng to hơn ~2 lần. Sau inference, dùng global NMS để loại bỏ detection trùng lặp ở vùng chồng lấp.

### Bất cân bằng lớp (Class Imbalance)

**Vấn đề:** Hàng chục nghìn anchor âm (background) so với vài chục anchor dương (person).

**Giải pháp:** Focal Loss với α=0.25, γ=2.0 — giảm đáng kể contribution của các mẫu âm dễ vào loss, buộc model tập trung vào các mẫu khó.

### Tọa độ box sau NMS

**Lưu ý:** TensorFlow `combined_non_max_suppression` sử dụng format `[ymin, xmin, ymax, xmax]` (y-first), khác với Pascal VOC format `[xmin, ymin, xmax, ymax]`. Cần chuyển đổi khi vẽ bounding box hoặc tính IoU.
