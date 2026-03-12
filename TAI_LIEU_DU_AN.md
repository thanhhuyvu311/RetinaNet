# TÀI LIỆU DỰ ÁN: HỆ THỐNG NHẬN DIỆN VẬT THỂ SỬ DỤNG RETINANET

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Mục đích
Dự án xây dựng hệ thống nhận diện và phát hiện đa vật thể (Multi-Object Detection) trong ảnh sử dụng mô hình RetinaNet với backbone ResNet-50.

### 1.2. Công nghệ sử dụng
- **Framework**: TensorFlow 2.20.0, Keras 3.12.0
- **Ngôn ngữ**: Python 3.10
- **Thư viện hỗ trợ**: NumPy 2.2.6, Pandas, Matplotlib 3.10.8
- **Kiến trúc mô hình**: RetinaNet + ResNet-50 + FPN (Feature Pyramid Network)

### 1.3. Cấu trúc dự án
```
de_tai_tot_nghiep/
├── Anchor_box/          # Module tạo anchor boxes
├── classification/       # Module phân loại ảnh
├── FPN/                 # Feature Pyramid Network
├── Label_encode/        # Module mã hóa nhãn
├── object_detect/       # Dữ liệu và kết quả huấn luyện
├── resnet_50/           # Backbone ResNet-50
├── RetinaNet/           # Kiến trúc RetinaNet
├── Xu_ly_du_lieu/       # Tiền xử lý dữ liệu
├── train.py             # Script huấn luyện chính
├── predict.py           # Script dự đoán
└── evaluate_model.py    # Đánh giá mô hình
```

---

## 2. CÁC MODULE CHÍNH VÀ GIẢI THÍCH CHI TIẾT

### 2.1. MODULE ANCHOR_BOX (`Anchor_box/main.py`)

#### Mục đích
Tạo các anchor boxes (hộp neo) tại mọi vị trí trên feature maps để làm cơ sở cho việc dự đoán bounding boxes.

#### Class: `Anchor_box`

**Thuộc tính:**
- `aspect_ratios = [0.4, 0.5, 1.0]`: Tỷ lệ khung hình của anchor boxes
- `scales = [2^0, 2^(1/3), 2^(2/3)]`: Các tỷ lệ scale cho mỗi anchor
- `_num_anchors = 9`: Số lượng anchor tại mỗi vị trí (3 ratios × 3 scales)
- `_strides = [8, 16, 32, 64, 128]`: Bước nhảy cho các feature maps từ P3 đến P7
- `_areas = [12², 24², 48², 96², 128²]`: Diện tích anchor cho mỗi pyramid level

**Các hàm chính:**

1. **`_compute_dims(self)`**
   - **Chức năng**: Tính toán kích thước (width, height) của tất cả anchor boxes
   - **Input**: Không có (sử dụng thuộc tính của class)
   - **Output**: List các tensor chứa dimensions của anchors cho mỗi pyramid level
   - **Công thức**:
     ```
     anchor_h = sqrt(area / ratio)
     anchor_w = area / anchor_h
     ```

2. **`_get_anchors(self, feature_h, feature_w, level)`**
   - **Chức năng**: Tạo anchor boxes cho một feature map cụ thể
   - **Input**:
     - `feature_h`: Chiều cao của feature map
     - `feature_w`: Chiều rộng của feature map
     - `level`: Level của pyramid (3-7)
   - **Output**: Tensor shape `(feature_h × feature_w × 9, 4)` chứa tọa độ `[x_center, y_center, width, height]`
   - **Quy trình**:
     - Tạo lưới tọa độ bằng `meshgrid`
     - Nhân với stride để map về ảnh gốc
     - Gán dimensions đã tính trước cho mỗi anchor

3. **`get_anchors(self, img_h, img_w)`**
   - **Chức năng**: Tạo TẤT CẢ anchor boxes cho toàn bộ feature pyramid
   - **Input**: Chiều cao và rộng của ảnh
   - **Output**: Tensor chứa tất cả anchors từ P3-P7 được concat lại
   - **Ví dụ**: Với ảnh 224×224, tạo ra khoảng 4,995 anchors

---

### 2.2. MODULE LABEL_ENCODE (`Label_encode/main.py`)

#### Mục đích
Gán nhãn cho anchor boxes dựa trên Ground Truth, tính toán target boxes và classes.

#### Class: `LabelEncoder`

**Thuộc tính:**
- `_match_iou = 0.4`: Ngưỡng IoU để coi anchor là positive
- `_ignore_iou = 0.2`: Ngưỡng IoU để bỏ qua anchor
- `_box_variance = [0.1, 0.1, 0.2, 0.2]`: Hệ số chuẩn hóa cho regression targets

**Các hàm chính:**

1. **`_compute_box_target(self, anchor_boxes, matched_gt_boxes)`**
   - **Chức năng**: Tính delta (độ lệch) giữa anchor và ground truth box
   - **Input**:
     - `anchor_boxes`: Tensor anchors `[x, y, w, h]`
     - `matched_gt_boxes`: Ground truth boxes tương ứng
   - **Output**: Target boxes đã được mã hóa
   - **Công thức encoding**:
     ```python
     t_x = (gt_x - anchor_x) / anchor_w / variance_x
     t_y = (gt_y - anchor_y) / anchor_h / variance_y
     t_w = log(gt_w / anchor_w) / variance_w
     t_h = log(gt_h / anchor_h) / variance_h
     ```

2. **`_encode_sample(self, gt_boxes, gt_classes, anchor_boxes)`**
   - **Chức năng**: Gán nhãn cho một ảnh duy nhất
   - **Input**: Ground truth boxes, classes và tất cả anchors
   - **Output**: `target_boxes`, `target_classes`
   - **Quy trình**:
     - Tính IoU matrix giữa GT boxes và anchors
     - Match mỗi anchor với GT box có IoU cao nhất
     - Phân loại anchors:
       - `Positive (class_id ≥ 0)`: IoU ≥ 0.4
       - `Negative (class_id = -1)`: IoU < 0.2
       - `Ignore (class_id = -2)`: 0.2 ≤ IoU < 0.4

3. **`encode_batch(self, batch_images, gt_boxes, gt_classes, anchor_boxes)`**
   - **Chức năng**: Encode cả batch ảnh
   - **Input**: Batch ảnh, boxes, classes, anchors
   - **Output**: Batch targets
   - **Sử dụng**: `tf.map_fn` để xử lý song song

---

### 2.3. MODULE RESNET_50 (`resnet_50/main.py`)

#### Mục đích
Xây dựng backbone ResNet-50 để trích xuất đặc trưng từ ảnh.

#### Các hàm chính:

1. **`Conv_block(x, filter, stride)`**
   - **Chức năng**: Bottleneck block có thay đổi kích thước
   - **Input**:
     - `x`: Input tensor
     - `filter`: Tuple `(f1, f2)` - số filters
     - `stride`: Bước nhảy (để giảm kích thước)
   - **Output**: Feature map sau khi qua block
   - **Cấu trúc**:
     ```
     Conv 1×1 (f1, stride) → BN → ReLU
     → Conv 3×3 (f1) → BN → ReLU
     → Conv 1×1 (f2) → BN
     → Add với skip connection → ReLU
     ```

2. **`Res_id_block(x, filter)`**
   - **Chức năng**: Identity block (không đổi kích thước)
   - **Tương tự Conv_block** nhưng stride=1 và skip connection không cần conv

3. **`resnet_50_backbone()`**
   - **Chức năng**: Xây dựng toàn bộ ResNet-50
   - **Input**: Ảnh shape `(None, None, 3)`
   - **Output**: 3 feature maps `[C3, C4, C5]`
   - **Kiến trúc**:
     ```
     Input (224×224×3)
     → Conv 7×7, stride 2 (112×112×64)
     → MaxPool 3×3, stride 2 (56×56×64)
     → Stage 2: Conv_block + 2×Identity (56×56×256)    [C3 output]
     → Stage 3: Conv_block + 3×Identity (28×28×512)    [C4 output]
     → Stage 4: Conv_block + 5×Identity (14×14×1024)   [C5 output]
     → Stage 5: Conv_block + 2×Identity (7×7×2048)
     ```

4. **`head_of_model(output_filters, bias_init)`**
   - **Chức năng**: Tạo head network cho classification hoặc box regression
   - **Input**: Số filters output và cách khởi tạo bias
   - **Output**: Sequential model
   - **Cấu trúc**: 4 lớp Conv 3×3 (256 filters) + 1 lớp Conv 3×3 output

---

### 2.4. MODULE FPN (`FPN/main.py`)

#### Mục đích
Xây dựng Feature Pyramid Network để tạo multi-scale feature maps.

#### Class: `FeaturePyramid`

**Các thành phần:**
- 3 Conv 1×1: Đồng bộ số kênh C3, C4, C5 về 256
- 5 Conv 3×3: Làm mịn sau upsampling cho P3-P7
- 2 Conv 3×3 stride 2: Tạo P6, P7 từ C5
- UpSampling2D: Phóng to feature map gấp đôi

**Hàm `call(self, images, training=False)`**
- **Chức năng**: Forward pass qua FPN
- **Input**: Batch ảnh
- **Output**: 5 feature maps `[P3, P4, P5, P6, P7]`
- **Quy trình**:
  ```
  1. C3, C4, C5 ← ResNet50(images)
  2. P5 ← Conv1×1(C5)
  3. P4 ← Conv1×1(C4) + Upsample(P5)
  4. P3 ← Conv1×1(C3) + Upsample(P4)
  5. P3-P5 ← Conv3×3(P3-P5)  # Làm mịn
  6. P6 ← Conv3×3_stride2(C5)
  7. P7 ← Conv3×3_stride2(ReLU(P6))
  ```

---

### 2.5. MODULE RETINANET (`RetinaNet/main.py`)

#### Mục đích
Kiến trúc chính của mô hình nhận diện vật thể.

#### Class: `RetinaNet`

**Hàm `__init__(self, num_classes, backbone)`**
- Khởi tạo FPN với backbone
- Tạo 2 head networks:
  - `cls_head`: Dự đoán classes (9 × num_classes outputs)
  - `box_head`: Dự đoán box deltas (9 × 4 outputs)

**Hàm `call(self, image, training=False)`**
- **Input**: Batch ảnh
- **Output**: Tensor `(batch, num_anchors, 4 + num_classes)`
- **Quy trình**:
  ```
  1. features ← FPN(image)  # [P3, P4, P5, P6, P7]
  2. Với mỗi feature:
     - box_pred ← box_head(feature)
     - cls_pred ← cls_head(feature)
  3. Concat tất cả predictions
  4. Return [box_outputs | cls_outputs]
  ```

#### Class: `RetinaNetBoxLoss`
- **Mục đích**: Tính Smooth L1 Loss cho box regression
- **Công thức**:
  ```python
  if |y_true - y_pred| < delta:
      loss = 0.5 × (y_true - y_pred)²
  else:
      loss = |y_true - y_pred| - 0.5
  ```

#### Class: `RetinaNetClassificationLoss`
- **Mục đích**: Tính Focal Loss để xử lý class imbalance
- **Công thức**:
  ```python
  FL = -α × (1 - p_t)^γ × log(p_t)
  ```
  - `α = 0.25`: Trọng số cho positive class
  - `γ = 2.0`: Focusing parameter

#### Class: `RetinaNetLoss`
- **Kết hợp cả 2 loss trên**
- **Normalization**: Chia cho số positive anchors để ổn định gradient
- **Masking**: Bỏ qua các anchor có label = -2

---

### 2.6. MODULE XỬ LÝ DỮ LIỆU (`Xu_ly_du_lieu/preprocessing_data_before_training.py`)

#### Các hàm chính:

1. **`change_string2number(input_csv)`**
   - **Chức năng**: Chuyển đổi string trong CSV thành list Python
   - **Input**: Đường dẫn file CSV
   - **Output**: DataFrame với bbox và class_id là list
   - **Sử dụng**: `ast.literal_eval()`

2. **`create_dataset_from_dataframe(data_frame)`**
   - **Chức năng**: Tạo tf.data.Dataset từ DataFrame
   - **Output**: Dataset với `(path, bbox, class_id)`
   - **Xử lý class_id**: Trừ 1 để chuyển về 0-indexed
   - **Sử dụng**: `tf.ragged.constant` cho dữ liệu có độ dài khác nhau

3. **`read_img_and_label(path, bbox, class_id)`**
   - **Chức năng**: Đọc ảnh từ đường dẫn
   - **Quy trình**:
     - `tf.io.read_file()`: Đọc file
     - `tf.image.decode_jpeg()`: Giải mã JPEG
     - `tf.cast()`: Chuyển sang float32

4. **`resize_and_pad_img(img, bbox, class_id, target_size)`**
   - **Chức năng**: Resize ảnh về 224×224 và giữ tỷ lệ khung hình
   - **Input**: Ảnh, bbox, class_id, target_size (224)
   - **Output**: Ảnh đã resize + pad, bbox đã scale
   - **Quy trình**:
     ```
     1. ratio = target_size / max(h, w)
     2. new_h, new_w = h × ratio, w × ratio
     3. Resize ảnh về (new_h, new_w)
     4. Pad thành (224, 224) bằng pixel đen
     5. bbox_scaled = bbox × ratio
     ```

5. **`visualize_data(img_tensor, bbox_tensor, ax, title)`**
   - **Chức năng**: Vẽ ảnh kèm bounding boxes
   - **Sử dụng**: Matplotlib để debug và kiểm tra dữ liệu

---

### 2.7. MODULE TÍNH IOU (`Anchor_box/iou.py`)

#### Các hàm:

1. **`convert_to_corners(boxes)`**
   - **Chức năng**: Chuyển đổi format bbox
   - **Input**: `[x_center, y_center, w, h]`
   - **Output**: `[x_min, y_min, x_max, y_max]`

2. **`compute_iou(truth_box, predict_box)`**
   - **Chức năng**: Tính IoU (Intersection over Union) giữa 2 tập boxes
   - **Input**:
     - `truth_box`: Shape `(M, 4)`
     - `predict_box`: Shape `(N, 4)`
   - **Output**: IoU matrix shape `(M, N)`
   - **Công thức**:
     ```
     IoU = Area_Intersection / Area_Union
     Area_Union = Area_GT + Area_Pred - Area_Intersection
     ```

---

## 3. SCRIPT HUẤN LUYỆN (`train.py`)

### 3.1. Quy trình huấn luyện

```python
# 1. KHỞI TẠO ANCHOR BOXES
target_size = 224
anchor_gene = Anchor_box()
all_anchor = anchor_gene.get_anchors(img_h=224, img_w=224)
label_encoder = LabelEncoder()

# 2. TẠO DATASET
train_dataset = preprocessing_data_before_training.create_dataset_from_dataframe(raw_train_data)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(read_img_and_label)
train_dataset = train_dataset.map(resize_and_pad_img)

# 3. GÁN NHÃN VÀ TẠO BATCH
batch_size = 32
train_dataset = train_dataset.padded_batch(
    batch_size=32,
    padding_values=(0.0, 1e-8, -2)  # Pad bbox=1e-8, class=-2
)
train_dataset = train_dataset.map(pack_targets)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# 4. KHỞI TẠO MÔ HÌNH
resnet50_backbone = resnet_50_backbone()
model = RetinaNet(num_classes=num_classes, backbone=resnet50_backbone)

# 5. CẤU HÌNH OPTIMIZER
learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries,
    values=learning_rates
)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

# 6. COMPILE VÀ TRAIN
model.compile(optimizer=optimizer, loss=RetinaNetLoss(num_classes=num_classes))
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[ModelCheckpoint, EarlyStopping]
)
```

### 3.2. Hàm `pack_targets`
- **Chức năng**: Gộp target_boxes và target_classes thành 1 tensor
- **Output**: `(img, y_true)` với `y_true` shape `(batch, num_anchors, 5)`
  - 4 cột đầu: box deltas
  - Cột cuối: class label

### 3.3. Callbacks
- **ModelCheckpoint**: Lưu weights tốt nhất vào file `.weights.h5`
- **EarlyStopping**: Dừng khi loss không giảm sau 15 epochs

---

## 4. SCRIPT DỰ ĐOÁN (`predict.py`)

### 4.1. Các hàm chính

1. **`decode_box_predictions(anchor_boxes, box_predictions)`**
   - **Chức năng**: Giải mã box deltas thành tọa độ thực
   - **Quy trình**:
     ```python
     # Nhân ngược variance
     t_x, t_y, t_w, t_h = box_predictions * [0.1, 0.1, 0.2, 0.2]

     # Giải mã
     x = t_x × anchor_w + anchor_x
     y = t_y × anchor_h + anchor_y
     w = exp(t_w) × anchor_w
     h = exp(t_h) × anchor_h

     # Chuyển về [y_min, x_min, y_max, x_max]
     ```

2. **`get_inference_model(num_classes)`**
   - **Chức năng**: Load model và weights đã train
   - **Quy trình**:
     ```python
     model = RetinaNet(num_classes, backbone)
     model(tf.zeros((1, 224, 224, 3)))  # Build model
     model.load_weights(WEIGHT_PATH)
     ```

3. **`run_inference(model, image_path, all_anchors, score_threshold=0.6)`**
   - **Chức năng**: Dự đoán trên 1 ảnh
   - **Input**: Model, đường dẫn ảnh, anchors, ngưỡng confidence
   - **Output**: Ảnh, boxes, scores, classes, số objects, ratio
   - **Quy trình**:
     ```
     1. Đọc và preprocess ảnh (resize + pad)
     2. predictions = model.predict(img)
     3. Tách box_predictions và class_predictions
     4. Decode boxes
     5. Áp dụng sigmoid cho class probabilities
     6. NMS (Non-Maximum Suppression):
        - IoU threshold: 0.4
        - Score threshold: 0.6
        - Max 10 objects
     7. Return kết quả
     ```

### 4.2. Non-Maximum Suppression (NMS)
- **Sử dụng**: `tf.image.combined_non_max_suppression`
- **Tham số**:
  - `max_output_size_per_class = 10`
  - `max_total_size = 10`
  - `iou_threshold = 0.4`: Loại bỏ box trùng lặp > 40% IoU
  - `score_threshold = 0.6`: Chỉ giữ box có confidence > 60%
  - `clip_boxes = False`: Không cắt box ra ngoài ảnh

---

## 5. LUỒNG DỮ LIỆU TOÀN HỆ THỐNG

### 5.1. Quá trình Training

```
CSV File (path, bbox, class_id)
    ↓
change_string2number() → DataFrame
    ↓
create_dataset_from_dataframe() → tf.data.Dataset
    ↓
read_img_and_label() → Đọc ảnh từ disk
    ↓
resize_and_pad_img() → Resize về 224×224
    ↓
padded_batch() → Tạo batch 32 ảnh
    ↓
pack_targets() → Encode labels bằng LabelEncoder
    ↓
[Images (32,224,224,3), Targets (32,4995,5)]
    ↓
RetinaNet Model
    ↓
    ResNet-50 Backbone → [C3, C4, C5]
    ↓
    FPN → [P3, P4, P5, P6, P7]
    ↓
    2 Heads → [Box Deltas, Class Logits]
    ↓
RetinaNetLoss (Focal Loss + Smooth L1)
    ↓
SGD Optimizer với Learning Rate Schedule
    ↓
ModelCheckpoint → Save best weights
```

### 5.2. Quá trình Inference

```
Image Path
    ↓
read_file() + decode_jpeg()
    ↓
resize_and_pad_img() → 224×224×3
    ↓
model.predict() → (1, 4995, 4+num_classes)
    ↓
Tách Box Deltas và Class Logits
    ↓
decode_box_predictions() → Boxes tọa độ thực
    ↓
sigmoid() → Class Probabilities
    ↓
NMS (Non-Maximum Suppression)
    ↓
Final Detections:
  - Boxes: [y_min, x_min, y_max, x_max]
  - Scores: Confidence 0-1
  - Classes: Class IDs
  - Num_detections: Số objects tìm thấy
```

---

## 6. ĐIỂM NỔI BẬT VÀ KỸ THUẬT ĐẶC BIỆT

### 6.1. Xử lý Class Imbalance
- **Focal Loss**: Giảm trọng số của easy examples, tập trung vào hard examples
- **Alpha balancing**: α=0.25 cho positive, 0.75 cho negative

### 6.2. Multi-Scale Detection
- **Feature Pyramid**: 5 levels (P3-P7) phát hiện vật thể từ nhỏ đến lớn
- **Strides**: [8, 16, 32, 64, 128] pixels
- **P3**: Vật thể nhỏ (8-16 pixels)
- **P7**: Vật thể lớn (128+ pixels)

### 6.3. Anchor Design
- **9 anchors/location**: 3 aspect ratios × 3 scales
- **Total anchors cho 224×224**: ~4,995 anchors
- **Aspect ratios**: [0.4, 0.5, 1.0] - phù hợp với người, xe

### 6.4. Data Augmentation
- **Resize + Padding**: Giữ tỷ lệ khung hình, không làm méo ảnh
- **Shuffle**: Buffer size 1000 để tăng tính ngẫu nhiên

### 6.5. Regularization
- **L2 Regularization**: 0.001 cho tất cả Conv layers trong ResNet
- **Batch Normalization**: Sau mỗi Conv layer
- **Early Stopping**: Patience=15 epochs

### 6.6. Optimization
- **SGD with Momentum**: Momentum=0.9
- **Learning Rate Schedule**: Tăng dần rồi giảm dần (warmup + decay)
- **Gradient Normalization**: Chia loss cho số positive anchors

---

## 7. CÁC FILE DỮ LIỆU

### 7.1. Cấu trúc CSV
```csv
path_img,bbox,class_id
/path/to/img.jpg,"[[x1,y1,w1,h1], [x2,y2,w2,h2]]","[1, 2]"
```

### 7.2. Thư mục `object_detect/`
```
object_detect/
├── csv_file/
│   ├── train_data.csv
│   ├── valid_data.csv
│   └── test_data.csv
├── imgs/              # Ảnh training
├── anno/              # Annotations (nếu có)
├── weight_store/      # Weights đã train
│   └── 100epochs-sgd.weights.h5
├── hist_store/        # Training history
│   └── 100epochs_sgd_csv.csv
└── recall_precision_csv/  # Metrics
```

---

## 8. CÁCH SỬ DỤNG

### 8.1. Training
```bash
python train.py
```
- Đọc dữ liệu từ CSV
- Train 100 epochs với SGD
- Lưu best weights vào `weight_store/`
- Lưu history vào `hist_store/`

### 8.2. Prediction
```bash
python predict.py
```
- Load weights từ `weight_store/100epochs-sgd.weights.h5`
- Dự đoán trên test set
- Hiển thị kết quả bằng Matplotlib:
  - Box màu cyan: Predictions
  - Box màu đỏ đứt nét: Ground Truth

### 8.3. Training Tiếp (Fine-tuning)
```bash
python train_tiep.py  # hoặc train_adam.py
```
- Load weights cũ
- Train tiếp với Adam optimizer
- Learning rate cố định 1e-4

---

## 9. HYPERPARAMETERS QUAN TRỌNG

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `target_size` | 224 | Kích thước ảnh input |
| `batch_size` | 32 | Số ảnh/batch |
| `num_anchors` | 9 | Anchors mỗi vị trí |
| `match_iou` | 0.4 | Ngưỡng IoU cho positive |
| `ignore_iou` | 0.2 | Ngưỡng IoU cho ignore |
| `focal_alpha` | 0.25 | Trọng số Focal Loss |
| `focal_gamma` | 2.0 | Focusing parameter |
| `box_variance` | [0.1,0.1,0.2,0.2] | Normalization cho box targets |
| `learning_rate` | 2.5e-6 → 0.0025 | LR schedule |
| `momentum` | 0.9 | SGD momentum |
| `epochs` | 100 | Số epochs |
| `patience` | 15 | Early stopping patience |
| `nms_iou` | 0.4 | NMS IoU threshold |
| `score_threshold` | 0.6 | Confidence threshold |

---

## 10. KẾT QUẢ VÀ ĐÁNH GIÁ

### 10.1. Metrics
- **Loss**: Classification loss + Box regression loss
- **Precision/Recall**: Được lưu trong `recall_precision_csv/`
- **mAP**: Mean Average Precision (cần chạy `evaluate_model.py`)

### 10.2. Visualize Training
```bash
python plot_history.py
```
- Vẽ đồ thị loss qua các epochs

---

## 11. TÀI LIỆU THAM KHẢO

### 11.1. Papers
- **RetinaNet**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **ResNet**: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **FPN**: "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)

### 11.2. Kiến trúc liên quan
- Faster R-CNN
- YOLO
- SSD

---

## 12. GHI CHÚ QUAN TRỌNG

1. **Ảnh rỗng**: LabelEncoder xử lý ảnh không có object (return all -1)
2. **NaN trong log**: Sử dụng `tf.maximum(gt_wh, 1e-7)` để tránh log(0)
3. **Clip boxes**: Set `clip_boxes=False` trong NMS để tránh mất box
4. **Variance**: PHẢI trùng khớp giữa encode (train) và decode (predict)
5. **Class indexing**: Class ID trong CSV bắt đầu từ 1, tự động trừ 1 về 0-indexed

---

**Ngày tạo tài liệu**: 11/03/2026
**Phiên bản TensorFlow**: 2.20.0
**Framework**: Keras 3.12.0
