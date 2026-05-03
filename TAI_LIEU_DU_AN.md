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
Áp dụng kỹ thuật **tiling** (chia ảnh thành các tile nhỏ có chồng lấp): thay vì resize cả ảnh 1280×1024 → 512×512, ta cắt ra các tile (640x512) từ ảnh gốc, sau đó resize từng tile về 512×512. Điều này giúp các đối tượng người to hơn trong mỗi tile, cải thiện đáng kể khả năng phát hiện.

### Công nghệ sử dụng
| Thành phần   | Phiên bản |
|--------------|---|
| Python       | 3.10.12 |
| TensorFlow   | 2.20.0 |
| Keras        | 3.12.1 |
| NumPy        | 2.2.6 |
| Pandas       | 2.3.3 |
| Matplotlib   | 3.10.8 |
| Pillow (PIL) | mới nhất |
---

## 2. CẤU TRÚC THƯ MỤC

```
de_tai_tot_nghiep/
│
├── Anchor_box/                     # Module sinh Anchor Box
│   └── main.py                     # Tạo 9 anchor/vị trí, 5 pyramid level (P3-P7)
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
│   ├── main_step_0.py              # Tổ chức dữ liệu từ train/val/test vào imgs/anno
│   ├── main_step_123.py            # Tạo tile images (720x576) và CSV annotation
│   ├── main_step_123_not_tilling.py # Tạo CSV annotation (không tiling)
│   ├── chia_data.py                # Chia train/val/test từ CSV gốc
│   ├── preprocessing_data_before_training.py  # Pipeline tf.data đọc ảnh training
│   └── analize_histogram_data.py   # Phân tích phân phối kích thước box
│
├── Drone Thermal.v4i.voc/             # Thư mục chứa dataset và file chạy phụ trợ
│   ├── imgs/                          # Toàn bộ ảnh gốc (.jpg)
│   ├── anno/                          # Toàn bộ annotations (.xml)
│   ├── tiles/                         # Ảnh tile đã được cắt ra (tiling mode)
│   ├── csv_file/                      # Các file CSV chứa thông tin box/path
│   │   ├── train_data_noneTile.csv    # Dataset không dùng tiling
│   │   ├── data_information_tiled_2.csv # Dataset dùng tiling
│   │   └── ...
│   ├── weight_store/               # Lưu trữ weights model (.weights.h5)
│   │   ├── RETINANET_TILE_2img.weights.h5
│   │   └── retinanet_tiled_512-new-.weights.h5
│   ├── predict_drone.py            # Inference tiling cho drone dataset
│   ├── evaluate_drone_model.py     # Đánh giá AP trên drone dataset
│   └── recall_precision_csv/       # Lưu kết quả Precision-Recall
│
├── train_adam.py                   # Script huấn luyện chính (Adam + tiling)
├── train.py                        # Script huấn luyện cơ bản (SGD/Adam)
├── train_tiep.py                   # Tiếp tục huấn luyện từ checkpoint
├── fine_tune_adam.py               # Fine-tuning model
├── predict.py                      # Inference + trực quan hóa kết quả (tiling)
├── predict_new.py                  # Inference trên thư mục ảnh mới (tiling)
├── evaluate_model.py               # Đánh giá mAP trên tập test (tiling)
├── plot_history.py                 # Vẽ biểu đồ loss/metric từ lịch sử training
├── TILING_README.md                # Hướng dẫn chi tiết kỹ thuật Tiling
└── TAI_LIEU_DU_AN.md               # Tài liệu này
```

---

## 3. BỘ DỮ LIỆU DRONE THERMAL

### Thông tin chung
- **Nguồn:** Roboflow — Drone Thermal v4i (định dạng Pascal VOC)
- **Tổng số ảnh gốc:** ~2,866 ảnh
- **Kích thước ảnh gốc:** 1280×1024 pixel
- **Số lớp:** 2 (0: background, 1: person)
- **Định dạng annotation:** XML (Pascal VOC)

### Quy trình chuẩn bị dữ liệu
1. **Gom dữ liệu:** `Xu_ly_du_lieu/main_step_0.py` di chuyển ảnh và XML từ các thư mục rải rác vào `imgs/` và `anno/`.
2. **Tiling:** `Xu_ly_du_lieu/main_step_123.py` chia ảnh 1280x1024 thành các tile 640x512 với overlap 20%. Bbox được lọc (giữ lại nếu diện tích trong tile > 30%) và chuyển sang tọa độ tile.
3. **Phân chia:** `Xu_ly_du_lieu/chia_data.py` thực hiện chia train/val/test theo ID ảnh gốc để tránh rò rỉ dữ liệu giữa các tile của cùng một ảnh.

---

## 4. KIẾN TRÚC MÔ HÌNH

### 4.1 Backbone (ResNet-50)
Sử dụng ResNet-50 trích xuất đặc trưng tại 3 cấp độ:
- **C3:** Stride 8 (64x64 cho input 512x512)
- **C4:** Stride 16 (32x32)
- **C5:** Stride 32 (16x16)

### 4.2 Neck (FPN)
Feature Pyramid Network tạo ra 5 cấp độ feature map (P3 đến P7):
- **P3, P4, P5:** Từ C3, C4, C5 qua phép cộng top-down và conv 1x1.
- **P6, P7:** Tạo ra từ P5 qua conv stride 2 để tăng receptive field.

### 4.3 Head (Subnets)
- **Classification Head:** 4 lớp Conv 3x3 + 1 lớp Conv 3x3 cuối (num_anchors * num_classes).
- **Regression Head:** 4 lớp Conv 3x3 + 1 lớp Conv 3x3 cuối (num_anchors * 4).

---

## 5. PIPELINE TIỀN XỬ LÝ DỮ LIỆU

Mỗi batch dữ liệu đi qua các bước:
1. **Load Image:** Đọc ảnh tile từ đường dẫn trong CSV.
2. **Resize & Pad:** Resize ảnh về kích thước mục tiêu (ví dụ 512x512) giữ nguyên tỉ lệ và bù (pad) các vùng trống.
3. **Label Encoding:** 
   - Khớp Ground Truth với Anchor Boxes dựa trên IoU.
   - IoU > 0.4: Positive.
   - IoU < 0.2: Negative.
   - Ở giữa: Ignore.
   - Tính toán delta (offsets) giữa GT box và Anchor box.

---

## 6. PIPELINE HUẤN LUYỆN

### Cấu hình chính (`train_adam.py`)
- **Mixed Precision:** `mixed_float16` giúp giảm sử dụng VRAM và tăng tốc training trên GPU hỗ trợ Tensor Cores.
- **Optimizer:** Adam với Learning Rate $5 \times 10^{-5}$.
- **Loss:** RetinaNetLoss (Focal Loss cho classification + Smooth L1 cho regression).
- **Batch Size:** 8 (tối ưu cho tile 512x512 trên GPU).

### Callbacks
- `ReduceLROnPlateau`: Giảm LR khi loss không cải thiện sau 5 epoch.
- `ModelCheckpoint`: Lưu weight tốt nhất dựa trên `val_loss`.
- `EarlyStopping`: Dừng training nếu `val_loss` không giảm sau 10 epoch.

---

## 7. PIPELINE SUY LUẬN (INFERENCE)

### Tiled Inference
Đây là cơ chế quan trọng nhất để xử lý ảnh độ phân giải cao:
1. **Chia Tile:** Ảnh gốc được chia thành các tile theo thông số `TILE_W`, `TILE_H`, `OVERLAP`.
2. **Dự đoán cục bộ:** Model chạy inference trên từng tile.
3. **Chuyển tọa độ:** Tọa độ box dự đoán từ tile (0->512) được scale và tịnh tiến về tọa độ ảnh gốc (0->1280, 0->1024).
4. **Global NMS:** Áp dụng thuật toán Non-Maximum Suppression trên toàn bộ danh sách box thu được từ tất cả các tile để loại bỏ các box trùng lặp tại vùng chồng lấp.

---

## 8. PIPELINE ĐÁNH GIÁ MÔ HÌNH

### Chỉ số mAP (Mean Average Precision)
Đánh giá được thực hiện trên tập Test bằng `evaluate_model.py`:
- Sử dụng **Tiled Inference** để dự đoán.
- So khớp với Ground Truth ở hệ tọa độ gốc.
- Tính toán Precision và Recall tại các ngưỡng score khác nhau.
- **Average Precision (AP):** Tính diện tích dưới đường cong Precision-Recall (PR Curve).

---

## 9. SIÊU THAM SỐ & CẤU HÌNH

| Tham số | Giá trị              |
|---|----------------------|
| Input Size | 512 x 512            |
| Tile Size | 640 x 512            |
| Overlap | 22%                  |
| Learning Rate | 5e-5                 |
| Batch Size | 8                    |
| Focal Loss Gamma | 2.0                  |
| Focal Loss Alpha | 0.25                 |
| Box Variances | [0.1, 0.1, 0.2, 0.2] |

---

## 10. CÁCH CHẠY DỰ ÁN

### Chuẩn bị
1. Cài đặt môi trường: `pip install tensorflow keras pandas numpy matplotlib pillow`
2. Đặt dữ liệu vào `Drone Thermal.v4i.voc/`.

### Thực hiện theo thứ tự
1. **Chuẩn bị data:** `python Xu_ly_du_lieu/main_step_0.py`
2. **Tạo tiles:** `python Xu_ly_du_lieu/main_step_123.py`
3. **Chia dataset:** `python Xu_ly_du_lieu/chia_data.py`
4. **Huấn luyện:** `python train_adam.py`
5. **Đánh giá:** `python evaluate_model.py`
6. **Dự đoán ảnh mới:** `python predict_new.py` (chỉnh sửa `TEST_IMG_FOLDER` trong code)

---

## 11. PHỤ LỤC: CÁC VẤN ĐỀ KỸ THUẬT VÀ CÁCH XỬ LÝ

### 11.1 Mixed Precision và NMS
Model output ở kiểu `float16` khi dùng Mixed Precision. Trước khi gọi `tf.image.combined_non_max_suppression`, các tensor boxes và scores cần được cast về `float32` để đảm bảo tính ổn định và tránh lỗi kiểu dữ liệu.

### 11.2 Anchor Boxes Optimization
Kích thước Anchor Boxes (`_areas` trong `Anchor_box/main.py`) đã được điều chỉnh dựa trên phân tích histogram kích thước đối tượng sau khi tiling. Việc này giúp model tăng tỉ lệ Recall đối với các đối tượng người vốn rất nhỏ.

### 11.3 Xử lý ranh giới Tile
Tại ranh giới giữa các tile, đối tượng có thể bị chia cắt. Nhờ cơ chế **Overlap 22%**, mọi đối tượng luôn nằm trọn vẹn trong ít nhất một tile, đảm bảo không bỏ sót đối tượng.
