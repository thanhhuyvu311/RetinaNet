import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt  # Thêm thư viện để vẽ đồ thị

from predict_drone import get_inference_model, run_inference, TARGET_SIZE, TEST_CSV
from Anchor_box import Anchor_box


def calculate_iou(box1, box2):
    """Tinh IoU cho 2 hop co dang [xmin, ymin, xmax, ymax]"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-8)
    return iou


def evaluate_model():
    print("Dang load Model va Data de danh gia...")
    test_df = pd.read_csv(TEST_CSV)

    # Lay thong tin model
    all_labels = []
    for cid in test_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)
    num_class = len(set(all_labels))

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    all_predictions = []
    total_gt_boxes = 0

    print(f"Bat dau quet {len(test_df)} anh trong tap Test...")
    for index, row in test_df.iterrows():
        img_path = row['path_img']
        gt_boxes_raw = ast.literal_eval(row['bbox'])

        # Chay du doan voi threshold thap de ve PR Curve
        _, pred_boxes, pred_scores, _, num_det, ratio = run_inference(
            model, img_path, all_anchors, score_threshold=0.3
        )

        # Xu ly Ground Truth (Chuyen tu [x_c, y_c, w, h] to -> [xmin, ymin, xmax, ymax] ty le 224)
        gt_boxes = []
        for box in gt_boxes_raw:
            x_c, y_c, w, h = box
            x_c, y_c, w, h = x_c * ratio, y_c * ratio, w * ratio, h * ratio
            xmin = x_c - w / 2.0
            ymin = y_c - h / 2.0
            xmax = x_c + w / 2.0
            ymax = y_c + h / 2.0
            gt_boxes.append([xmin, ymin, xmax, ymax])
            total_gt_boxes += 1

        # Luu lai danh sach cac GT chua bi khop
        gt_matched = [False] * len(gt_boxes)

        # Luu lai cac hop du doan
        for i in range(num_det):
            ymin, xmin, ymax, xmax = pred_boxes[i]
            all_predictions.append({
                'image_idx': index,
                'box': [xmin, ymin, xmax, ymax],
                'score': pred_scores[i],
                'gt_boxes': gt_boxes,  # Gan kem GT cua anh nay de so sanh
                'gt_matched': gt_matched  # Trang thai khop cua GT trong anh nay
            })

    # --- TINH TOAN AP (Average Precision) ---
    print("Dang tinh toan mAP...")
    if total_gt_boxes == 0 or len(all_predictions) == 0:
        print("Khong co Ground Truth hoac khong co Du doan nao. mAP = 0.0")
        return

    # 1. Sap xep tat ca cac du doan theo Score giam dan
    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    true_positives = np.zeros(len(all_predictions))
    false_positives = np.zeros(len(all_predictions))

    # 2. So khop tung cai hop du doan voi Ground Truth
    for i, pred in enumerate(all_predictions):
        pred_box = pred['box']
        gt_boxes = pred['gt_boxes']
        gt_matched = pred['gt_matched']

        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Neu IoU > 0.5 va cai GT do chua bi cai khung du doan nao khac "an" mat
        if best_iou >= 0.5 and not gt_matched[best_gt_idx]:
            true_positives[i] = 1
            gt_matched[best_gt_idx] = True  # Danh dau GT nay da bi khop
        else:
            false_positives[i] = 1

    # 3. Tinh cong don
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)

    # 4. Tinh Precision va Recall
    recalls = cum_tp / total_gt_boxes
    precisions = cum_tp / (cum_tp + cum_fp)

    # 5. Tinh dien tich duoi bieu do (AUC) de ra AP
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Lam phang duong Precision (de do thi khong bi giat cuc - chuan Pascal VOC)
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # Tinh tich phan
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    # --- LƯU RA FILE CSV & ĐỒ THỊ ---
    pr_df = pd.DataFrame({
        'recall': recalls,
        'precision': precisions
    })

    base_dir = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc'
    recall_precision_dir = os.path.join(base_dir, 'recall_precision_csv')
    pr_df_path = os.path.join(recall_precision_dir, 'drone_Thermal_512_tiling_new-')

    # Tạo folder nếu chưa tồn tại để tránh lỗi
    os.makedirs(pr_df_path, exist_ok=True)

    # 1. Lưu file CSV với đuôi mở rộng rõ ràng
    csv_file = os.path.join(pr_df_path, 'recall_precision.csv')
    pr_df.to_csv(csv_file, index=False)

    # 2. Vẽ đồ thị Precision-Recall Curve và lưu lại
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', lw=2, label=f'AP = {ap * 100:.2f}%')
    plt.fill_between(recalls, precisions, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left")

    # Lưu ảnh đồ thị
    plot_file = os.path.join(pr_df_path, 'pr_curve.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # Đóng figure sau khi vẽ xong

    print("--------------------------------------------------")
    print(f"Tong so Ground Truth (Thuc te): {total_gt_boxes}")
    print(f"Tong so Boxes mo hinh ve ra: {len(all_predictions)}")
    print(f"AP (Average Precision @ IoU=0.5): {ap * 100:.2f}%")
    print(f"==> Da luu file CSV tai: {csv_file}")
    print(f"==> Da luu do thi tai: {plot_file}")
    print("--------------------------------------------------")


if __name__ == '__main__':
    evaluate_model()