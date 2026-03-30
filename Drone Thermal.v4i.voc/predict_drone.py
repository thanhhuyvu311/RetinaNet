import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- TOI UU HOA VRAM ---
# Bat Mixed Precision va Memory Growth de chay sieu toc
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from Anchor_box import Anchor_box
from RetinaNet import RetinaNet
from resnet_50 import resnet_50_backbone

# --- CAU HINH ---
TARGET_SIZE = 512
BATCH_SIZE = 1

WEIGHT_PATH = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/weight_store/512_finetune.weights.h5'
TEST_CSV = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/csv_file/test_data.csv'


def decode_box_predictions(anchor_boxes, box_predictions):
    """
    Giai ma toa do tu delta sang [ymin, xmin, ymax, xmax] cho NMS
    """
    # Tach toa do Anchor (x_center, y_center, w, h)
    a_x, a_y, a_w, a_h = tf.split(anchor_boxes, 4, axis=-1)

    # Tach Delta du doan
    t_x, t_y, t_w, t_h = tf.split(box_predictions, 4, axis=-1)

    # Nhan nguoc lai voi variances da nen luc train
    t_x = t_x * 0.1
    t_y = t_y * 0.1
    t_w = t_w * 0.2
    t_h = t_h * 0.2

    # Giai ma Center X, Y
    x = t_x * a_w + a_x
    y = t_y * a_h + a_y

    # Giai ma Width, Height
    w = tf.exp(t_w) * a_w
    h = tf.exp(t_h) * a_h

    # Chuyen ve dang goc [xmin, ymin, xmax, ymax]
    x1, y1 = x - w / 2.0, y - h / 2.0
    x2, y2 = x + w / 2.0, y + h / 2.0

    # Tra ve y-first cho ham combined_non_max_suppression cua TF
    return tf.concat([y1, x1, y2, x2], axis=-1)


def get_inference_model(num_classes):
    backbone = resnet_50_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=backbone)

    # Khoi tao shape cho model truoc khi load weight
    model(tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3)))
    model.load_weights(WEIGHT_PATH)
    return model


def run_inference(model, image_path, all_anchors, score_threshold):
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.cast(img, tf.float32)

    # Tien xu ly (Pad & Resize) y het nhu luc Train
    shape = tf.cast(tf.shape(img)[:2], dtype=tf.float32)
    img_h, img_w = shape[0], shape[1]
    ratio = tf.cast(TARGET_SIZE, tf.float32) / tf.math.maximum(img_h, img_w)

    new_img_h = tf.cast(img_h * ratio, tf.int32)
    new_img_w = tf.cast(img_w * ratio, tf.int32)

    img_resized = tf.image.resize(img, [new_img_h, new_img_w])
    img_padded = tf.image.pad_to_bounding_box(img_resized, 0, 0, TARGET_SIZE, TARGET_SIZE)

    img_input = tf.expand_dims(img_padded, axis=0)

    # Du doan
    predictions = model.predict(img_input, verbose=0)

    # Vi dang dung mixed_float16 nen can ep kieu ve float32 de NMS xu ly chinh xac
    box_predictions = tf.cast(predictions[:, :, :4], tf.float32)
    class_predictions = tf.cast(predictions[:, :, 4:], tf.float32)

    decoded_boxes = decode_box_predictions(all_anchors, box_predictions)
    class_probs = tf.nn.sigmoid(class_predictions)

    # Dung NMS de loc cac hop trung lap
    nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
        tf.expand_dims(decoded_boxes, axis=2),
        class_probs,
        max_output_size_per_class=20,  # Co the tang len neu anh co nhieu vat the
        max_total_size=20,
        iou_threshold=0.4,
        score_threshold=score_threshold,
        clip_boxes=False
    )

    return img_padded.numpy().astype("uint8"), nms_boxes[0], nms_scores[0], nms_classes[0], valid_detections[
        0], ratio.numpy()


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_CSV)
    all_labels = []
    for cid in test_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)

    num_class = len(set(all_labels))

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    # Lay 20 anh test thu
    for i in range(20):
        row = test_df.iloc[i]
        img_path = row['path_img']
        gt_boxes = ast.literal_eval(row['bbox'])

        # Ha score_threshold xuong neu muon soi cac box co do tu tin thap hon
        img, boxes, scores, classes, num_det, ratio = run_inference(model, img_path, all_anchors, score_threshold=0.5)

        print(f"\n--- Anh thu {i + 1} ---")
        print(f"So luong vat the tim thay: {num_det}")
        if num_det > 0:
            print(f"Diem tin tuong (Scores): {scores[:num_det].numpy()}")

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img)

        # Ve Box du doan (Mau Cyan)
        for j in range(num_det):
            ymin, xmin, ymax, xmax = float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3])

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"Pred: {scores[j]:.2f}", color='cyan', fontsize=12, backgroundcolor='black')

        # Ve Ground Truth (Mau Do - Net Dut)
        for box in gt_boxes:
            x_c, y_c, w, h = box
            x_c_scaled, y_c_scaled = x_c * ratio, y_c * ratio
            w_scaled, h_scaled = w * ratio, h * ratio

            rect_gt = patches.Rectangle((x_c_scaled - w_scaled / 2.0, y_c_scaled - h_scaled / 2.0),
                                        w_scaled, h_scaled, linewidth=2, edgecolor='red',
                                        facecolor='none', linestyle='--')
            ax.add_patch(rect_gt)

        plt.title(f"Test Image {i + 1} - Found {num_det} objects", fontsize=14)
        plt.axis('off')
        plt.show()