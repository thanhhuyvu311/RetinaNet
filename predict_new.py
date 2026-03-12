import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from Anchor_box import Anchor_box
from RetinaNet import RetinaNet
from resnet_50 import resnet_50_backbone

# --- CAU HINH ---
TARGET_SIZE = 224
# Nho kiem tra dung duong dan file weight moi nhat cua bro
WEIGHT_PATH = '/home/huy/Documents/de_tai_tot_nghiep/object_detect/weight_store/new_model.weights.h5'

# THU MUC CHUA ANH CAN TEST (Bro sua lai thanh duong dan thuc te)
TEST_IMG_FOLDER = '/home/huy/Documents/de_tai_tot_nghiep/object_detect/test_img/train'


def decode_box_predictions(anchor_boxes, box_predictions):
    """
    Giai ma toa do tu delta sang [ymin, xmin, ymax, xmax] cho NMS
    """
    a_x, a_y, a_w, a_h = tf.split(anchor_boxes, 4, axis=-1)
    t_x, t_y, t_w, t_h = tf.split(box_predictions, 4, axis=-1)

    # Nhan nguoc lai voi variances da nen luc train
    t_x = t_x * 0.1
    t_y = t_y * 0.1
    t_w = t_w * 0.2
    t_h = t_h * 0.2

    x = t_x * a_w + a_x
    y = t_y * a_h + a_y
    w = tf.exp(t_w) * a_w
    h = tf.exp(t_h) * a_h

    x1, y1 = x - w / 2.0, y - h / 2.0
    x2, y2 = x + w / 2.0, y + h / 2.0

    return tf.concat([y1, x1, y2, x2], axis=-1)


def get_inference_model(num_classes):
    backbone = resnet_50_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    # Khoi tao shape cho model truoc khi load weight
    model(tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3)))
    model.load_weights(WEIGHT_PATH)
    return model


def run_inference(model, image_path, all_anchors, score_threshold=0.5):
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

    box_predictions = predictions[:, :, :4]
    class_predictions = predictions[:, :, 4:]

    decoded_boxes = decode_box_predictions(all_anchors, box_predictions)
    class_probs = tf.nn.sigmoid(class_predictions)

    # NMS (Luon nho clip_boxes=False nhe bro)
    nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(
        tf.expand_dims(decoded_boxes, axis=2),
        class_probs,
        max_output_size_per_class=10,
        max_total_size=10,
        iou_threshold=0.4,
        score_threshold=score_threshold,
        clip_boxes=False
    )

    return img_padded.numpy().astype("uint8"), nms_boxes[0], nms_scores[0], nms_classes[0], valid_detections[0]


if __name__ == '__main__':
    # Fix cung num_class = 1 (vi dataset cua bro chi co class 'person')
    num_class = 1

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    # Quet tat ca cac file .jpg trong thu muc test_img_folder
    # Bro co the doi thanh .png neu anh cua bro duoi png
    img_paths = glob.glob(os.path.join(TEST_IMG_FOLDER, '*.jpg'))

    if len(img_paths) == 0:
        print(f"Khong tim thay buc anh nao trong thu muc: {TEST_IMG_FOLDER}")

    # Duyet qua tung anh de du doan
    for i, img_path in enumerate(img_paths):
        # Chinh score_threshold tuy theo do "chin" cua model (Hien tai cu de 0.5 hoac 0.4)
        img, boxes, scores, classes, num_det = run_inference(model, img_path, all_anchors, score_threshold=0.4)

        print(f"--- Anh: {os.path.basename(img_path)} ---")
        print(f"So luong vat the tim thay: {num_det}")

        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(img)

        # Ve du doan (Mau Xanh/Cyan), khong con ve Ground Truth nua
        for j in range(num_det):
            ymin, xmin, ymax, xmax = float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3])
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='cyan',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"Pred: {scores[j]:.2f}", color='cyan', fontsize=10, backgroundcolor='black')

        plt.title(f"Du doan tu do - Found {num_det} objects")
        plt.axis('off')
        plt.show()