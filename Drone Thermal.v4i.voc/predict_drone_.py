import tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
from Anchor_box import Anchor_box
from RetinaNet import RetinaNet
from resnet_50 import resnet_50_backbone
from Xu_ly_du_lieu.tiling_utils import generate_tile_coords, apply_global_nms

# --- CAU HINH ---
TARGET_SIZE = 512
WEIGHT_PATH = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/weight_store/retinanet_tiled_512-new.weights.h5'

# THU MUC CHUA ANH CAN TEST (sua lai thanh duong dan thuc te)
TEST_IMG_FOLDER = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/new-raw-img'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
def decode_box_predictions(anchor_boxes, box_predictions):
    anchor_boxes = tf.cast(anchor_boxes, box_predictions.dtype)
    a_x, a_y, a_w, a_h = tf.split(anchor_boxes, 4, axis=-1)
    t_x, t_y, t_w, t_h = tf.split(box_predictions, 4, axis=-1)

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
    model(tf.zeros((1, TARGET_SIZE, TARGET_SIZE, 3)))
    model.load_weights(WEIGHT_PATH)
    return model


def run_tiled_inference(model, image_path, all_anchors,
                        tile_w=640, tile_h=512, overlap=0.2,
                        score_threshold=0.5):
    """
    Chia anh goc thanh cac tile, chay inference tren tung tile,
    gop ket qua va ap dung Global NMS.
    Tra ve boxes trong he toa do anh goc.
    """
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.cast(img, tf.float16)
    orig_shape = tf.shape(img)
    orig_h = int(orig_shape[0])
    orig_w = int(orig_shape[1])

    tile_coords = generate_tile_coords(orig_w, orig_h, tile_w, tile_h, overlap)

    all_boxes = []
    all_scores = []

    for (tx, ty, tw, th) in tile_coords:
        tile = img[ty:ty + th, tx:tx + tw]

        tile_shape = tf.cast(tf.shape(tile)[:2], dtype=tf.float16)
        t_h_f, t_w_f = tile_shape[0], tile_shape[1]
        tile_ratio = tf.cast(TARGET_SIZE, tf.float16) / tf.math.maximum(t_h_f, t_w_f)

        new_h = tf.cast(t_h_f * tile_ratio, tf.int32)
        new_w = tf.cast(t_w_f * tile_ratio, tf.int32)

        tile_resized = tf.image.resize(tile, [new_h, new_w])
        tile_padded = tf.image.pad_to_bounding_box(tile_resized, 0, 0, TARGET_SIZE, TARGET_SIZE)
        tile_input = tf.expand_dims(tile_padded, axis=0)

        predictions = model.predict(tile_input, verbose=0)
        box_preds = predictions[:, :, :4]
        class_preds = predictions[:, :, 4:]

        decoded_boxes = decode_box_predictions(all_anchors, tf.cast(box_preds, tf.float32))
        class_probs = tf.nn.sigmoid(tf.cast(class_preds, tf.float32))

        nms_boxes, nms_scores, _, valid_det = tf.image.combined_non_max_suppression(
            tf.expand_dims(decoded_boxes, axis=2),
            class_probs,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.3,
            score_threshold=score_threshold,
            clip_boxes=False
        )

        ratio_val = float(tile_ratio)
        n_det = int(valid_det[0])

        for i in range(n_det):
            ymin, xmin, ymax, xmax = [float(v) for v in nms_boxes[0][i]]
            all_boxes.append([
                ymin / ratio_val + ty,
                xmin / ratio_val + tx,
                ymax / ratio_val + ty,
                xmax / ratio_val + tx,
            ])
            all_scores.append(float(nms_scores[0][i]))

    final_boxes, final_scores, num_det = apply_global_nms(
        all_boxes, all_scores,
        iou_threshold=0.3,
        score_threshold=score_threshold
    )

    orig_img_uint8 = tf.cast(img, tf.uint8).numpy()
    return orig_img_uint8, final_boxes, final_scores, num_det


if __name__ == '__main__':
    num_class = 1

    anchor_gene = Anchor_box()
    all_anchors = anchor_gene.get_anchors(img_h=TARGET_SIZE, img_w=TARGET_SIZE)
    model = get_inference_model(num_class)

    # Ho tro ca .jpg va .png
    img_paths = glob.glob(os.path.join(TEST_IMG_FOLDER, '*.jpg'))
    img_paths += glob.glob(os.path.join(TEST_IMG_FOLDER, '*.png'))

    if len(img_paths) == 0:
        print(f"Khong tim thay anh nao trong thu muc: {TEST_IMG_FOLDER}")

    for img_path in img_paths:
        img, boxes, scores, num_det = run_tiled_inference(
            model, img_path, all_anchors, score_threshold=0.5
        )

        print(f"--- Anh: {os.path.basename(img_path)} ---")
        print(f"So luong vat the tim thay: {num_det}")

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        for j in range(num_det):
            ymin, xmin, ymax, xmax = float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2]), float(boxes[j][3])
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"Pred: {scores[j]:.2f}", color='cyan',
                    fontsize=10, backgroundcolor='black')

        plt.title(f"Du doan - Found {num_det} objects")
        plt.axis('off')
        plt.show()
