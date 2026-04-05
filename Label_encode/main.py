import tensorflow as tf
from Anchor_box import compute_iou

class LabelEncoder:
    def __init__(self):

        self._match_iou = 0.5
        self._ignore_iou = 0.2
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target_xy = (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]

        safe_gt_wh = tf.math.maximum(matched_gt_boxes[:, 2:], 1e-7)
        box_target_wh = tf.math.log(safe_gt_wh / anchor_boxes[:, 2:])

        box_target = tf.concat([box_target_xy, box_target_wh], axis=-1)
        return box_target / self._box_variance

    def _encode_sample(self, gt_boxes, gt_classes, anchor_boxes):

        if tf.shape(gt_boxes)[0] == 0:
            return tf.zeros_like(anchor_boxes), tf.fill([tf.shape(anchor_boxes)[0]], -1.0)


        iou_matrix = compute_iou(gt_boxes, anchor_boxes)

        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        matched_gt_classes = tf.gather(gt_classes, matched_gt_idx)

        matched_iou = tf.reduce_max(iou_matrix, axis=1)


        positive_mask = tf.greater_equal(matched_iou, self._match_iou)
        negative_mask = tf.less(matched_iou, self._ignore_iou)


        target_classes = tf.cast(tf.fill(tf.shape(positive_mask), -2.0), tf.float32)
        target_classes = tf.where(negative_mask, -1.0, target_classes)
        target_classes = tf.where(positive_mask, tf.cast(matched_gt_classes, tf.float32), target_classes)

        target_boxes = self._compute_box_target(anchor_boxes, matched_gt_boxes)

        return target_boxes, target_classes

    def encode_batch(self, batch_images, gt_boxes, gt_classes, anchor_boxes):
        target_boxes, target_classes = tf.map_fn(
            fn=lambda x: self._encode_sample(x[0], x[1], anchor_boxes),
            elems=(gt_boxes, gt_classes),
            fn_output_signature=(tf.float32, tf.float32)
        )
        return batch_images, target_boxes, target_classes