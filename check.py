import tensorflow as tf
from resnet_50 import resnet_50_backbone
resnet50_backbone = resnet_50_backbone()

# 2. Tạo một model "nháp" chỉ chứa riêng backbone để xem summary
def debug_backbone(model_to_check, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    outputs = model_to_check(inputs)

    # Tạo wrapper model
    check_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Backbone_Debug")

    print("\n" + "=" * 50)
    print("🔍 ĐANG SOI NỘI TẠNG BACKBONE RESNET-50")
    print("=" * 50)
    check_model.summary()
    print("=" * 50 + "\n")


# Gọi hàm soi
debug_backbone(resnet50_backbone, input_shape=(224, 224, 3))
