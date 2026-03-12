import tensorflow as tf
from Anchor_box import Anchor_box
from Label_encode import LabelEncoder
from Xu_ly_du_lieu import preprocessing_data_before_training
from RetinaNet import RetinaNet,RetinaNetLoss
from resnet_50 import resnet_50_backbone
import os
import pandas as pd
import ast

if __name__ == '__main__':
    base_dir = '/home/huy/Documents/de_tai_tot_nghiep/object_detect'
    #---KHOI TAO ANCHOR BOX ---#
    target_size = 224
    anchor_gene = Anchor_box()
    #tao anchor
    all_anchor = anchor_gene.get_anchors(img_h=target_size,img_w=target_size)
    #khoi tao label encoder
    label_encoder = LabelEncoder()

    def pack_targets(img, bdb, cls):
        # Lấy target_boxes (4 cột) và target_classes (1 cột) từ encoder
        _,target_boxes, target_classes = label_encoder.encode_batch(img, bdb, cls, all_anchor)

        # Ép kiểu target_classes về float32 để đồng nhất với target_boxes
        target_classes = tf.cast(target_classes, tf.float32)

        # Mở rộng chiều của target_classes để nó có dạng (Batch, Num_Anchor, 1)
        target_classes = tf.expand_dims(target_classes, axis=-1)

        # Hàn 2 cái lại thành 1 cục y_true duy nhất có 5 cột (Batch, Num_Anchor, 5)
        y_true = tf.concat([target_boxes, target_classes], axis=-1)

        # Trả về ĐÚNG 2 món: Ảnh (Input) và Nhãn gộp (Target) cho Keras
        return img, y_true

    #--- XU LY DATASET VA GAN NHAN ---#

    train_csv_path = '/home/huy/Documents/de_tai_tot_nghiep/object_detect/csv_file/train_data.csv'
    val_csv_path = '/home/huy/Documents/de_tai_tot_nghiep/object_detect/csv_file/valid_data.csv'
    raw_train_data = preprocessing_data_before_training.change_string2number(train_csv_path)
    raw_val_data = preprocessing_data_before_training.change_string2number(val_csv_path)
    train_dataset = preprocessing_data_before_training.create_dataset_from_dataframe(raw_train_data).shuffle(buffer_size=1000)
    val_dataset = preprocessing_data_before_training.create_dataset_from_dataframe(raw_val_data)

    #doc anh,resize,scale bdb
    train_dataset = train_dataset.map(preprocessing_data_before_training.read_img_and_label,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(
        lambda img,bdb,cls : preprocessing_data_before_training.resize_and_pad_img(img,bdb,cls,target_size),
        num_parallel_calls= tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.map(preprocessing_data_before_training.read_img_and_label,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        lambda img, bdb, cls: preprocessing_data_before_training.resize_and_pad_img(img, bdb, cls, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    #gom batch size
    batch_size = 32

    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(0.0,1e-8,-2),
        drop_remainder=True #cat bo nhung anh le khong cung 1 batch
    )

    train_dataset = train_dataset.map(
        pack_targets,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_dataset = val_dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(0.0, 1e-8, -2),
        drop_remainder=True  # cat bo nhung anh le khong cung 1 batch
    )

    val_dataset = val_dataset.map(
        pack_targets,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


    #------------------------------#
    #                              #
    #      Khoi tao mo hinh        #
    #------------------------------#

    train_df = pd.read_csv(train_csv_path)
    all_labels = []
    for cid in train_df['class_id'].apply(ast.literal_eval):
        all_labels.extend(cid)

    num_class = len(set(all_labels))
    num_classes = num_class

    resnet50_backbone = resnet_50_backbone()
    model = RetinaNet(num_classes=num_classes, backbone=resnet50_backbone)

    # ====== ĐOẠN CODE MỚI ĐỂ LOAD WEIGHTS CŨ ======
    weight_path = os.path.join(base_dir, 'weight_store', '100epochs-sgd.weights.h5')

    # Kiểm tra xem file weights có tồn tại không
    if os.path.exists(weight_path):
        print(f"🔥🔥 Phát hiện 'não' cũ tại: {weight_path}")
        print("Đang tiến hành lắp não để train tiếp...")

        # 1. Dựng khung xương cho model bằng 1 batch ảnh rỗng (Dummy input)
        model(tf.zeros((1, target_size, target_size, 3)))

        # 2. Bơm trọng số cũ vào
        model.load_weights(weight_path)
        print("✅ Lắp não thành công! Mô hình sẽ học tiếp từ đẳng cấp hiện tại.")
    else:
        print("🌟 Không tìm thấy file weights cũ. Mô hình sẽ học lại từ đầu như một tờ giấy trắng!")
    # ==============================================

    # 2. Cài đặt Optimizer

    # Dùng thuật toán Stochastic Gradient Descent (SGD) với momentum
    optimizer = tf.optimizers.SGD(learning_rate=0.000025, momentum=0.9)
    loss_fn = RetinaNetLoss(num_classes=num_classes)

    # 3. Compile model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 4. Thiết lập Callbacks để lưu trọng số và tự dừng
    weight_folder = os.path.join(base_dir,'weight_store')
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_folder+"/200epochs-sgd.weights.h5",  # File lưu weights
            monitor="loss",
            save_best_only=True,  # Chỉ lưu khi loss giảm
            save_weights_only=True,  # Subclassed model bắt buộc phải lưu dạng weights_only
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=15,  # Nếu sau 10 Epochs mà loss không giảm thì tự ngắt
            verbose=1,
            restore_best_weights=True
        )
    ]
    EPOCHS = 100

    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list,
    )
    hist_df = pd.DataFrame(hist.history)

    model_store_folder = os.path.join(base_dir,'model_store')
    hist_store_folder = os.path.join(base_dir,'hist_store')
    hist_df.to_csv(hist_store_folder+'/200epochs_sgd_csv.csv')



