import tensorflow as tf


def Conv_block(x, filter, stride):
    """
    x: input
    filter: so luong filter
    stride: buoc nhay
    Day la block lam thay doi kich thuoc input (chi giam hoac khong giam)
    """
    x_skip = x
    f1, f2 = filter

    # block dau tien
    #neu filter stride la 1 thi chi la buoc nhay 1, neu stride la 2 thi no dai dien cho viec maxpool
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block thu 2
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block thu 3 (tang so kenh ban do dac trung)
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # tang so kenh x_skip de cong vao x tai block 3
    x_skip = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), strides=(stride, stride), padding='valid',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # cong x_skip vao x
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)

    return x #(so kenh ban do dac trung = f2)


def Res_id_block(x, filter):
    """
    resnet block khong thay doi kich thuoc
    """
    x_skip = x #(so kenh x skip = f2 vi la return cua ham conv_block)
    f1, f2 = filter

    # block 1
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block 2
    x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # block 3
    x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(1, 1), padding='valid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # add
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)
    return x


def resnet_50_backbone():
    # input size 224x224x3
    input_dim = tf.keras.layers.Input(shape=(None, None, 3))

    # su dung padding de giu lai bien anh truoc khi vao conv2d 7x7
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_dim)

    # khoi 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', use_bias=False,
                               name='Conv7x7')(x)
    block_1_out = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)


    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='overlapping')(x)

    # khoi 2
    x = Conv_block(x, (64, 256), 1)
    x = Res_id_block(x, (64, 256))
    x = Res_id_block(x, (64, 256))
    block_2_out = x
    # khoi 3 -> P3 (Stride 8)
    x = Conv_block(x, (128, 512), 2)
    x = Res_id_block(x, (128, 512))
    x = Res_id_block(x, (128, 512))
    x = Res_id_block(x, (128, 512))
    block_3_out = x

    # khoi 4 -> P4 (Stride 16)
    x = Conv_block(x, (256, 1024), 2)
    x = Res_id_block(x, (256, 1024))
    x = Res_id_block(x, (256, 1024))
    x = Res_id_block(x, (256, 1024))
    x = Res_id_block(x, (256, 1024))
    x = Res_id_block(x, (256, 1024))
    block_4_out = x

    # khoi 5 -> P5 (Stride 32)
    x = Conv_block(x, (512, 2048), 2)
    x = Res_id_block(x, (512, 2048))
    x = Res_id_block(x, (512, 2048))
    block_5_out = x

    model = tf.keras.models.Model(inputs=input_dim, outputs=[block_3_out, block_4_out, block_5_out], name='resnet_50')
    return model


def head_of_model(output_filters, bias_init):
    """
    xay dung dau ra cho du doan class va bbox
    """
    head = tf.keras.Sequential([tf.keras.layers.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=kernel_init))
        head.add(tf.keras.layers.ReLU())
    head.add(tf.keras.layers.Conv2D(output_filters, 3, 1, padding='same', kernel_initializer=kernel_init,
                                    bias_initializer=bias_init))
    return head


