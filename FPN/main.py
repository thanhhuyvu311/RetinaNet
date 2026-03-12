import keras.layers
import tensorflow as tf
from resnet_50 import resnet_50_backbone

class FeaturePyramid(keras.layers.Layer):
    """
    xay dung dac trung kim tu thap tu cac ban do dac trung cua backbone
    thuoc tinh:
        so luong class : so luong class trong dataset
        khung xuong: khung xuong dc resnet 50
    class nay la con cua class Layer
    """
    def __init__(self,backbone = None, **kwargs):
        super().__init__(name='FeaturePyramid',**kwargs)
        self.backbone = backbone if backbone else resnet_50_backbone() #lay khung xuong
        #tao cac tich chap 1x1 de dua cac output ve cung 1 kich thuoc kenh (256) [c3,c4,c5]
        self.conv_c3_1x1 = keras.layers.Conv2D(256,1,1,padding='same')
        self.conv_c4_1x1 = keras.layers.Conv2D(256,1,1,padding='same')
        self.conv_c5_1x1 = keras.layers.Conv2D(256,1,1,padding='same')
        #tao cac tich chap 3x3 de lam min pixel sau khi phong to kich thuoc ban do dac trung nho = kich thuoc ban do dac trung lon
        self.conv_c3_3x3 = keras.layers.Conv2D(256,3,1,padding='same')
        self.conv_c4_3x3 = keras.layers.Conv2D(256,3,1,padding='same')
        self.conv_c5_3x3 = keras.layers.Conv2D(256,3,1,padding='same')
        #tao them lop tich chap c6 c7 de bat vat the to (giam kich thuoc tu c5)
        self.conv_c6_3x3 = keras.layers.Conv2D(256,3,2,padding='same')
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, padding='same')
        #upscale ban do dac trung len 2 lan
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training = False):

    #training = false la de no vao trang thai huan luyen, sau khi lap trinh xong va muon huan luyen mo hinh
    #model.fit se ghi de gia tri True len False nay.

        #lay dau ra c3 c4 c5
        c3_output,c4_output,c5_output = self.backbone(images,training = training)

        #thuc hien dong tac dong bo hoa so kenh
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)

        #thuc hien phep cong
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)

        #thuc hien lam min sau khi upscale
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)

        #thuc hien tich chap p6 p7
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

        return p3_output, p4_output, p5_output, p6_output, p7_output






