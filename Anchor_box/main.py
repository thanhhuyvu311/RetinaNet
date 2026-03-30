import tensorflow as tf

class Anchor_box:
    """
    tao anchor box cho cac ban do dac trung tai cac strides [8,16,32,64,128]
    box co thuoc tinh [x,y,w,h]

    aspect_ratios (ti le khung hinh): la 1 list so thuc dai dien cho ti le
    khung hinh cua cac achor box at moi vi tri tren ban do dac trung

    scales (ti le): la 1 list so thuc dai dien cho ti le cua cac anchor box
    tai moi vi tri tren ban do dac trung

    num_anchors: The number of anchor boxes at each location on feature map

    areas (dien tich): la 1 list so thuc dai dien cho dien tich cua anchor box
    cho moi ban do dac trung tren kim tu thap dac trung

    strides: la 1 list so thuc dai dien cho cac buoc nhay tren moi ban do dac trung
    tren kim tu thap dac trung
    """

    def __init__(self):
        #self.aspect_ratios = [0.5,1.0,2.0]
        self.aspect_ratios = [0.75,1.0,1.3]
        self.scales = [2 ** x for x in [0,1/3,2/3]]
        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3,8)]
        #self._areas = [x ** 2 for x in [32.0,64.0,128.0,256.0,512.0]]
        self._areas = [x ** 2 for x in [2.0, 4.0, 8.0, 10.0, 12.0]]
        #self._areas = [x ** 2 for x in [1.8, 2.8, 4.8, 6.0, 8.0]]
        #self._areas = [x ** 2 for x in [8.0, 16.0, 24.0, 36.0, 64.0]]
        self._anchor_dims = self._compute_dims()
    def _compute_dims(self):
        """
        tinh chieu cua cac anchor box cho tat ca ti le tai tat ca cap cua FPN
        :return:
        """

        anchor_dims_all = []

        for area in self._areas:

            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_h = tf.math.sqrt(area/ratio)
                anchor_w = area/anchor_h
                dims = tf.reshape(
                    tf.stack([anchor_w,anchor_h],axis=-1),[1,1,2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale*dims)
            anchor_dims_all.append(tf.stack(anchor_dims,axis=-2))
        return anchor_dims_all
    def _get_anchors(self,feature_h,feature_w,level):
        """
        tao anchor box cho ban do dac trung da cho size va level
        :param feature_h: chieu cao cua ban do dac trung
        :param feature_w: chieu rong cua ban do dac trung
        :param level: level cua ban do dac trung tren kim tu thap dac trung
        :return:
        anchor boxes voi kich thuoc
        '(feature_h * feature_w * num_anchors,4)'
        """
        ry = tf.range(feature_w, dtype=tf.float32) + 0.5 # cong 0.5 de co the lay dc tam chinh giua, vi du tai y = 1 thi center y = 1.5
        rx = tf.range(feature_h, dtype=tf.float32) + 0.5

        X, Y = tf.meshgrid(ry, rx) # ghep ry,rx lai thanh 1 luoi ma tran

        centers = tf.stack([X, Y], axis=-1) * self._strides[level - 3] #dua toa do tai cac ban do dac trung thanh toa do goc cua anh
        centers = tf.expand_dims(centers,axis=-2)
        centers = tf.tile(centers,[1,1,self._num_anchors,1])
        dims = tf.tile(
            self._anchor_dims[level-3],[feature_h,feature_w,1,1]
        ) #tao 9 anchor_box tai tam nay

        anchors = tf.concat([centers,dims],axis=-1) #ghep tam va kich thuoc
        #center = [x,y,x,y]
        #dims = [w,h,w,h]
        # => anchor box co kich thuoc [x,y,w,h] voi x,y la toa do goc cua anh, w,h la kich thuoc cua anh

        return tf.reshape(
            anchors,[feature_h*feature_w*self._num_anchors,4]
        )

    def get_anchors(self,img_h,img_w):
        """
        tao tat ca anchor box cho tat ca ban do dac trung tren kim tu thap dac trung
        :param img_h: chieu cao anh
        :param img_w: chieu rong anh
        :return:
        cac anchorbox cho tat ca ban do dac trung, duoc sap xep chong tat ca len nhau thanh 1 tensor duy nhat
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(img_h/2 ** i),
                tf.math.ceil(img_w/2 ** i),
                i,
            )
            for i in range(3,8)
        ]
        return tf.concat(anchors,axis=0)

