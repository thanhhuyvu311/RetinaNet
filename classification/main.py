import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall,Precision
from tensorflow.keras.callbacks import EarlyStopping
import lib
import pandas as pd
from resnet_50 import resnet_50_backbone
import os
"""
file main nay chay cho bai toan phan loai,
bai toan phan loai multi class
"""

if __name__ == '__main__':
    #img = mpimg.imread('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/train/video-2Af3dwvs6YPfwSSf6-frame-000300-YKDrgdzngkZkkPNcz_jpg.rf.7a90edc4fcd42dad44e40c8071059acf.jpg')
    #print(img.shape)
    #doc file csv chua ten nhan
    train_df = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/train/train_data_modified.csv')
    train_df.columns = train_df.columns.str.strip() #xoa khoang trang
    valid_df = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/valid/valid_data_modified.csv')
    valid_df.columns = valid_df.columns.str.strip() #xoa khoang trang

    #lay danh sach ten cac nhan
    columns = list(train_df.columns)
    #print(columns)
    class_name = columns[1:] #bo cot filename
    #print(class_name)

    #set seed
    tf.random.set_seed(44)

    #chuan hoa anh
    train_data_gen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=10, #xoay ngang 10 do
                                        width_shift_range= 0.1, #dich ngang 10%
                                        height_shift_range= 0.2, #dich doc 10 %
                                        zoom_range= 0.2, #zoom 20%
                                        horizontal_flip=True) #cho phep lat
    valid_data_gen = ImageDataGenerator(rescale=1./255)

    #dua data lai thanh batch
    train_data = train_data_gen.flow_from_dataframe(dataframe=train_df,
                                                    directory='/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/train', #duong dan chua file anh
                                                    x_col = 'filename', #ten cot chua file anh
                                                    y_col = class_name,#ten cot chua nhan
                                                    batch_size=32,
                                                    target_size=(224,224),
                                                    class_mode='raw', #vi da danh nhan trong csv nen su dung class_mode la raw
                                                    seed= 44)
    valid_data = valid_data_gen.flow_from_dataframe(dataframe=valid_df,
                                                    directory='/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/valid',
                                                    x_col='filename',
                                                    y_col=class_name,
                                                    batch_size=32,
                                                    target_size=(224,224),
                                                    class_mode='raw', #vi da danh nhan trong csv nen su dung class_mode la raw
                                                    seed=44)

    #use Renes50 model  
    
    model = resnet_50_backbone()
    model.summary()
    #compile model
    model.compile(
        loss = "binary_crossentropy",
        optimizer= tf.keras.optimizers.Adam(),
        metrics=["accuracy",Recall(name='recall'),Precision(name='precision')]
    )

    #callbacks
    early_stop = EarlyStopping(
        monitor= 'val_loss',
        min_delta= 0.0001,
        patience= 20,
        mode='min',
        restore_best_weights=True
    )
    """
    history = model.fit(
        train_data,
        epochs=200,
        steps_per_epoch=len(train_data),
        validation_data= valid_data,
        validation_steps= len(valid_data),
        callbacks=[early_stop]
    )

    history_df = pd.DataFrame(history.history)
    source_dir = '/home/huy/Documents/de_tai_tot_nghiep/classification'
    store_model_dir = os.path.join(source_dir,'model_store')
    store_history_dir = os.path.join(source_dir,'history_store')
    model.save(store_model_dir + '/model_modified_data.keras')
    history_df.to_csv(store_history_dir+'/model_modified_data.csv')
    """






