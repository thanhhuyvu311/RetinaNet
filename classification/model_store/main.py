import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from classification.lib import plot_history,read_img_to_tensor,predict_and_plot_for_multic_label,visualize_img_multi_label,confuse_matrix
"""
this main evaluate model
"""
if __name__ == '__main__':

    model_path = 'model_modified_data.keras'
    history_path = '/home/huy/Documents/de_tai_tot_nghiep/classification/history_store/model_modified_data.csv'
    model =  tf.keras.models.load_model(model_path)
    #doc file csv chua ten nhan
    test_df = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test/test_data_modified.csv')
    test_df.columns = test_df.columns.str.strip() #xoa khoang trang

    #lay danh sach ten cac nhan
    columns = list(test_df.columns)
    #print(columns)
    class_name = columns[1:] #bo cot filename
    print(class_name)

    #set seed
    #tf.random.set_seed(44)


    #model.evaluate(test_data)
    #plot_fig = plot_history(history_path)

    img_path = visualize_img_multi_label('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test')
    test_img=read_img_to_tensor('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test/'+img_path)
    prediction = model.predict(test_img)
    #print(prediction[0])
    predict_and_plot_for_multic_label(prediction,class_name,'/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test/'+img_path)

    # chuan hoa anh
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    # confuse metrix
    # dua data lai thanh batch
    test_data = test_data_gen.flow_from_dataframe(dataframe=test_df,
                                                  directory='/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test',
                                                  # duong dan chua file anh
                                                  x_col='filename',  # ten cot chua file anh
                                                  y_col=class_name,  # ten cot chua nhan
                                                  batch_size=32,
                                                  target_size=(224, 224),
                                                  class_mode='raw',
                                                  # vi da danh nhan trong csv nen su dung class_mode la raw
                                                  seed=44)
    confuse_matrix(test_data,model)




