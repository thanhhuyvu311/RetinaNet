import matplotlib.pyplot as plt
from hypothesis.extra.pandas import columns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/train/train_data.csv')
    train_df.columns = train_df.columns.str.strip() #xoa khoang trang

    columns = list(train_df.columns)
    #print(columns)
    class_name = columns[1:]
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,  # Xoay nhẹ 10 độ
        width_shift_range=0.1,  # Dịch ngang 10%
        height_shift_range=0.1,  # Dịch dọc 10%
        zoom_range=0.2,  # Phóng to/thu nhỏ 20%
        horizontal_flip=True,  # Cho phép lật ngang
        fill_mode='nearest'
    )

    train_data = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                   directory='/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/train',
                                                   x_col='filename',
                                                   y_col=class_name,
                                                   batch_size=10,#rut ra 10 tam
                                                   target_size=(224,224),
                                                   class_mode='raw',
                                                   seed=44)
    #rut 1 batch ra khoi train_data de check
    imgs,labels = next(train_data)
    fig,ax = plt.subplots(2,5,figsize = (10,5))
    ax = ax.flatten() #trai dai axis ra thanh 25
    for i in range(len(ax)):
        img = imgs[i]
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()