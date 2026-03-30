import os
import pathlib
from hmac import digest_size
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def read_class():
    train_dir = pathlib.Path("train")
    class_name = np.array(sorted([item.name for item in train_dir.glob("*")]))  #tao 1 list nhan trong folder
    return class_name

def visualize_img_multi_label(target_dir):
    #chon folder class muon hien thi anh
    target_folder = target_dir
    #lay random anh trong folder do
    random_img = random.sample(os.listdir(target_folder),1)
    print(random_img)
    #doc anh va hien thi anh
    img = mpimg.imread(target_folder+"/"+random_img[0])
    plt.imshow(img)
    plt.axis("off")
    print(f"Img shape : {img.shape}")
    plt.show()
    return random_img[0]

def plot_history(history):

    history_df = pd.read_csv(history)

    fig,ax = plt.subplots(2,2, figsize = (10,7))

    epochs = range(len(history_df['loss']))
    #--accuracy--
    accuracy = history_df['accuracy']
    accuracy_val = history_df['val_accuracy']
    #--loss--
    loss = history_df['loss']
    loss_val = history_df['val_loss']
    #--precision--
    precision_tr = history_df['precision']
    precision_val = history_df['val_precision']
    #--recall--
    recall_tr = history_df['recall']
    recall_val = history_df['val_recall']
    #accuracy
    ax[0,0].plot(epochs,accuracy,label = 'accuracy_train')
    ax[0,0].plot(epochs,accuracy_val, label='val_accuracy')
    ax[0,0].set_title('ACCURACY')
    ax[0,0].legend()
    #loss
    ax[0,1].plot(epochs,loss,label='loss')
    ax[0,1].plot(epochs,loss_val,label='val_los')
    ax[0,1].set_title('LOSS')
    ax[0,1].legend()
    #precision
    ax[1,0].plot(epochs,precision_tr,label='precision')
    ax[1,0].plot(epochs,precision_val,label='val_precision')
    ax[1,0].set_title('PRECISION')
    ax[1,0].legend()
    #call
    ax[1,1].plot(epochs,recall_tr,label ='recall')
    ax[1,1].plot(epochs,recall_val,label ='val_recall')
    ax[1,1].set_title('RECALL')
    ax[1,1].legend()
    plt.tight_layout()

def read_img_to_tensor(img_path):
    #read img
    img = tf.io.read_file(img_path)
    #decode img to tensor
    img = tf.image.decode_image(img)
    #resize
    img = tf.image.resize(img,[224,224])
    #nchuan hoa anh
    img = img/255.

    img = tf.expand_dims(img,axis=0) #tang dims cua anh nay len thanh 1 batch cho giong voi dau vao
    return img

def predict_and_plot_for_multic_label(array_prob_of_predict,class_name,img):
    """
    truyen 1 mang gia tri xs dang vao ham nay
    truyen class name vao day
    """
    turn_the_array_to_1d = array_prob_of_predict[0]

    #create a list to store predicted class
    class_list = []
    #tao list de luu chuoi hien thi tren title (nhan + xac suat)
    display_text_list = []
    for index in range(len(turn_the_array_to_1d)):
        prob = turn_the_array_to_1d[index]
        if prob > 0.5:
            class_predicted = class_name[index]
            class_list.append(class_predicted)
            display_text_list.append(f"{class_predicted}: {prob:.4f}")
    print(class_list)

    read_img = mpimg.imread(img)
    plt.imshow(read_img)
    title_text = "Du doan:\n" + " | ".join(display_text_list)
    plt.title(title_text,color = 'green', fontsize = 12)
    plt.axis('off')
    plt.show()
def  confuse_matrix(test_data,model):
    test_data.reset()
    y_pred_probs = model.predict(test_data)
    # chuyen nhan thanh 0 1 dua tren nguong 0.5
    y_pred = (y_pred_probs > 0.5).astype(int)
    # lay nhan thuc
    y_true = np.array(test_data.labels)
    print(y_true)
    # duoi mang 2 chieu y_pred va y_true
    y_pred_flatten = y_pred.flatten()
    y_true_flatten = y_true.flatten()
    cm = confusion_matrix(y_true_flatten,y_pred_flatten)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])
    disp.plot()
    plt.show()