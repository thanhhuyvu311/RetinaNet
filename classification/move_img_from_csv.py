import os,glob,shutil
import pandas as pd

if __name__ == '__main__':
    #doc file csv
    test_csv = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/test/test_data.csv')
    valid_csv = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass/valid/valid_data.csv')

    #chuyen doi thanh dang set
    test_filenames = set(test_csv['filename'].tolist())
    valid_filenames = set(valid_csv['filename'].tolist())
    #print(test_filenames)
    #thu muc
    base_dir = '/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass'
    img_path = os.path.join(base_dir,'train')

    test_dir = os.path.join(base_dir,'test/')
    valid_dir = os.path.join(base_dir,'valid/')

    for file_path in glob.glob(os.path.join(img_path,'*.jpg')):
        #print(file_path)
        base_name = os.path.basename(file_path)
        #print(base_name)
        if base_name in test_filenames:
            new_path = os.path.join(test_dir,base_name)
            shutil.move(file_path,new_path)
            print('da chuyen anh sang test_folder')
        elif base_name in valid_filenames:
            new_path = os.path.join(valid_dir,base_name)
            shutil.move(file_path,new_path)
            print('da chuyen anh sang valid_folder')