from sklearn.model_selection import train_test_split
import pandas as pd
import os
if __name__ == "__main__":
    data_dir = '/home/huy/Documents/de_tai_tot_nghiep/classification/FLIR_ADAS_v2.v1i.multiclass'
    csv_file = os.path.join(data_dir,'train')

    read_csv = pd.read_csv(csv_file+'/_classes.csv')
    #lay tat ca ten trong path_img vao 1 mang
    unique_file = read_csv['filename'].unique()

    #chia data train, valid, test

    train_data,temp_data = train_test_split(unique_file,test_size=0.3,random_state=33)  #chia tap train thanh 70% va tap temp 30% (tu tap data unique)
    valid_data,test_data = train_test_split(temp_data,test_size=1/3,random_state=33) #chia tap valid 20% tap test 10% (tu tap temp)
    train_df = read_csv[read_csv['filename'].isin(train_data)]
    valid_df = read_csv[read_csv['filename'].isin(valid_data)]
    test_df = read_csv[read_csv['filename'].isin(test_data)]

    print(f"Tong so anh: {len(unique_file)}")
    print(f"so anh trong tap train: {len(train_data)}")
    print(f"so anh trong tap test: {len(test_data)}")
    print(f"so anh trong tap valid: {len(valid_data)}")

    train_df.to_csv(csv_file+'/train_data.csv',index=False)
    valid_df.to_csv(csv_file+'/valid_data.csv',index=False)
    test_df.to_csv(csv_file+'/test_data.csv',index=False)

    print('Chia data thanh cong')