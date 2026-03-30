import os, glob, shutil, pathlib
import random
import numpy as np
def create_folder():
    """
    chay code nay de tao folder nhan cho tap test va valid

    :return:
    """

    data_dir = pathlib.Path('train/')
    name_dir = np.array(sorted([item.name for item in data_dir.glob("*")]))
    test_dir = 'test/'
    valid_dir = 'val/'
    for i in range(len(name_dir)):
        new_folder = os.path.join(test_dir, name_dir[i])
        os.makedirs(new_folder, exist_ok=True)
def move_file():
    """
    chay code nay de lay random anh tu thuc muc train sang folder test va valid
    :return:
    """
    ti_le_test = 0.1
    ti_le_valid = 0.3

    #dia chi anh se duoc chuyen toi
    test_dir = 'test/'
    valid_dir = 'val/'

    #lay folder chua anh
    data_dir = pathlib.Path('train/')
    #tao mang numpy chua ten cac nhan
    class_name = np.array(sorted([item.name for item in data_dir.glob("*")]))
    print(class_name, len(class_name))
    for i in range(len(class_name)):
        #lay duong dan vao cac folder nhan
        file = os.path.join(data_dir,class_name[i])
        #truy cap vao folder nhan de tim file anh
        read_file = os.path.join(file,"*.jpeg")
        #doc file anh
        num_file = glob.glob(read_file)
        #print(f"folder {class_name[i]} co {len(num_file)} anh")

        #xao tron anh
        random.shuffle(num_file)

        #tinh so luong anh can chuyen
        test_num = int(len(num_file) * ti_le_test)
        val_num = int(len(num_file) * ti_le_valid)

        #cat danh sach anh
        test_img = num_file[:test_num]
        val_img = num_file[test_num:test_num+val_num]
        #so luong anh sau khi chia
        print(len(test_img))
        print(len(val_img))

        #ghep duong dan
        dest_test = os.path.join(test_dir,class_name[i])
        dest_val = os.path.join(valid_dir, class_name[i])

        #di chuyen file
        for f in test_img:
            shutil.move(f,dest_test)

        for f in val_img:
            shutil.move(f,dest_val)

if __name__ == '__main__':
    move_file()