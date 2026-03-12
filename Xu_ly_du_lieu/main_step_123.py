import xml.etree.ElementTree as ET
import os, glob
import pandas as pd

if __name__ == '__main__':
    data_dir = '/home/huy/Documents/de_tai_tot_nghiep/object_detect'
    anno_dir = os.path.join(data_dir, 'anno')  # lay dia chi thuc muc anno
    img_dir = os.path.join(data_dir, 'imgs')

    # khoi tao dic gan nhan background voi id 0
    class_mapping = {'background': 0}

    # tao 1 mang de chua cac gia tri sau khi doc ra vao day
    xml_list = []

    # lay dia chi file xml trong anno
    for xml_file in glob.glob(anno_dir + '/*.xml'):
        # lay vi tri cua file xml tren o dia
        tree = ET.parse(xml_file)
        # lay nut goc
        root = tree.getroot()

        # lay kich thuoc anh
        width_img = int(root.find('size/width').text)
        height_img = int(root.find('size/height').text)
        channel_img = int(root.find('size/depth').text)
        path_img = os.path.join(img_dir, root.find('filename').text)  # lay dia chi cung tung anh

        # Khoi tao danh sach rong cho tung anh de chua bndbox va class_id
        bboxes_of_img = []
        class_ids_of_img = []

        # truy cap vao phan tu object de doc nhan, bbox
        for member in root.findall('object'):
            # tim ten class cua tung object
            class_name = member.find('name').text
            class_id = class_mapping.setdefault(class_name, len(class_mapping))

            bndbox = member.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w / 2.0
                y = ymin + h / 2.0

                # BỘ LỌC CHẶN BÓNG MA: Chi lay nhung box co width va height > 0
                if w > 0 and h > 0:
                    bboxes_of_img.append([x, y, w, h])
                    class_ids_of_img.append(class_id)

        # Sau khi quet xong tat ca object, luu thong tin cua buc anh vao list.
        # Neu anh background (khong co object hoac object loi), bboxes_of_img se la []
        value = {
            'path_img': path_img,
            'bbox': bboxes_of_img,
            'class_id': class_ids_of_img
        }
        xml_list.append(value)

    # Tao DataFrame truc tiep tu list da duoc gom nhom
    grouped_df = pd.DataFrame(xml_list)

    # luu file csv
    os.chdir(data_dir + '/csv_file/')
    grouped_df.to_csv('data_information_grouped.csv', index=False)
    print("Xong! Đã dọn dẹp bóng ma và hỗ trợ ảnh Background.")