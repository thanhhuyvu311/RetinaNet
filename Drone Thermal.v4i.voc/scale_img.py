from PIL import Image
import os
import glob

base_dir = '/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc'

imgs_folder = os.path.join(base_dir,'imgs')

os.makedirs('resize_imgs',exist_ok=True)

img_path = glob.glob(os.path.join(imgs_folder,'*.jpg'))

for img in img_path:
    base_name = os.path.basename(img)
    img_ = Image.open(img)
    resize_img =img_.resize((256,256),Image.Resampling.LANCZOS)
    resize_img.save(os.path.join('resize_imgs',base_name))
