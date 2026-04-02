import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
read_csv = pd.read_csv('/home/huy/Documents/de_tai_tot_nghiep/Drone Thermal.v4i.voc/csv_file/data_information_grouped_thermal_drone.csv')

img_path = read_csv['path_img'].tolist()
TARGET_SIZE = 256
for i in range(len(img_path)):
    img = Image.open(img_path[i])
    img_resize = img.resize((TARGET_SIZE, TARGET_SIZE))
    fig,ax = plt.subplots(1,1,figsize = (12,10))
    ax.imshow(img_resize)
    plt.show()
