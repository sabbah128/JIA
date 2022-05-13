import os
from PIL import Image
import numpy as np



folder_dir = 'E:\\Test-programing\\JIA\\JIA_github_repo\\JIA\\knee_augmentation\\'
data = []
lbl = []

for images in os.listdir(folder_dir):
    img = Image.open(folder_dir + images)
    data.append(np.asarray(img))    
    if images.split('.')[-2] == '00':
        lbl.append([0, 0])
    elif images.split('.')[-2] == '01':
        lbl.append([0, 1])
    elif images.split('.')[-2] == '10':
        lbl.append([1, 0])
    else :
        lbl.append([1, 1])


lbl = np.array(lbl)
data = np.array(data)
print(data.shape, lbl.shape)