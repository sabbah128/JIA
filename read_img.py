import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image


path='E:\\Test-programing\\JIA\\JIA_github_repo\\JIA\\dicom_img'

names, img_path = [], []


for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.IMA' in f:
            names.append(f.split('.')[0])
            img_path.append(root+"\\"+f)     

folder = 'Images\\'
for i in range(len(img_path)):
    img = pydicom.dcmread(img_path[i])
    img = img.pixel_array
    img_shape = img.shape

    if len(img_shape) == 3 and img_shape[0] == 2 and (img_shape[1] == 128 or img_shape[1] == 256):
        plt.imsave(folder + names[i] + str(i) + '1.jpeg', img[0], cmap=cm.gray)
        plt.imsave(folder + names[i] + str(i) + '2.jpeg', img[1], cmap=cm.gray)
        if img_shape[1] == 256:
            im1 = Image.open(folder + names[i] + str(i) + '1.jpeg')
            im2 = Image.open(folder + names[i] + str(i) + '1.jpeg')
            im1 = im1.resize((128, 128))
            im2 = im2.resize((128, 128))
            im1.save(folder + names[i] + str(i) + '1.jpeg')
            im2.save(folder + names[i] + str(i) + '2.jpeg')
    else:
        img_path[i] = 'null'
