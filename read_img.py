# 1
import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


path = '.\\All_Images'
names, img_path = [], []

for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.IMA' in f:
            names.append(f.split('.')[0])
            img_path.append(root+"\\"+f)

folder_img = 'Images'
if not os.path.isdir(folder_img):
    os.mkdir(folder_img)

for i in range(len(img_path)):
    img = pydicom.dcmread(img_path[i])
    img = img.pixel_array
    img_shape = img.shape

    if img_shape == (2, 128, 128) or img_shape == (2, 256, 256):
        # img[0, :, ::-1] => mirror
        plt.imsave(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg', img[0, :, ::-1], cmap=cm.gray)
        plt.imsave(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg', img[1], cmap=cm.gray)
        if img_shape == (2, 256, 256):
            im1 = Image.open(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg')
            im1 = im1.resize((128, 128))
            im1.save(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg', mode='L')

            im2 = Image.open(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg')
            im2 = im2.resize((128, 128))
            im2.save(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg', mode='L')
    else:
        img_path[i] = 'null'
