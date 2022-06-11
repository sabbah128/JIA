import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def read_img():
    
    path = '.\\All_Images'
    # path = '.\\crop_knee'
    folder_img = 'Images'
    folder_unShape = 'unShape'
    names, img_path = [], []

    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.IMA' in f:
                names.append(f.split('.')[0])
                img_path.append(root+"\\"+f)

    if os.path.isdir(folder_img):
        print('folder was exist.')
        exit()
    else:
        os.mkdir(folder_img)

    if not os.path.isdir(folder_unShape):
        os.mkdir(folder_unShape)    

    for i in range(len(img_path)):
        img = pydicom.dcmread(img_path[i])
        img = img.pixel_array
        img_shape = img.shape

        if img_shape == (2, 128, 128) or img_shape == (2, 256, 256):
            # img[1, :, ::-1] => mirror            
            plt.imsave(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg', img[0], cmap=cm.gray)
            plt.imsave(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg', img[1, :, ::-1], cmap=cm.gray)
            if img_shape == (2, 256, 256):
                im1 = Image.open(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg')
                im1 = im1.resize((128, 128))
                im1.save(folder_img+'\\'+ names[i] + '.' + str(i) + '.1.jpg', mode='L')

                im2 = Image.open(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg')
                im2 = im2.resize((128, 128))
                im2.save(folder_img+'\\'+ names[i] + '.' + str(i) + '.2.jpg', mode='L')
        else:
            plt.imsave(folder_unShape+'\\'+ names[i] + '.' + str(i) + '.1.jpg', img[0], cmap=cm.gray)
            plt.imsave(folder_unShape+'\\'+ names[i] + '.' + str(i) + '.2.jpg', img[1, :, ::-1], cmap=cm.gray)

    print('read all images are done.')
