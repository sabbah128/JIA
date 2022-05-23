import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def read_img():
    
    path = '.\\All_Images'
    folder_img = 'Images'
    names, img_path = [], []

    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.IMA' in f:
                names.append(f.split('.')[0])
                img_path.append(root+"\\"+f)

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
    print('read images are done.')

def img_edit():
    char = input('did you separate and edit knee photos ? (Y/N): ')
    if char.lower() == 'y':
        print('Manually separate & edit knee photos (crop, resize or remove) is done.')
        return True
    else:
        print('This section must be checked manually, please proceed.')
        return False