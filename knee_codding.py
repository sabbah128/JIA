import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array


def codding():

    folder_read = '.\\knee_augmentation\\'
    imgs = []
    lbl = []
    dict_lbl = {
            '00': [1, 0, 0, 0],
            '01': [0, 1, 0, 0],
            '10': [0, 0, 1, 0],
            '11': [0, 0, 0, 1]}

    for images in os.listdir(folder_read):
        img = Image.open(folder_read + images)
        imgs.append(img_to_array(img))
        lbl.append(dict_lbl[images.split('.')[-2]])

    imgs_code = np.array(imgs)
    lbl_code = np.array(lbl)
    
    print('image shape : ', imgs_code.shape)
    print('label shape : ', lbl_code.shape)
    print('knee_codding is done.')
    return imgs_code, lbl_code

