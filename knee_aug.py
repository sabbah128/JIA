import os
from PIL import Image
import pandas as pd
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def augmentation():

    folder_save = 'knee_augmentation'
    if not os.path.isdir(folder_save):
        os.mkdir(folder_save)
    lbl = pd.read_excel('.\\knee_name.xlsx', header=0)        
    folder_dir = '.\\knee\\'
    names = []

    for images in os.listdir(folder_dir):
        indx = next(iter(lbl[lbl['names']==images.split('.')[0].lower()].index), 'no match')
        if indx == 'no match':
            print('no mathc '+ images.split('.')[0].lower())
            exit()
        else:
            l = lbl.iloc[int(indx), 2]
            r = lbl.iloc[int(indx), 1]

        img = Image.open(folder_dir + images)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(height_shift_range=0.1, 
                                    # width_shift_range=0.1,
                                    rotation_range=5, 
                                    brightness_range=[0.6,1.0], 
                                    zoom_range=[0.7,1.0])
        it = datagen.flow(samples, batch_size=1)

        for i in range(4):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(folder_save+'\\'+images+'.'+images.split('.')[-2]+str(i)+'.'+str(r)+str(l)+'.jpg', image)

        names.append(images)
    print('knee augmentation is done.')
