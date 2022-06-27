import os
from PIL import Image
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import random

def augmentation(joint):

    folder_save = joint+'_train_aug'
    if not os.path.isdir(folder_save):
        os.mkdir(folder_save) 

    for img in os.listdir(joint+'_train'):
        image = Image.open(joint+'_train\\' + img)
        data = img_to_array(image)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(height_shift_range=0.1, 
                                    width_shift_range=0.15,
                                    rotation_range=5, 
                                    # brightness_range=[0.6,1.0], 
                                    zoom_range=[0.8,1.1]
                                    )
        it = datagen.flow(samples, batch_size=1)

        for i in range(4):
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imsave(folder_save+'\\'+img.split('.')[0]+'.'+str(random.randint(0, 100000))+'.'+img.split('.')[-2]+'.jpg', image)