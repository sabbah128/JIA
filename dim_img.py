import os
from PIL import Image
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



folder_save = 'knee_augmentation\\'
folder_dir = 'E:\\GitHub\\JIA\\knee\\'
names = []

for images in os.listdir(folder_dir):
    img = Image.open(folder_dir + images)
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(height_shift_range=0.2, 
                                rotation_range=10, 
                                brightness_range=[0.6,1.0], 
                                zoom_range=[0.5,1.0])
    it = datagen.flow(samples, batch_size=1)
    images = images.split('.')[0]
    if images not in names:
        ii = 0
        print(' >>>>>> ', images)
        while True:
            l = input('Left Label  (N=0, P=1) :')
            if (l in '01'):
                r = input('Right Label (N=0, P=1) :')
                if r in '01':
                    break
    else :
        ii = 1
    for i in range(9):
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imsave(folder_save+images+'.'+str(ii)+str(i)+'.'+str(r)+str(l)+'.jpg', image)
    ii = 0
    names.append(images)
