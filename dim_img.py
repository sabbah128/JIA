from os import listdir
from matplotlib import image


loaded_images = list()
for filename in listdir('Images'):
    img_data = image.imread('Images/' + filename)
    loaded_images.append(img_data)
    print(' %s %s' % (filename.split('.')[0], img_data.shape))
