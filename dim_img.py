from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


img = load_img('E:\Test-programing\JIA\JIA_github_repo\JIA\Images\AHMADI_FATEMEH.1.1.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(height_shift_range=0.4, 
                            rotation_range=30, 
                            brightness_range=[0.2,1.0], 
                            zoom_range=[0.5,1.0])

it = datagen.flow(samples, batch_size=1)

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    # plt.imsave(str(i) + '.jpg', image)
    pyplot.imshow(image)

pyplot.show()










# loaded_images = list()
# for filename in listdir('Images'):
    # pixels = asarray(Image.open('Images/' + filename))
    # loaded_images.append(pixels)
# pixels = asarray(Image.open('E:\Test-programing\JIA\JIA_github_repo\JIA\Images\AHMADI_FATEMEH.1.1.jpg'))
# pixels = pixels.astype('float32')    
# print(pixels.shape)
# example of global centering (subtract mean)
# mean = pixels.mean()
# pixels -= mean
# mean, std = pixels.mean(), pixels.std()
# print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# pixels = (pixels - mean) / std
# mean, std = pixels.mean(), pixels.std()
# print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))

# im0 = Image.fromarray(pixels[:, :, 0])
# im1 = Image.fromarray(pixels[:, :, 1])
# im2 = Image.fromarray(pixels[:, :, 2])
# im0.show()
# im1.show()
# im2.show()