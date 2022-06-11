from PIL import Image
import os


folder_img = '.\\knee'
imgs = os.listdir(folder_img)

for img in imgs:
    im = Image.open(folder_img + '\\' + img)
    if im.size != (128, 128):
        im = im.resize((128, 128))
        im.save(folder_img + '\\' + img, mode='L')
