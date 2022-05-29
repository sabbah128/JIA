import os
from PIL import Image


for img in os.listdir('unShape'):
    im = Image.open('unShape\\'+img)
    im = im.resize((128, 128))
    im.save('biz\\'+img)
    

    