from PIL import Image
import glob
import numpy as np


image_list = []
for filename in glob.glob('E:\Test-programing\JIA\JIA_github_repo\JIA\Images\*.*'):
    im=np.asarray(Image.open(filename))
    print(im.shape)

