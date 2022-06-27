import os
import random
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle


def train_test(joint, ratio=0.2):

    train_num= test_num = 0
    os.mkdir(joint+'_train')
    os.mkdir(joint+'_test')

    lbl = pd.read_excel('.\\'+joint+'_names.xlsx', header=0) 
    imgs = shuffle(os.listdir(joint))
    for img in imgs:
        indx = next(iter(lbl[lbl['names']==img.split('.')[0].lower()].index), 'no match')
        if indx == 'no match':
            print('no mathc '+ img.split('.')[0].lower())
            exit()
        else:
            r = lbl.iloc[int(indx), 1]
            l = lbl.iloc[int(indx), 2]
        
        image = Image.open('.\\'+joint+'\\'+ img)
        rnd = random.random()
        if rnd > ratio:
            image.save(joint+'_train'+'\\'+img.split('.')[0]+'.'+str(rnd)+'.'+str(r)+str(l)+'.jpg', mode='L')
            train_num += 1
        else:
            image.save(joint+'_test'+'\\'+img.split('.')[0]+'.'+str(rnd)+'.'+str(r)+str(l)+'.jpg', mode='L')
            test_num += 1

    print('number of Train images : ', train_num)
    print('number of Test images : ', test_num)




# def train_val_test(joint):
    # os.mkdir(joint+'_train')
    # os.mkdir(joint+'_val')
    # os.mkdir(joint+'_test')    
    # return joint


# lbl = pd.read_excel('.\\knee_name.xlsx', header=0)        
# names = []

# for images in os.listdir(folder_dir):
#     indx = next(iter(lbl[lbl['names']==images.split('.')[0].lower()].index), 'no match')
#     if indx == 'no match':
#         print('no mathc '+ images.split('.')[0].lower())
#         exit()
#     else:
#         r = lbl.iloc[int(indx), 1]
#         l = lbl.iloc[int(indx), 2]
#     rnd = np.ran
#     if rnd > test_ratio:
#         plt.imsave(knee_train+'\\'+images+'.'+images.split('.')[-2]+str(i)+'.'+str(r)+str(l)+'.jpg', image)
#     else:
#         plt.imsave(knee_test+'\\'+images+'.'+images.split('.')[-2]+str(i)+'.'+str(r)+str(l)+'.jpg', image)
#     print(images.split('.')[0]+'.'+str(r)+str(l))
#     print(rnd)
