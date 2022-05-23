from multiprocessing.spawn import import_main_path
import read
import knee_label
import knee_aug
import knee_codding
import knee_keras
import knee_kfold

if __name__ == '__main__' :

    read.read_img()
    if not read.img_edit():
        exit()
    
    knee_label.names_excl()
    if not knee_label.labeled():        
        exit()

    knee_aug.augmentation()
    code_imgs, code_lbl = knee_codding.codding()
    m = knee_keras.model_keras()
    knee_kfold.model_kfold(m, code_imgs, code_lbl)



    



