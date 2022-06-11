import read
import knee_label
import knee_aug
import knee_codding
import knee_model
import knee_kfold


if __name__ == '__main__' :

    # read.read_img()
    
    # knee_label.names_excl()
    # if not knee_label.labeled():        
    #     exit()

    # knee_aug.augmentation()
    code_imgs, code_lbl = knee_codding.codding()
    # model = knee_model.model_keras()
    # knee_kfold.model_kfold(model, code_imgs, code_lbl)
    print(100 * '.')


