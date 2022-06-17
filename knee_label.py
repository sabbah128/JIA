import os
import pandas as pd
# from collections import Counter


def names_excl():
    
    folder_dir = '.\\ankle\\'
    names = []

    for images in os.listdir(folder_dir):
        names.append(images.split('.')[0].lower())

    # print(Counter(names))
    print(len(names))    
    names = list(set(names))
    names.sort()
    print(len(names))

    df = pd.DataFrame(names, columns=['names'])
    writer = pd.ExcelWriter('ankle_name.xlsx')
    df.to_excel(writer, sheet_name='ankle_names', index=False)
    writer.save()
    print('convert names to excel (ankle_label.xlsx) are done.')

def labeled():
    char = input('did you labeled knee photos ? (Y/N): ')
    if char.lower() == 'y':
        print('Manually labeled knee photos are done.')
        return True
    else:
        print('This section must be checked manually, please proceed.')
        return False
