import os
import pandas as pd


def names_excl():
    
    folder_dir = '.\\knee\\'
    names = []

    for images in os.listdir(folder_dir):
        names.append(images.split('.')[0].lower())

    names = list(set(names))
    names.sort()

    df = pd.DataFrame(names, columns=['names'])
    writer = pd.ExcelWriter('knee_name.xlsx')
    df.to_excel(writer, sheet_name='knee_names', index=False)
    writer.save()
    print('convert names to excel (knee_label.xlsx) are done.')

def labeled():
    char = input('did you labeled knee photos ? (Y/N): ')
    if char.lower() == 'y':
        print('Manually labeled knee photos are done.')
        return True
    else:
        print('This section must be checked manually, please proceed.')
        return False
