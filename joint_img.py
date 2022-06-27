import os
import pandas as pd


def set_label(joint):
    names = []

    for images in os.listdir(joint):
        names.append(images.split('.')[0].lower())

    names = list(set(names))
    names.sort()
    print('joint' , joint, ': ', len(names))

    df = pd.DataFrame(names, columns=['names'])
    writer = pd.ExcelWriter(joint+'_names.xlsx')
    df.to_excel(writer, sheet_name=joint+'_names', index=False)
    writer.save()
