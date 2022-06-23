import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


knee_num = []
ankle_num = []
labels = ['NN', 'NP', 'PN', 'PP']

knee_df = pd.read_excel('.\\knee_name.xlsx', header=0)
ankle_df = pd.read_excel('.\\ankle_name.xlsx', header=0)

knee_df = knee_df.values
ankle_df = ankle_df.values

NN =NP =PN =PP = 0
for i in range(len(knee_df)):
    if knee_df[i, 1]==0 and knee_df[i, 2]==0:
        NN += 1
    elif knee_df[i, 1]==0 and knee_df[i, 2]==1:
        NP += 1
    elif knee_df[i, 1]==1 and knee_df[i, 2]==0:
        PN += 1
    elif knee_df[i, 1]==1 and knee_df[i, 2]==1:
        PP += 1
knee_num = [NN, NP, PN, PP]


NN =NP =PN =PP = 0
for j in range(len(ankle_df)):
    if ankle_df[j, 1]==0 and ankle_df[j, 2]==0:
        NN += 1
    elif ankle_df[j, 1]==0 and ankle_df[j, 2]==1:
        NP += 1
    elif ankle_df[j, 1]==1 and ankle_df[j, 2]==0:
        PN += 1
    elif ankle_df[j, 1]==1 and ankle_df[j, 2]==1:
        PP += 1

ankle_num= [NN, NP, PN, PP]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, knee_num, width, label='Knee=%i' %np.sum(knee_num))
rects2 = ax.bar(x + width/2, ankle_num, width, label='Ankle=%i' %np.sum(ankle_num))

ax.set_ylabel('Number')
ax.set_title('Number of classes')
ax.set_xticks(x, labels)
ax.legend(loc='upper center')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("NumClass.jpg")
plt.show()