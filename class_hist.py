import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


labels = ['NN', 'NP', 'PN', 'PP']
knee_df = pd.read_excel('.\\knee_name.xlsx', header=0)
ankle_df = pd.read_excel('.\\ankle_name.xlsx', header=0)

x = np.arange(len(labels))
width = 0.3

men_means = [55, 8, 4, 63]
women_means = [53, 6, 9, 44]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Knee=121')
rects2 = ax.bar(x + width/2, women_means, width, label='Ankle=122')

ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend(loc='upper center')

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()