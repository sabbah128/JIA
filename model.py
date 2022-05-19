import numpy as np
# import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers import BatchNormalization
from sklearn.metrics import roc_auc_score, auc
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Flatten, MaxPool2D
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def model_keras(imgs, lbl):
    print('run model_keras')

    dropout = 0.4
    n_splits = 2
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    a, p, r, acc, cm = [], [], [], [], []
    prs, stdprs, aucs, tprs = [], [], [], []
    mean_recall = np.linspace(0, 1, 100)
    mean_fpr = np.linspace(0, 1, 100)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(Dropout(dropout))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(units=250, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    i = 1
    for train, test in kfold.split(imgs):
        print('fold %i ...' %i)
        train_X, test_X = imgs[train], imgs[test]
        train_y, test_y = lbl[train], lbl[test]
        model.fit(train_X, train_y, batch_size=128, epochs=15, shuffle=True, verbose=2)  
        # scores = model.evaluate(test_X, test_y, verbose=0)
        # print(scores) 
        yhat = model.predict(test_X)
        fpr, tpr, _ = roc_curve(test_y[:, 1], yhat[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax1.plot(fpr, tpr, lw=2, alpha=0.5)
        precision, recall, _ = precision_recall_curve(test_y[:, 1], yhat[:, 1])
        prs.append(np.interp(mean_recall, precision, recall))
        pr_auc = auc(recall, precision)
        stdprs.append(pr_auc)
        ax2.plot(recall, precision, lw=2, alpha=0.5)
        i += 1

    # ROC
    ax1.plot([0,1],[0,1], linestyle = '--',lw = 1, color = 'red', label='No Skill', alpha=0.6)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax1.plot(mean_fpr, mean_tpr, color='navy', label=r'Mean ROC (AUC = %0.3f$\pm$%dropoutf)' % (mean_auc, std_auc), lw=3, alpha=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC')
    ax1.legend(loc="best")

    # Recall - Precision
    ax2.plot([0, 1], [0, 0], linestyle='--', lw = 1, color='red', label='No Skill', alpha=0.6)
    mean_precision = np.mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_pr = np.std(stdprs)
    ax2.plot(mean_precision, mean_recall, color='navy', label=r'Mean (AUC = %0.3f$\pm$%dropoutf)' % (mean_auc, std_pr), lw=3)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Recall vs Precision')
    ax2.legend(loc = 'best')

    plt.show()