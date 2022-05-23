from statistics import mean
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def model_kfold(model, imgs, lbl):    
    print('Kfold is running...')

    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    a, p, r, acc, cm, scores = [], [], [], [], [], []
    prs, stdprs, aucs, tprs = [], [], [], []
    mean_recall = np.linspace(0, 1, 100)
    mean_fpr = np.linspace(0, 1, 100)

    i = 1
    for train, test in kfold.split(imgs):
        print('fold %i ...' %i)
        train_X, test_X = imgs[train], imgs[test]
        train_y, test_y = lbl[train], lbl[test]
        model.fit(train_X, train_y, batch_size=128, epochs=1, shuffle=True, verbose=2)
         
        score = model.evaluate(test_X, test_y, verbose=0)
        print('Accuracy is : ', score)
        scores.append(score)

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
    print('Average Accuracy : ', np.mean(scores))

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
    print('Kfold is done.')