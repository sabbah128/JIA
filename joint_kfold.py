from statistics import mean
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def model_kfold(model, imgs, lbl): 

    n_split = 4
    epoch = 80

    kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 5))

    acc, err = [], []
    prs, stdprs, aucs, tprs = [], [], [], []
    acc_per_fold, loss_per_fold = [], []
    mean_recall = np.linspace(0, 1, 100)
    mean_fpr = np.linspace(0, 1, 100)
    fold_acc = np.linspace(1, epoch, epoch)

    fold_no = 1
    for train, test in kfold.split(imgs):
        train_X, test_X = imgs[train], imgs[test]
        train_y, test_y = lbl[train], lbl[test]
        history = model.fit(train_X, train_y, batch_size=256, epochs=epoch, shuffle=True, verbose=0)

        print(f'Training for fold {fold_no} ...')

        acc.append(history.history['accuracy'])
        ax1.plot(fold_acc, history.history['accuracy'], lw=2, alpha=0.5)

        err.append(history.history['loss'])
        ax2.plot(fold_acc, history.history['loss'], lw=2, alpha=0.5)
        
        scores = model.evaluate(test_X, test_y, verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        yhat = model.predict(test_X)
        fpr, tpr, _ = roc_curve(test_y[:, 1], yhat[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax3.plot(fpr, tpr, lw=2, alpha=0.5)

        precision, recall, _ = precision_recall_curve(test_y[:, 1], yhat[:, 1])
        prs.append(np.interp(mean_recall, precision, recall))
        pr_auc = auc(recall, precision)
        stdprs.append(pr_auc)
        ax4.plot(recall, precision, lw=2, alpha=0.5)

        fold_no += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]:.3f} - Accuracy: {acc_per_fold[i]:.3f}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold):.3f} (+- {np.std(acc_per_fold):.3f})')
    print(f'> Loss: {np.mean(loss_per_fold):.3f}')
    print('------------------------------------------------------------------------')

    # Accuracy
    ax1.plot([0,epoch],[0,1], linestyle = '--',lw = 1, color = 'red', label='No Skill', alpha=0.6)
    mean_acc = np.mean(acc, axis=0)
    ax1.plot(fold_acc, mean_acc, color='navy', label=r'Mean Acc = %.3f (%.2f)' 
            % (np.mean(acc_per_fold), np.std(acc_per_fold)), lw=3)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend(loc="best")

    # Loss
    mean_err = np.mean(err, axis=0)
    ax2.plot(fold_acc, mean_err, color='navy', label=r'Mean Loss (AUC = %.3f (%.2f)' 
            % (np.mean(loss_per_fold), np.std(loss_per_fold)), lw=3, alpha=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss fold')
    ax2.set_title('Loss')
    ax2.legend(loc="best")

    # ROC
    ax3.plot([0,1],[0,1], linestyle = '--',lw = 1, color = 'red', label='No Skill', alpha=0.6)
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax3.plot(mean_fpr, mean_tpr, color='navy', label=r'Mean ROC (AUC = %.3f/%.2f)' % (mean_auc, std_auc), lw=3)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC')
    ax3.legend(loc="best")

    # Recall - Precision
    ax4.plot([0, 1], [0, 0], linestyle='--', lw = 1, color='red', label='No Skill', alpha=0.6)
    mean_precision = np.mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_pr = np.std(stdprs)
    ax4.plot(mean_precision, mean_recall, color='navy', label=r'Mean (AUC = %.3f (%.2f))' % (mean_auc, std_pr), lw=3)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Recall vs Precision')
    ax4.legend(loc = 'best')
    plt.show()

    print('Kfold is done.')