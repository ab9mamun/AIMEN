import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
def print_metrics(Y_plain, Y_pred_plain):
    """
    Prints classification results in different metrics. For now, it only prints accuracy. Eventually, it will print f-1 scores, recall, precision, etc.
    :param Y_plain:
    :param Y_pred_plain:
    :return:
    """
    #print('True:', Y_plain)
    #print('Pred:', Y_pred_plain)

    acc = np.count_nonzero(Y_pred_plain == Y_plain) / len(Y_plain)
    pr, rec, f1, sup = precision_recall_fscore_support(Y_plain, Y_pred_plain, average='macro')
    print(f'Accuracy: {acc}, Precision: {pr}, Recall: {rec}, F1: {f1}')

def get_metrics_txt(Y_plain, Y_pred_plain):
    """
    Prints classification results in different metrics. For now, it only prints accuracy. Eventually, it will print f-1 scores, recall, precision, etc.
    :param Y_plain:
    :param Y_pred_plain:
    :return:
    """
    #print('True:', Y_plain)
    #print('Pred:', Y_pred_plain)
    report = classification_report(Y_plain, Y_pred_plain, output_dict=True)
    acc = report["accuracy"]
    sens = report["1"]["recall"]
    spec = report["0"]["recall"]
    PPV = report["1"]["precision"]
    NPV = report["0"]["precision"]
    F1plus = report["1"]["f1-score"]
    F1minus = report["0"]["f1-score"]
    avgF1 = report["macro avg"]["f1-score"]
    return f'{acc},{sens},{spec},{PPV},{NPV},{F1plus},{F1minus},{avgF1}'

def common_confmat_plot(Y_plain, Y_pred_plain, title, confmat_filename, trainval_or_test='train'):
    confmat = confusion_matrix(Y_plain, Y_pred_plain)  # turn this into a dataframe
    matrixdf = pd.DataFrame(confmat)  # plot the result
    matrix = matrixdf.to_numpy()

    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.0)
    minval = np.min(matrix)
    maxval = np.max(matrix)
    diagonal_mask = np.eye(*matrix.shape, dtype=bool)
    # plot the main diagonal
    sns.heatmap(matrixdf, annot=True, fmt='g', mask=~diagonal_mask, cmap='Blues', vmin=minval, vmax=maxval)
    # plot the off diagonal
    sns.heatmap(matrixdf, annot=True, fmt='g', mask=diagonal_mask, cmap='Reds', vmin=minval, vmax=maxval,
                cbar_kws=dict(ticks=[]))
    plt.title(title)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)
    plt.xticks([0.5, 1.5], labels=['Normal', 'Abnormal'])
    plt.yticks([0.5, 1.5], labels=['Normal', 'Abnormal'])
    plt.savefig(confmat_filename)
    plt.close()

    print(f"Printing {trainval_or_test} classification metrics")
    print(classification_report(Y_plain, Y_pred_plain))
    return classification_report(Y_plain, Y_pred_plain, output_dict=True)

def plot_the_confusion_matrix(Y_plain, Y_pred_plain, output_folder,datestamp, title, trainval_or_test='train'):
    confmat_filename = f'{output_folder}/Confusion_matrix_{trainval_or_test}_{datestamp}.png'
    return common_confmat_plot(Y_plain, Y_pred_plain, title, confmat_filename, trainval_or_test)


def plot_the_confusion_matrix_for_test(Y_plain, Y_pred_plain, output_folder,train_datestamp, now_timestamp, title, trainval_or_test='train'):
    confmat_filename = f'{output_folder}/Confusion_matrix_{trainval_or_test}_CLF_Trained_{train_datestamp}_Results_saved_{now_timestamp}.png'
    return common_confmat_plot(Y_plain, Y_pred_plain, title, confmat_filename, trainval_or_test)




def plot_learning_curve(history, output_folder, datestamp, outcome_name, fold, model_type):
    plt.figure()
    indices = [i for i in range(1, len(history.history['loss']) + 1)]
    plt.plot(indices, history.history['loss'])
    plt.plot(indices, history.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'])
    plt.title(f'Learning curves of {outcome_name} with {model_type} - fold {fold+1}')
    plt.savefig(f'{output_folder}/Learning_curve_{outcome_name}_fold_{fold+1}_{datestamp}.png')
    plt.close()


def get_datestamp():
    return str(datetime.now()).replace(" ", " ").replace(":", "-")

def get_auc(Y_plain, Y_pred_raw):
    try:
        return roc_auc_score(Y_plain, Y_pred_raw)
    except ValueError:
        return 0.0

def get_roc_metrics_txt(model_type,data_tag, Y_plain, Y_pred_raw):
    try:
        txt = 'model_type,data_tag,threshold,accuracy,sensitivity,specificity,PPV,NPV,F1+,F1-,Avg F1\n'
        for t in range(0, 11, 1):
            threshold = t / 10.0

            Y_pred = np.array([1 if x >= threshold else 0 for x in Y_pred_raw]).astype(int)
            #if t == 0 or t == 10:
            #    print('threshold:', threshold)
            #    print(Y_pred)
            txt +=f'{model_type},{data_tag},{threshold},{get_metrics_txt(Y_plain, Y_pred)}\n'

        return txt
    except ValueError:
        return ''