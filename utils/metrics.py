import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


def get_multiLabels_acc(y, y_pred):
    intersection = np.logical_and(y, y_pred)  
    union = np.logical_or(y, y_pred)  
    union_count = union.sum(axis=1) 
    union_count[union_count == 0] = 1
    accuracy_per_sample = intersection.sum(axis=1) / union_count
    overall_accuracy = accuracy_per_sample.mean()
    return round(overall_accuracy, 4)

def get_multi_labels_metrics(y, y_pre):
    y = y.cpu().detach().numpy()
    y_pre = y_pre.cpu().detach().numpy()
    ml_acc = get_multiLabels_acc(y, y_pre) * 100
    hanming_loss = round(metrics.hamming_loss(y, y_pre), 4)
    micro_f1 = round(metrics.f1_score(y, y_pre, average='micro'), 4) * 100
    macro_f1 = round(metrics.f1_score(y, y_pre, average='macro'), 4) * 100
    return ml_acc, hanming_loss, micro_f1, macro_f1

