from sklearn import metrics
import numpy as np


def cal_f1(test_label, pred):
    TP, FN, TN, FP = [0,0,0,0]
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            FN += 1
        elif pred[i] == test_label[i] == 0:
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            FP += 1
    
    pd, pf, prec, f_measure = [0, 0, 0, 0]
    if TP + FN != 0:
        pd = TP / (TP + FN)
    if FP + TN != 0:
        pf = FP / (FP + TN)
    if TP + FP != 0:
        prec = TP / (TP + FP)
    if pd + prec != 0:
        f_measure = 2 * pd * prec / (pd + prec)
    
    m = {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "precision": prec, "recall": pd, "f1": f_measure}
    return m


def getArea(sbrs:np.ndarray)->float:
    # the last one is the total number
    totalSBRs = [-1]
    # ger ratio
    delta_y = 1 / len(sbrs)
    sbrs = sbrs / totalSBRs  # recall 

    area: float = 0
    for index in range(len(sbrs)):
        if index >= len(sbrs)-1:
            break
        else:
            area += 0.5 * (sbrs[index]+sbrs[index+1]) * delta_y
    return area


def model_measure(test_label, pred, pred_score, sample_id):
    # pred is the predicted label
    # test_label is the ground truth
    # pred_score is the predicted score
    # label for positive sample(CIR) is 1，label for negtive sample(NCIR) is 0
    print("num of testing sample:", len(test_label))
    init_index = [0, 0, 0, 0, 0, 0, 0, 0]
    TP, FN, TN, FP, pd, pf, prec, f_measure = init_index
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            FN += 1
        elif pred[i] == test_label[i] == 0:
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            FP += 1
    if TP + FN != 0:
        pd = TP / (TP + FN)
    if FP + TN != 0:
        pf = FP / (FP + TN)
    if TP + FP != 0:
        prec = TP / (TP + FP)
    if pd + prec != 0:
        f_measure = 2 * pd * prec / (pd + prec)
    if TP + TN + FN + FP !=0:
        success_rate = float((TP + TN) / (TP + TN + FN + FP))

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    ap = metrics.average_precision_score(test_label, pred_score, pos_label=1)  # area under precision recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(test_label, pred_score, pos_label=1)

    result = {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "pd&recall": pd, "prec": prec, "f1": f_measure, "ap": ap, "auc": auc}
    
    return result, fpr, tpr
