from typing import Optional, List, Dict, Any

import torch
from allennlp.training.metrics import Metric, metric
from overrides import overrides
import numpy as np
from sklearn import metrics

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
    prec, pd, f_measure = [0,0,0]
    if TP + FN != 0:
        # recall
        pd = TP / (TP + FN)
    if FP + TN != 0:
        pf = FP / (FP + TN)
    if TP + FP != 0:
        prec = TP / (TP + FP)
    if pd + prec != 0:
        f_measure = 2 * pd * prec / (pd + prec)
    
    m = {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "precision": prec, "recall": pd, "f1": f_measure}
    return m


def find_best_thres(test_label, pred_score, interval=[0.5, 0.9]):
    best_f1 = 0
    best_metric = None
    for thres in np.arange(interval[0], interval[1], 0.01):
        pred = list()
        for s in pred_score:
            if s >= thres:
                pred.append(1)
            else:
                pred.append(0)
        m = cal_f1(test_label, pred)
        # print(thres, m["f1"])
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            m["thres"] = thres
            best_metric = m

    return best_metric


@Metric.register("siamese_measure_v1")
class SiameseMeasureV1(Metric):
    def __init__(self, same_idx, thres=0.5) -> None:
        self._same_idx = same_idx  # same_idx
        self._thres = thres
        # self._result = dict()
        self._result = list()
    
    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 metadata: List[Dict[str, Any]] = None,
                 mask: Optional[torch.Tensor] = None):
        for probs, meta in zip(predictions.tolist(), metadata):
            label = 0 if meta["instance"][0]["label"] == "neg" else 1
            # id_ = meta["instance"][0]["Issue_Url"]
            score = probs[self._same_idx]
            self._result.append({"label": label, "prob": score})

    def get_metric(self, reset: bool):
        test_label = [_["label"] for _ in self._result]
        prob = [_["prob"] for _ in self._result]

        # metrics_pos = cal_f1(test_label, pred)
        metrics_pos = {"precision": 0, "recall": 0, "f1": 0, "thres": 0, "auc": 0, "ave_precision_score": 0}  # initialize
        if len(prob) == 0:
            # train
            return metrics_pos
        
        if reset:
            # when whole evaluation is done
            metrics_pos = find_best_thres(test_label, prob, interval=[0.5, 0.9])
            # metrics_pos["auc"] = metrics.roc_auc_score(test_label, prob)
            fpr, tpr, thresholds = metrics.roc_curve(test_label, prob, pos_label=1)
            metrics_pos["auc"] = metrics.auc(fpr, tpr)
            metrics_pos["ave_precision_score"] = metrics.average_precision_score(test_label, prob, pos_label=1)

        if reset:
            self.reset()
        
        return metrics_pos

    def reset(self) -> None:
        self._result.clear()