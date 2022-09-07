from cgi import test
import csv
import pandas as pd
import numpy as np
import time
import warnings
import json
import random

from sklearn.feature_extraction.text import CountVectorizer
import dimension_reduce as dr

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifierF
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import shuffle
from model_measure_new import model_measure

warnings.filterwarnings('ignore')


DATA_PATH = "xxx"
TARGET = "Security_Issue_Full"

def data_transfer(X, y, T, l):
    global train_x
    global train_y
    global test_x
    global test_y
    
    train_x = X
    train_y = y
    test_x = T
    test_y = l


def main():
    '''
    TSE 21 replication package simple text classification
    '''
    train_data = json.load(open(DATA_PATH + f"data/train_project_all.json", 'r'))  # use train set + validation set
    test_data = json.load(open(DATA_PATH + f"data/test_project.json", 'r'))

    random.shuffle(train_data)
    # random.shuffle(test_data)

    train_content = [f"{_['Issue_Title']}. {_['Issue_Body']}" for _ in train_data]
    test_content = [f"{_['Issue_Title']}. {_['Issue_Body']}" for _ in test_data]

    # 1 for pos and 0 for neg
    train_label = [int(_[TARGET]) for _ in train_data]
    test_label = [int(_[TARGET]) for _ in test_data]

    test_id = [_["Issue_Url"] for _ in test_data]

    print("converting text into matrix ...")
    vectorizer = CountVectorizer(stop_words='english')  # converting to matrix
    train_content_matrix = vectorizer.fit_transform(train_content)
    test_content_matrix = vectorizer.transform(test_content)
    print("feature selection ...")
    train_content_matrix_dr, test_content_matrix_dr = dr.selectFromLinearSVC2(train_content_matrix, train_label, test_content_matrix)

    global train_x
    global test_x
    global train_y
    global test_y

    train_x = train_content_matrix_dr.toarray()
    train_y = train_label
    test_x = test_content_matrix_dr.toarray()
    test_y = test_label
    data_transfer(train_x, train_y, test_x, test_y)

    learners = ["RF", "NB", "MLP", "LR", "KNN"]  #  "RF", "NB","MLP","LR","KNN"
    # metrics_all = dict()
    
    for l in learners:
        if l == "RF":
            print("===============RF training===============")
            clf = RandomForestClassifier(oob_score=True, n_estimators=30)
            clf.fit(train_x, train_y)
            print(clf.classes_)
            predict = clf.predict(test_x)
            proba = clf.predict_proba(test_x)
            score = [_[1] for _ in proba]
            metrics, fpr, tpr = model_measure(test_label=test_y, pred=predict, pred_score=score, sample_id=test_id)
            print(metrics)
            # metrics_all[l] = metrics
            with open(DATA_PATH + f"baseline/{l}_metric.json", 'w') as f:
                json.dump(metrics, f, indent=4)
            
            test_results = list()
            for label, pred, prob, id_ in zip(test_y, predict, score, test_id):
                # convert to python datatype before save as a json
                test_results.append({"label": int(label), "predict": int(pred), "prob": float(prob), "Issue_Url": id_})
            with open(DATA_PATH + f"baseline/{l}_result.json", 'w') as f:
                json.dump(test_results, f, indent=4)


        elif l == "NB":
            print("===============NB training===================")
            clf = MultinomialNB()
            clf.fit(train_x, train_y)
            predict = clf.predict(test_x)
            proba = clf.predict_proba(test_x)
            score = [_[1] for _ in proba]
            metrics, fpr, tpr = model_measure(test_label=test_y, pred=predict, pred_score=score, sample_id=test_id)
            # metrics_all[l] = metrics
            print(metrics)
            with open(DATA_PATH + f"baseline/{l}_metric.json", 'w') as f:
                json.dump(metrics, f, indent=4)

            test_results = list()
            for label, pred, prob, id_ in zip(test_y, predict, score, test_id):
                test_results.append({"label": int(label), "predict": int(pred), "prob": float(prob), "Issue_Url": id_})
            with open(DATA_PATH + f"baseline/{l}_result.json", 'w') as f:
                json.dump(test_results, f, indent=4)


        elif l == "MLP":
            print("===============MLP training===============")
            # fail to run using the config from TSE 21 since our dataset is too large
            # we use the default setting in sklearn
            clf = MLPClassifier(max_iter=10)  # default setting

            clf.fit(train_x, train_y)
            predict = clf.predict(test_x)
            proba = clf.predict_proba(test_x)
            score = [_[1] for _ in proba]
            metrics, fpr, tpr = model_measure(test_label=test_y, pred=predict, pred_score=score, sample_id=test_id)
            # metrics_all[l] = metrics
            print(metrics)
            with open(DATA_PATH + f"baseline/{l}_metric.json", 'w') as f:
                json.dump(metrics, f, indent=4)

            test_results = list()
            for label, pred, prob, id_ in zip(test_y, predict, score, test_id):
                test_results.append({"label": int(label), "predict": int(pred), "prob": float(prob), "Issue_Url": id_})
            with open(DATA_PATH + f"baseline/{l}_result.json", 'w') as f:
                json.dump(test_results, f, indent=4)

        
        elif l == "LR":
            print("===============LR training===============")
            clf = LogisticRegression()
            clf.fit(train_x, train_y)
            # predicted = clf.predict(test_x)
            predict = clf.predict(test_x)
            proba = clf.predict_proba(test_x)
            score = [_[1] for _ in proba]
            metrics, fpr, tpr = model_measure(test_label=test_y, pred=predict, pred_score=score, sample_id=test_id)
            # metrics_all[l] = metrics
            print(metrics)
            with open(DATA_PATH + f"baseline/{l}_metric.json", 'w') as f:
                json.dump(metrics, f, indent=4)

            test_results = list()
            for label, pred, prob, id_ in zip(test_y, predict, score, test_id):
                test_results.append({"label": int(label), "predict": int(pred), "prob": float(prob), "Issue_Url": id_})
            with open(DATA_PATH + f"baseline/{l}_result.json", 'w') as f:
                json.dump(test_results, f, indent=4)

        elif l == "KNN":
            print("===============KNN training===============")
            clf = KNeighborsClassifier(n_jobs=32)
            clf.fit(train_x, train_y)
            print("begin prediction")
            # predict = clf.predict(test_x)
            proba = clf.predict_proba(test_x)
            print("end prediction")
            score = [_[1] for _ in proba]
            predict = [1 if s >= 0.5 else 0 for s in score]  # saving time
            metrics, fpr, tpr = model_measure(test_label=test_y, pred=predict, pred_score=score, sample_id=test_id)
            # metrics_all[l] = metrics
            print(metrics)
            with open(DATA_PATH + f"baseline/{l}_metric.json", 'w') as f:
                json.dump(metrics, f, indent=4)

            test_results = list()
            for label, pred, prob, id_ in zip(test_y, predict, score, test_id):
                test_results.append({"label": int(label), "predict": int(pred), "prob": float(prob), "Issue_Url": id_})
            with open(DATA_PATH + f"baseline/{l}_result.json", 'w') as f:
                json.dump(test_results, f, indent=4)
        

if __name__ == "__main__":
    main()