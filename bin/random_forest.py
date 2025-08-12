import csv
import copy
import timeit
import json
import joblib
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scikitplot as skplt
import shap
import re
import argparse
import sklearn
from numpy import mean, std
from pathlib import Path
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to input data")
parser.add_argument("--output", help="path to result folder")
args = parser.parse_args()


"""insert path here where results will be saved"""
PATH = args.output

isExist = os.path.exists(PATH)
if not isExist:
    os.makedirs(PATH)
original_stdout = sys.stdout

small_stats={"accuracy": [], "sens": [], "spec": [], "precision": [], "recall": [], "f1-score": [], "mcc": [], "auc": []}
stats = {"accuracy": [], "sens": [], "spec": [], "precision": [], "recall": [], "f1-score": [], "mcc": [], "auc": [],
        "used_data": [], "random_state": [], "hyperparameters": [], "threshold": [], "target_variable":[]}

acc = []
sens = []
spec = []
prec = []
REcall = []
f1 = []
mcc = []
AUC = []

""" global Parameters """
SEED = None #206
THRESHOLD = 0.5
MIN_SAMPLES_LEAF = 8 #2
MIN_SAMPLES_SPLIT = 10 #10
N_ESTIMATORS = 1200
MAX_DEPTH = 6
CLASS_WEIGHT = "balanced"
TARGET_VARIABLE = "DEATH"

SAVE_FIGURES = "no" # put "yes" if you want pictures



def run_rf(number):
    """main function to run the training of the random forest"""

    data_path = "/Users/webermac/Projekte/feature_importance_all/data_2024/no_correlation_07_KKB.csv"
    stats["used_data"] = data_path
    # create data sets 
    x_train, x_test, y_train, y_test, columns, features = load_and_split_data(data_path)
    
    # get cardinality of the training set for features
    get_cardinality(x_train=x_train)

    # get predictions and trained model clf
    predicted, probas, clf, = train_rf(x_train, y_train, x_test, y_test, columns, number)
    y_test = (list(map(float, y_test)))

    """start feature importance methods"""
    get_shap(clf, x_train,number)
    permutation(clf, x_test, y_test,number)
    
    """get evaluation metrics """
    give_metrics(y_test, predicted, probas)




def load_and_split_data(data_path: str):

    data_full = pd.read_csv(data_path, delimiter="," ,decimal=".")
    print(data_full)
    """remove special characters from colum names"""
    data = data_full.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    data = data.drop(["Unnamed0"],axis=1) # "PSN" #"Unnamed0"
    #top 5 of all three methods: Lactate, Normoblasten, proBNP, Peep, thrombocytes
    data = data[[TARGET_VARIABLE, "labor_Lactat", "labor_Normoblasten", "labor_Thrombozyten", "beatmung_PEEP", "labor_proBNP"]]
    data = data.astype("float")
    
    data = data.dropna(subset=[TARGET_VARIABLE])
    data = data.dropna(thresh=data.shape[0] * 0.7, axis=1)
    
    """ remove target variable and create features"""
    features = data.drop(TARGET_VARIABLE, axis=1)
    columns = features.columns.values.tolist()

    """create labels"""
    label = data[TARGET_VARIABLE]
    labels = np.array(label).tolist()
    print("SEED is : ", SEED)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, stratify=labels ,test_size=0.3, random_state=SEED)
    
    x_train.to_csv(PATH + "train_split.csv")
    x_test.to_csv(PATH + "test_split.csv")

    """ imputed with pandas fill NaNs, because shapely will not know the feature name otherwise"""
    imputed_test = x_test.fillna(x_test.median())
    imputed_train = x_train.fillna(x_train.median())
    
    """scale the variables"""
    scaler = StandardScaler()
    scaled_test = scaler.fit_transform(imputed_test)
    scaled_train = scaler.fit_transform(imputed_train)
    
    scaled_test_df = pd.DataFrame(scaled_test, columns = imputed_test.columns)
    scaled_train_df = pd.DataFrame(scaled_train, columns= imputed_train.columns)
    

    """shuffle the labels"""
    #y_train = shuffle(y_train)

    return scaled_train_df, scaled_test_df, y_train, y_test, columns, features


def train_rf(x_train, y_train, x_test, y_test, columns, number):

    # create the random forest classifier
    clf = RandomForestClassifier(min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS, 
                                max_depth=MAX_DEPTH, class_weight=CLASS_WEIGHT)

    # train the rf on the training set
    clf.fit(x_train, y_train)

    # save the model
    joblib.dump(clf, PATH + "rf_model.sav")

    # get class probabilites for test set
    probas = clf.predict_proba(x_test)

    """it's possible to change the threshold for the class prediction here"""
    predicted = (probas[:, 1] >= THRESHOLD).astype("int")

    """ Gini impurity importance here (random forest intrinsic)"""
    importances = clf.feature_importances_

    forest_importances = pd.Series(importances, index=columns)
    forest_importance_pd = forest_importances.to_frame(name="importance_values")
    forest_importance_pd.sort_values(by=['importance_values'], ascending=False, inplace=True)
    forest_importance_pd.to_csv(PATH + "forest"+str(number)+".csv")

    return predicted, probas, clf

def get_cardinality(x_train: pd.DataFrame):
    counts = {}
    for col in x_train.columns:
        counts[col] = x_train[col].nunique()
    
    counts_sorted =  dict(sorted(counts.items(), key=lambda item: item[1]))
    counts_df = pd.DataFrame.from_dict(counts_sorted, orient="index")
    counts_df.to_csv(PATH + "cardinality_counts.csv")
    plt.figure(figsize=(18,8))
    plt.bar(list(counts_sorted.keys()), counts_sorted.values())
    plt.xticks(rotation="vertical")
    plt.ylabel("number of unique values")
    plt.savefig(PATH + "cardinality.svg", format="svg",bbox_inches="tight")
    plt.close()
    
    
    
def give_metrics(y_val: list, y_pred: list, probas):
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    spec.append(tn / (tn + fp))
    sens.append(tp / (tp + fn))
    acc.append(metrics.accuracy_score(y_val, y_pred))
    prec.append(metrics.precision_score(y_val, y_pred))
    REcall.append(metrics.recall_score(y_val, y_pred))
    f1.append(sklearn.metrics.f1_score(y_val, y_pred))
    mcc.append(sklearn.metrics.matthews_corrcoef(y_val, y_pred))
    AUC.append(sklearn.metrics.roc_auc_score(y_val, probas[:, 1]))


def get_shap(clf, x_train,number):
    """ shapley tree feature importance"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    shap_result = pd.DataFrame(shap_values[0], columns=x_train.columns)
    shap_result.to_csv(PATH + "shap"+str(number)+".csv")
    vals = np.abs(shap_values[0]).mean(0)
    feature_importance = pd.DataFrame(list(zip(x_train.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.to_csv(PATH + "shap_values"+str(number)+".csv")



def permutation(model, X_val, y_val,number):
    print("Calculating feature permutation")
    r = permutation_importance(model, X_val, y_val,
                            n_repeats=30,
                            random_state=None)
    permu_dic = {}
    for i in r.importances_mean.argsort()[::-1]:
        permu_dic[X_val.columns[i]] = r.importances_mean[i]
    permu_tmp_df  =  pd.DataFrame(permu_dic.items(), columns=["col_name", "importances"])
    permu_tmp_df.to_csv(PATH + "permu" + str(number)+".csv")



"""starts multiple runs of the random forest training"""
for number in range(1000):
    run_rf(number)
    stats["accuracy"] = acc
    stats["sens"] = sens
    stats["spec"] = spec
    stats["precision"] = prec
    stats["recall"] = REcall
    stats["f1-score"] = f1
    stats["mcc"] = mcc
    stats["auc"] = AUC
    stats["random_state"] = SEED
    stats["hyperparameters"] = [
        {"min_samples_leaf": MIN_SAMPLES_LEAF, "min_samples_split": MIN_SAMPLES_SPLIT, "n_estimators": N_ESTIMATORS,
        "class_weight": CLASS_WEIGHT}]
    stats["threshold"] = THRESHOLD
    stats["target_variable"] = TARGET_VARIABLE
    with open(PATH + "/meta.json", "w") as outfile:
        json.dump(stats, outfile)
    small_stats["accuracy"] = acc
    small_stats["sens"] = sens
    small_stats["spec"] = spec
    small_stats["precision"] = prec
    small_stats["recall"] = REcall
    small_stats["f1-score"] = f1
    small_stats["mcc"] = mcc
    small_stats["auc"] = AUC
    with open(PATH + "/small_meta.json", "w") as small_outfile:
        json.dump(small_stats, small_outfile)