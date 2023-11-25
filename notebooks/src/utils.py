#!/usr/bin/env python
#coding=utf8
"""
Author: Raziel Amador Rios
Version: 0.0.1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import (RepeatedStratifiedKFold, train_test_split)
from sklearn.metrics import (roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score)
from xgboost import XGBClassifier

def print_training_results(scoring_dict:dict, input_model:XGBClassifier, cv_results:dict) -> None:
    """
    A convinient way to print the results after ML training
    Args:
        scoring_dict: dict with keys and values representing how you to want to print the results and how are represented in sklearn, respectively.
        input_model: XGBoost model.
        cv_results: dict from sklearn.model_selection.cross_validate.
    Returns:
        None
    """
    model_name = input_model.__class__.__name__
    print("-"*10)
    for i in range(len(scoring_dict)):
        mean_result_testset = cv_results['test_%s' % list(scoring_dict.values())[i]].mean()
        std__result_testset = cv_results['test_%s' % list(scoring_dict.values())[i]].std()
        print(f"{model_name} mean-{list(scoring_dict.keys())[i]}: {mean_result_testset:.4f} (+/- {std__result_testset: .2f})")
    print("-"*10)

def plot_AUROC(model:XGBClassifier, cv:RepeatedStratifiedKFold, X: pd.DataFrame, y:pd.DataFrame) -> None:
    """
    Plot AUROC (Area Under the ROC) for each cross-validation split and seed
    Args:
        model: input clf model
        cv: sklearn cross-validation Class
        X: training and test sets for the numeric and categorical features
        y: training and test sets for target clf
    Returns:
        None
    """
    # Plot configs
    sns.set_context("paper", font_scale= 2)
    plt.figure(figsize= (10, 10))
    # Vars
    mean_fpr = np.linspace(start=0, stop= 1, num=100)
    values_auroc = [] 
    values_tpr = []
    color_values= ["red", "green", "yellow"]
    i = 1
    # Plot each AUROC for each CV split
    for train, test in tqdm(cv.split(X, y.values.ravel())):
        fit_model = model.fit(X.iloc[train], y.loc[train].values.ravel())
        prediction = fit_model.predict_proba(X.iloc[test])[:,1]
        [fpr, tpr, _] = roc_curve(y.iloc[test].values.ravel(), prediction)
        roc_auc = auc(fpr, tpr)
        values_auroc.append(roc_auc)
        values_tpr.append(np.interp(mean_fpr, fpr, tpr)) #gives you an array
        if i <= 10:
            color = color_values[0]
        elif i > 10 and i <= 20:
            color = color_values[1]
        else:
            color = color_values[2]
        plt.plot(fpr, tpr, alpha= 0.2, color= color)
        i += 1
    # Mean AUROC
    mean_tpr = np.mean(values_tpr, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot([0,1], [0,1], linestyle= "--", color= "black")
    plt.plot(mean_fpr, mean_tpr, label= f"Mean AUROC= {mean_auc: .4f}", color= "black", lw= 2)
    # Plot std of mean AUROC
    std_tpr = np.std(values_tpr, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4, label=r'$\pm$ 1 std. dev.')
    # Plot configs
    plt.xlabel("False positive rate (1- Specificity)")
    plt.ylabel("True positive rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.tight_layout()

def _cm_percentage(input_cm: np.ndarray) -> np.ndarray:
    """
    Obtain the percetage of:
        1) TP, FP
        2) TN, FN
    """
    negative_values = np.round(input_cm.flatten()[0:2] / np.sum(input_cm.flatten()[0:2]) * 100, 2)
    positive_values = np.round(input_cm.flatten()[2:5] / np.sum(input_cm.flatten()[2:5]) * 100, 2)
    return np.array([negative_values, positive_values]).reshape(2, 2)

def _cm_labels(input_cm: np.ndarray) -> np.ndarray:
    """
    Obtain the labes to place them within the Confussion matrix
    """
    cm_class = ["TN", "FP", "FN", "TP"]
    cm_counts = [f"{i:,.0f}" for i in input_cm.flatten()]
    negative_values = input_cm.flatten()[0:2] / np.sum(input_cm.flatten()[0:2])
    positive_values = input_cm.flatten()[2:5] / np.sum(input_cm.flatten()[2:5])
    percentage = [f"{i:.2%}" for i in np.append(negative_values, positive_values)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(cm_counts, percentage, cm_class)]
    return np.asarray(labels).reshape(2, 2)

def plot_roc_cm(model, cv, X, y) -> None:
    """
    Function to create a binary confusion matrix using the test-set.
    """
    # CM variables
    final_tp = []
    final_fp = []
    final_tn = []
    final_fn= []
    for train, test in tqdm(cv.split(X, y.values.ravel())):
        fit_model = model.fit(X.iloc[train], y.iloc[train].values.ravel())
        y_pred = fit_model.predict(X.iloc[test])
        cm = confusion_matrix(y.iloc[test], y_pred)
        final_tp.append(cm[0][0])
        final_fp.append(cm[0][1])
        final_tn.append(cm[1][1])
        final_fn.append(cm[1][0])
    # Plot CM
    final_cm = np.array([np.mean(final_tp), 
                        np.mean(final_fp), 
                        np.mean(final_fn), 
                        np.mean(final_tn)]).reshape(2,2)
    cm_per = _cm_percentage(input_cm= final_cm)
    labels = _cm_labels(input_cm= final_cm)
    # Plot Configs
    sns.set_context("paper", font_scale=1.8)
    plt.figure(figsize=(9, 6))
    tick_labels = ['Not hit', 'Hit']
    # CM plot
    sns.heatmap(cm_per, 
                annot= labels, 
                fmt= "", 
                cmap= "Blues", 
                xticklabels= tick_labels, 
                yticklabels= tick_labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

def plot_AUPRC(model: XGBClassifier, X: pd.DataFrame, y: pd.DataFrame, test_size:float=0.2, seed:float=2) -> None:
    """
    Plot AUROC (Area Under the Precision-Recall Curve) using the test set.
    """
    trainX, testX, trainy, testy = train_test_split(X, 
                                                    y.values.ravel(), 
                                                    test_size=test_size, 
                                                    random_state=seed)
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(testX)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    no_skill = len(testy[testy==1]) / len(testy)
    # Plot
    sns.set_context("paper", font_scale= 1.8)
    plt.figure(figsize= (12, 7))
    plt.plot(lr_recall, lr_precision, lw= 2, label= f"Mean PR AUC= {lr_auc:.2f}")
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color= "black", label=f"No skill= {no_skill: .3f}", lw= 2)
    plt.legend(loc= "upper right")
    plt.tight_layout()