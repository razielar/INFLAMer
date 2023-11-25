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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_curve, auc)
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
