#!/usr/bin/env python
#coding=utf8
"""
Author: Raziel Amador Rios
Version: 0.0.1
"""
from xgboost import XGBClassifier

def print_training_results(scoring_dict: dict, input_model: XGBClassifier, cv_results: dict) -> None:
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