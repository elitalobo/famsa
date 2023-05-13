import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import math
from compas_adversarial import *

if __name__=='__main__':
    path = "data/german_lmodified_nbestim_50_maxdepth_3_testsplit_0.0.mod_nbestim_100_maxdepth_3_testsplit_0.0/g_expls.pkl"
    expls = joblib.load(path)
    p="../bench/fairml/german_lmodified/"


    features ="NumberOfLiableIndividuals,NumberOfOtherLoansAtBank,LoanRateAsPercentOfIncome,YearsAtCurrentHome,LoanDuration,Age,LoanAmount,Gender"
    features = features.split(",")

    p_features = ['Gender']
    neg_features = ['LoanRateAsPercentOfIncome']
    map_f, map_b = get_indices_maps(features)
    indComputer = IndexComputer(expls,len(features),features,path,p_features,neg_features,map_b)
    indComputer.get_indice_values()
    indComputer.extract_points()


    formatted_explanations = []
    for idx, exp in indComputer.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Responsibility rank and Pct Occurances one unrelated features:")
    e= experiment_summary(formatted_explanations, features)
    print(e)
    save_results(p,e,"resp")



    formatted_explanations = []
    for idx, exp in indComputer.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Holler-Packel rank and Pct Occurances one unrelated features:")
    print(e)
    e= experiment_summary(formatted_explanations, features)
    save_results(p,e,"holler")


    formatted_explanations = []
    for idx, exp in indComputer.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Deegan-Packel rank and Pct Occurances one unrelated features:")
    print(e)
    e= experiment_summary(formatted_explanations, features)
    save_results(p,e,"deegan")

    path = "data/german_smodified_nbestim_60_maxdepth_3_testsplit_0.0.mod_nbestim_100_maxdepth_3_testsplit_0.0/g_expls.pkl"
    expls = joblib.load(path)
    p = "../bench/fairml/german_smodified/"

    features = "NumberOfLiableIndividuals,NumberOfOtherLoansAtBank,LoanRateAsPercentOfIncome,YearsAtCurrentHome,LoanDuration,Age,LoanAmount,OtherLoansAtStore,ForeignWorker,HasGuarantor,CheckingAccountBalance_geq_200,Gender"
    features = features.split(",")
    p_features = ['Gender']
    neg_features = ['LoanRateAsPercentOfIncome']
    map_f, map_b = get_indices_maps(features)
    indComputer = IndexComputer(expls, len(features), features, path, p_features, neg_features, map_b)
    indComputer.get_indice_values()
    indComputer.extract_points()


    formatted_explanations = []
    for idx, exp in indComputer.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Responsibility rank and Pct Occurances one unrelated features:")
    e = experiment_summary(formatted_explanations, features)
    print(e)
    save_results(p, e, "resp")

    formatted_explanations = []
    for idx, exp in indComputer.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Holler-Packel rank and Pct Occurances one unrelated features:")
    print(e)
    e = experiment_summary(formatted_explanations, features)
    save_results(p, e, "holler")

    formatted_explanations = []
    for idx, exp in indComputer.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Deegan-Packel rank and Pct Occurances one unrelated features:")
    print(e)
    e = experiment_summary(formatted_explanations, features)
    save_results(p, e, "deegan")




