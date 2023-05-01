import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import math
from find_compass_custom import *

if __name__=='__main__':
    # compare_index("new_compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # compare_index("compas_ood_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # compare_index("lending_data_nbestim_60_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # # extract_points(filet)
    # extract_points("../mwc_data/adult_data_nbestim_30_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    path = "data/german_ood_nbestim_50_maxdepth_3_testsplit_0.0.mod_nbestim_100_maxdepth_3_testsplit_0.0/g_expls.pkl"
    expls = joblib.load(path)
    p="../bench/fairml/german_ood/"


    features = ["Age","LoanDuration","LoanAmount","LoanRateAsPercentOfIncome","YearsAtCurrentHome","NumberOfOtherLoansAtBank","NumberOfLiableIndividuals","gender","unrelated_column_one"]

    # features = ["age", "priors_count", "length_of_stay", 'race', 'unrelated_column_one', 'f2']
    p_features = ['gender']
    neg_features = ['unrelated_column_one']
    map_f, map_b = get_indices_maps(features)
    indComputer = IndexComputer(expls,len(features)-3,features,path,p_features,neg_features,map_b)
    indComputer.get_indice_values()

    # indComputer = IndexComputer(expls,len(features)-2,features,path,2)
    # indComputer.get_indice_values()

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




