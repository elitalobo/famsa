from utils import *




if __name__=='__main__':
    # compare_index("new_compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # compare_index("compas_ood_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # compare_index("lending_data_nbestim_60_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # # extract_points(filet)

    print("compasood")
    path1="../bench/fairml/compas_ood/"
    # extract_points("../mwc_data/adult_data_nbestim_30_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    path = "data/compas_ood_nbestim_100_maxdepth_3_testsplit_0.0.mod_nbestim_100_maxdepth_3_testsplit_0.0/o_expls.pkl"
    expls = joblib.load(path)
    print(len(expls))

    features = ["age","priors_count","length_of_stay",'race','unrelated_column_one','unrelated_column_two']
    p_features = ['race']
    neg_features = ['unrelated_column_one']
    map_f, map_b = get_indices_maps(features)
    indComputer = IndexComputer(expls,len(features)-3,features,path,p_features,neg_features,map_b)
    indComputer.get_indice_values()

    formatted_explanations = []
    for idx, exp in indComputer.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Responsibility rank and Pct Occurances one unrelated features:")
    e= experiment_summary(formatted_explanations, features)
    print(e)
    save_results(path1,e,"resp")


    formatted_explanations = []
    for idx, exp in indComputer.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Holler-Packel rank and Pct Occurances one unrelated features:")
    e= experiment_summary(formatted_explanations, features)
    print(e)
    save_results(path1,e,"holler")

    formatted_explanations = []
    for idx, exp in indComputer.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Deegan-Packel rank and Pct Occurances one unrelated features:")
    e = experiment_summary(formatted_explanations, features)
    print(e)
    save_results(path1,e,"deegan")

    features = ["age","priors_count","length_of_stay",'race','unrelated_column_one','unrelated_column_two']
    p="../bench/fairml/compas_ood1/"

    print("compasood1")

    path1 = "data/compas_ood1_nbestim_100_maxdepth_3_testsplit_0.0.mod_nbestim_100_maxdepth_3_testsplit_0.0/o_expls.pkl"


    expls = joblib.load(path1)
    neg_features=["unrelated_column_one","unrelated_column_two"]
    indComputer = IndexComputer(expls, len(features)-3, features, path1,p_features,neg_features,map_b)
    indComputer.get_indice_values()

    formatted_explanations = []
    for idx, exp in indComputer.resp.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Responsibility rank and Pct Occurances one unrelated features:")
    e=experiment_summary(formatted_explanations, features)
    print(e)
    save_results(p,e,"resp")

    formatted_explanations = []
    for idx, exp in indComputer.holler.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Holler-Packel rank and Pct Occurances one unrelated features:")
    e=experiment_summary(formatted_explanations, features)
    print(e)
    save_results(p,e,"holler")

    formatted_explanations = []
    for idx, exp in indComputer.deegan.items():
        formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
    print("Deegan-Packel rank and Pct Occurances one unrelated features:")
    e=experiment_summary(formatted_explanations, features)
    print(e)
    save_results(p,e,"deegan")

    # indComputer.extract_points()

#     print("shapood")
#
#
#     features = "age,two_year_recid,priors_count,length_of_stay,c_charge_degree_F,c_charge_degree_M,sex_Female,sex_Male,race,unrelated_column_one,unrelated_column_two,is_not_ood"
#     features = features.split(",")
#     map_f, map_b = get_indices_maps(features)
#
#
#     p="../bench/fairml/compas_shapood/"
#
#     path1 = "data/compas_shapood_nbestim_50_maxdepth_3_testsplit_0.01.mod_nbestim_100_maxdepth_3_testsplit_0.01/o_expls.pkl"
#
#     expls = joblib.load(path1)
#
#     neg_features = ["unrelated_column_one"]
#     indComputer = IndexComputer(expls, len(features) - 3, features, path1, p_features, neg_features, map_b)
#     indComputer.get_indice_values()
#
#     formatted_explanations = []
#     for idx, exp in indComputer.resp.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#     print("Responsibility rank and Pct Occurances one unrelated features:")
#     e=experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p,e,"resp")
#
#     formatted_explanations = []
#     for idx, exp in indComputer.holler.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#     print("Holler-Packel rank and Pct Occurances one unrelated features:")
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#
#     e = experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p, e, "holler")
#
#     formatted_explanations = []
#     for idx, exp in indComputer.deegan.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#     print("Deegan-Packel rank and Pct Occurances one unrelated features:")
#
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#
#     e = experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p, e, "deegan")
#
#     print("shapoods1")
#     p="../bench/fairml/compas_shapood1/"
#
#     path1 = "data/compas_shapood1_nbestim_50_maxdepth_3_testsplit_0.01.mod_nbestim_100_maxdepth_3_testsplit_0.01/o_expls.pkl"
#
#     expls = joblib.load(path1)
#     neg_features = ["unrelated_column_one", "unrelated_column_two"]
#     indComputer = IndexComputer(expls, len(features) - 3, features, path1, p_features, neg_features, map_b)
#
#     indComputer.get_indice_values()
#
#     formatted_explanations = []
#     for idx, exp in indComputer.resp.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#     print("Responsibility rank and Pct Occurances one unrelated features:")
#     e = experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p, e, "resp")
#
#     formatted_explanations = []
#     for idx, exp in indComputer.holler.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#
#     print("Holler-Packel rank and Pct Occurances one unrelated features:")
#     e = experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p, e, "holler")
#
#     formatted_explanations = []
#     for idx, exp in indComputer.deegan.items():
#         formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
#
# #    formatted_explanations = remove_features(formatted_explanations,map_f,map_b)
#
#     print("Deegan-Packel rank and Pct Occurances one unrelated features:")
#     e = experiment_summary(formatted_explanations, features)
#     print(e)
#     save_results(p, e, "deegan")


