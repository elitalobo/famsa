from utils import *




if __name__=='__main__':

    print("compas")
    path1="../bench/fairml/compas/"
    path = "data1/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.0/expls.pkl"
    expls = joblib.load(path)
    print(len(expls))


    features = "Number_of_Priors,score_factor,Age_Above_FourtyFive,Age_Below_TwentyFive,African_American,Asian,Hispanic,Native_American,Other,Female,Misdemeanor,Two_yr_Recidivism"
    features = features.split(",")
    p_features = []
    neg_features = []
    map_f, map_b = get_indices_maps(features)
    indComputer = IndexComputer(expls,len(features),features,path,p_features,neg_features,map_b)
    indComputer.get_indice_values(add_features=False)

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

