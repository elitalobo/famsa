"""
The experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to SHAP yielding only 0 shapley values 
"""
import warnings
import random

import numpy as np
import joblib


from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap
import torch

random.seed(1)
np.random.seed(1)
# torch.manual_seed(0)


from copy import deepcopy

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

columns = X.columns

X = X.values
xmin = np.min(X,axis=0)
xmax = np.max(X,axis=0)

print (features)
###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def save_data(path,fname, X,y,cols,ycol,ypred):

		y = y.reshape(-1, 1)
		actual_cols = list(columns)
		# X = X[:, self.numerical_cols]
		res = np.concatenate((X, y), axis=1)
		actual_cols.append(y)
		df = pd.DataFrame(res, columns=actual_cols)
		df[ycol]= df[ycol].astype(int)
		fname = path + fname

		df.to_csv(fname, index=False)
		np.save(path + "ypred.npy",np.array(ypred))

def save_results(path,results,filename=None):
	joblib.dump(results,path + filename + ".pkl")





def experiment_main(path, path1, pathshap, pathshap1, ycol, columns,xmin,xmax):
	"""
	Run through experiments for LIME/SHAP on compas using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

	print ('---------------------')
	print ("Beginning LIME COMPAS Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ") # 'two_year_recid','c_charge_degree'
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model1(xtrain, xtest, path,  columns, ss, xmin,xmax, racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'), features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], feature_names=features, perturbation_multiplier=30,xgb_estimators=100)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, sample_around_instance=True, feature_names=adv_lime.get_column_names(), categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], discretize_continuous=False)



	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	e= experiment_summary(explanations, features)
	print (e)
	save_results(path,e,"lime")
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))


	# Repeat the same thing for two features
	adv_lime = Adversarial_Lime_Model1(xtrain, xtest,path1,  columns, ss, xmin,xmax,racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], feature_names=features, perturbation_multiplier=30,xgb_estimators=100)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two'),features.index('c_charge_degree_F'), features.index('c_charge_degree_M'), features.index('two_year_recid'), features.index('race'), features.index("sex_Male"), features.index("sex_Female")], discretize_continuous=False)
                                               
	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())


	print ("LIME Ranks and Pct Occurances two unrelated features:")
	e= experiment_summary(explanations, features)
	save_results(path1,e,"lime")

	print (e)
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	print ('---------------------')
	print ('Beginning SHAP COMPAS Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(xtrain, xtest, pathshap, columns,racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features,xgb_estimators=100)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	explanations = adv_kerenel_explainer.shap_values(xtest)

	# format for display
	formatted_explanations = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	print ("SHAP Ranks and Pct Occurances one unrelated features:")
	e = experiment_summary(formatted_explanations, features)
	save_results(pathshap,e,"shap")
	print (e)
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(xtrain,xtest,pathshap1,columns,racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features,xgb_estimators=100)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	explanations = adv_kerenel_explainer.shap_values(xtest)

	# format for display
	formatted_explanations = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	print ("SHAP Ranks and Pct Occurances two unrelated features:")
	e = experiment_summary(formatted_explanations, features)
	save_results(pathshap1,e,"shap")
	print (e)
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))
	print ('---------------------')

if __name__ == "__main__":
	path = "../../bench/fairml/compas_ood/"
	path1 = "../../bench/fairml/compas_ood1/"
	pathshap = "../../bench/fairml/compas_shapood/"
	pathshap1 = "../../bench/fairml/compas_shapood1/"


	experiment_main(path,path1 ,pathshap, pathshap1,race_indc,columns,xmin,xmax)
