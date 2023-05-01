import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import math

def get_string_form(expl_indices):
    e = sorted(expl_indices)
    e_str = [str(x) for x in e]
    e_str = "-".join(e_str)
    return e_str


class IndexComputer():
    def __init__(self,expls,n_features,preamble,path, p_features, neg_features,map_f):
        self.expls = expls
        self.n_features = n_features + len(p_features) + len(neg_features)
        self.nf = n_features
        self.preamble = preamble
        self.basename = path.split("/")[1]
        self.num_features_add = len(p_features) + len(neg_features)
        self.p_features = p_features
        self.neg_features = neg_features
        self.map_f = map_f


    def get_indice_values(self):
        self.holler={}
        self.deegan={}
        self.resp={}
        self.res = []
        self.res_holler = []
        self.labels={}
        self.resp_rank = {}
        self.deegan_rank = {}
        self.holler_rank = {}



        idx=0

        for expl in self.expls:
            explanations = {}

            ypred = expl[1]
            explanation = expl[0]
            es=[]
            for e in explanation:

                if int(ypred)==1:

                    for jdx in range(len(self.p_features)):
                        k = self.map_f[self.p_features[jdx]]
                        if k not in e:
                            e.append(k)

                else:
                    for jdx in range(len(self.neg_features)):
                        k = self.map_f[self.neg_features[jdx]]
                        if k not in e:
                            e.append(k)
                es_str = get_string_form(e)

                explanations[es_str] = True

                es.append(e)




            new_es=[]
            expls = explanations.keys()
            for es_str in expls:
                flag=False
                for ex in expls:

                    if (es_str in ex and ex!=es_str):
                        flag = True
                        break
                if flag == True:
                    continue
                else:
                    e = [int(x) for x in es_str.split("-")]
                    new_es.append(e)

            es = new_es



            r, h, d, p = self.compare_index(es)
            self.resp[idx]=r

            self.deegan[idx] = d
            self.deegan_rank = np.argsort(d)
            self.holler_rank = np.argsort(h)
            self.resp_rank =  np.argsort(r)
            self.holler[idx] = h
            self.labels[idx] = p

            abs_vals = np.max(np.abs(r - h) + np.abs(r - d) + np.abs(
                h- d))
            values = np.vstack((r,h,d))
            self.res.append(tuple([values, abs_vals, idx]))

            abs_vals = np.max(np.abs(
                values[1, :] - values[2, :]))
            self.res_holler.append(tuple([values, abs_vals, idx]))
            idx+=1

    def responsibility_index(self, expls, sort=False):
        n_features = self.n_features
        importances = {}

        for expl in expls:
            for f in expl:
                if importances.get(f) is None:
                    importances[f] = len(expl)
                else:
                    importances[f] = min(importances[f], len(expl))
        for idx in range(n_features):
            if importances.get(idx) is None:
                importances[idx] = 0.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            if val == 0:
                ranks.append(tuple([key, 0.0]))
            else:

                ranks.append(tuple([key, 1.0 / val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)

        return ranks


    def deegan_packel(self, expls, sort=False):
        n_features = self.n_features
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:

                if len(expl) != 0:
                    importances[f] += 1.0 / len(expl)
        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks


    def holler_packel(self, expls, sort=False):
        n_features = self.n_features
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:
                importances[f] += 1.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks


    def normalize(self, ranks):
        scores = []
        for rank in ranks:
            scores.append(rank[1])

        scores = np.array(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        if np.sum(scores) == 0:
            normalized_scores = scores
        else:
            normalized_scores = np.round(scores / np.sum(scores), 2)

        return normalized_scores


    def compare_index(self, expls, writer=None):
        response_rank = self.responsibility_index(expls,sort=False)
        holler_rank = self.holler_packel(expls, sort=False)
        deegan_rank = self.deegan_packel(expls, sort=False)

        norm_response = self.normalize(response_rank)
        norm_holler = self.normalize(holler_rank)
        norm_deegan = self.normalize(deegan_rank)


        return norm_response, norm_holler, norm_deegan, self.preamble





    def plot_bar(self, xvals, yvals,ylabel, xlabel,basename, filename, dir="plots/",pred=None):
        plt.clf()
        plt.rc('legend', fontsize=28)
        # plt.rc('legend', fontsize='x-large')
        plt.rc('axes', titlesize=28, labelsize=28)
        plt.rc('xtick', labelsize=24)
        plt.rc('ytick', labelsize=24)
        fig = plt.figure(figsize=(26,23))

        # creating the bar plot
        plt.bar(xvals, yvals,
                width=0.4,label="Pred: "+str(pred))
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        if os.path.exists(dir + "/" + basename) is False:
            os.mkdir(dir + "/" + basename)
        path = dir + "/" + basename
        plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print("current dir")
        print(os.getcwd())
        plt.xticks(rotation=20)


        plt.savefig(path + "/" + filename)
        plt.close()

    def extract_points(self):

        basename = self.basename
        resp = self.resp
        holler = self.holler
        deegan = self.deegan
        labels = self.preamble

        self.res.sort(key=lambda a: a[1],reverse=True)
        self.res_holler.sort(key=lambda a: a[1],reverse=True)

        for idx in range(5):
            yvals = self.res[idx][0][0,:]
            xvals = labels
            xlabel = 'Responsibility index value'
            ylabel = 'features'
            file = "response_" + str(self.res[idx][2])
            self.plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

            yvals = self.res[idx][0][1,:]
            xvals = labels
            xlabel = 'Holler-packel index value'
            ylabel = 'features'
            file = "holler_" + str(self.res[idx][2])
            self.plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

            yvals = self.res[idx][0][2,:]
            xvals = labels
            xlabel = 'Deegan-packel index value'
            ylabel = 'features'
            file = "deegan_" + str(self.res[idx][2])
            self.plot_bar(xvals, yvals, xlabel, ylabel, basename, file)


        for idx in range(5):

            yvals = self.res_holler[idx][0][1,:]
            xvals = labels
            xlabel = 'Holler-packel index value'
            ylabel = 'features'
            file = "x_holler_" + str(self.res_holler[idx][2])
            self.plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

            yvals = self.res_holler[idx][0][2,:]
            xvals = labels
            xlabel = 'Deegan-packel index value'
            ylabel = 'features'
            file = "x_deegan_" + str(self.res_holler[idx][2])
            self.plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

        return deegan, holler, resp, self.preamble






def find_top(scores, p):
    scores = np.array(scores)
    if p==0:

        ranks = np.argsort(scores)
    else:
        ranks = np.argsort(-1.0 * scores)
    return ranks

def find_imp(scores,p):
    scores = np.array(scores)
    if p==0:
        flags = scores<0
    else:
        flags = scores>=0
    lent = len(flags)
    imp=[]
    for idx in range(lent):
        if flags[idx]==1:
            imp.append(idx)


    return imp





def check_superset(features,expl_ids):
    for expl in expl_ids:
        bool = True
        for id in expl:
            if id not in features:
                bool = False
        if bool == True:
            return True
    return False




def get_complementary(ids, N):
    c=[]
    for idx in range(N):
        if idx not in ids:
            c.append(idx)
    return c






def compute_ranks(top_indices, labels):
    lime_features={}
    d_features={}
    for key, value in top_indices.items():
        all_label = labels[key]
        indices = value['deegan-packel']
        l_indices = value['lime']
        rank=0
        for idx in indices:
            label = all_label[idx]
            if d_features.get(label) is None:
                d_features[label]=np.zeros(100)
            d_features[label][rank]+=1

            rank+=1

        rank=0
        for idx in l_indices:
            label = all_label[idx]
            if lime_features.get(label) is None:
                lime_features[label] = np.zeros(100)
            lime_features[label][rank] += 1

            rank += 1

    print("lime features")
    # print(lime_features)
    for key, value in lime_features.items():
        print("lime, " + str(key) + " " + str(np.argmax(value)))



    print("deegan packel features")

    for key, value in d_features.items():
        print("deegan, " + str(key) + " " + str(np.argmax(value)))
    # print(d_features)



def experiment_summary(explanations, features):
    """ Provide a high level display of the experiment results for the top three features.
    This should be read as the rank (e.g. 1 means most important) and the pct occurances
    of the features of interest.

    Parameters
    ----------
    explanations : list
    explain_features : list
    bias_feature : string

    Returns
    ----------
    A summary of the experiment
    """
    # features_of_interest = explain_features + [bias_feature]
    top_features = [[], [], []]

    # sort ranks into top 3 features

    for exp in explanations:
        ranks = rank_features(exp)
        for tuple in ranks:
            if tuple[0]<3:
                r = tuple[0]
                top_features[r].append(tuple[1])

        # for i in range(3):
        #     j=0
        #     for f in features:
        #        if ranks[j]==i+1:
        #             top_features[i].append(f)
        #        j+=1

    return get_rank_map(top_features, len(explanations))


def rank_features(explanation):
    """ Given an explanation of type (name, value) provide the ranked list of feature names according to importance

    Parameters
    ----------
    explanation : list

    Returns
    ----------
    List contained ranked feature names
    """
    # from scipy.stats import rankdata
    # ranks = rankdata(explanation, method='min')
    # return ranks
    ordered_tuples = sorted(explanation, key=lambda x : abs(x[1]), reverse=True)
    ranks = []
    r=0
    score = ordered_tuples[0][1]
    for tuple in ordered_tuples:
        if tuple[1]!=score:
            score = tuple[1]
            r+=1

        ranks.append((r,tuple[0],tuple[1]))


    # results = [tup[0] if tup[1] != 0 else ("Nothing shown",0) for tup in ordered_tuples]

    return ranks


def get_rank_map(ranks, to_consider):
    """ Give a list of feature names in their ranked positions, return a map from position ranks
    to pct occurances.

    Parameters
    ----------
    ranks : list
    to_consider : int

    Returns
    ----------
    A dictionary containing the ranks mapped to the uniques.
    """
    unique = {i+1 : [] for i in range(len(ranks))}

    for i, rank in enumerate(ranks):
        for unique_rank in np.unique(rank):
            unique[i+1].append((unique_rank, np.sum(np.array(rank) == unique_rank) / to_consider))

    return unique


def get_indices_maps(features):
    map_f = {}
    map_b = {}
    for idx in range(len(features)):
        map_f[idx] = features[idx]
        map_b[features[idx]] = idx

    return map_f, map_b


def remove_features(formated,map_f,map_b):
    expls = {}
    for e in formated:
        exp=[]
        for e_ in e:
            exp.append(map_b[e_[0]])
        exp1 = sorted(exp)
        exp = [str(x) for x in exp1]
        str_ = "-".join(exp)
        expls[str_]=True
    new_formatted=[]
    for e in formated:
        exp = []
        for e_ in e:
            exp.append(map_b[e_[0]])
        exp = np.sorted(exp)
        exp = [str(x) for x in exp]
        str_ = "-".join(exp)
        flag=False
        for key in expls.keys():
            if (str_ in key and str_!= key):
                flag=True
        if flag!=True:
            new_formatted.append(e)
    return new_formatted


def save_results(path,results,filename=None):
	joblib.dump(results,path + filename + ".pkl")

