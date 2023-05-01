import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
def plot_bar(xvals, yvals,ylabel, xlabel,basename, filename, dir="plots/",pred=None):
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

def extract_points(filename):
    filet = open(filename,'r+')
    res=[]
    res_holler=[]
    vals=[]
    labels=None
    deegan = {}
    holler = {}
    resp = {}

    all_labels = {}
    for line in filet:
        words = line.strip("\n").split(",")[:-2]
        lent = int((len(words) - 1) / 4)
        data_pt = int(words[0])
        labels = words[1].split(":")
        value1 = [float(x) for x in words[2].split(";")]
        value2 = [float(x) for x in words[3].split(":")]
        value3 = [float(x) for x in words[4].split(":")]

        values = np.zeros((3,len(value1)))

        values[0,:] = value1
        values[1, :] = value2
        values[2, :] = value3

        # values = values.reshape((3,-1))

        deegan[data_pt]=value2
        holler[data_pt]=value3
        resp[data_pt]=value1
        all_labels[data_pt] = labels
        # if data_pt== 4:
        #     print("here")

        abs_vals = np.max(np.abs(values[0,:]-values[1,:])+np.abs(values[0,:]-values[2,:])+ np.abs(values[1,:]-values[2,:]))
        res.append(tuple([values,abs_vals,data_pt]))

        abs_vals = np.max( np.abs(
            values[1, :] - values[2, :]))
        res_holler.append(tuple([values, abs_vals, data_pt]))


    res.sort(key=lambda a: a[1],reverse=True)
    res_holler.sort(key=lambda a: a[1],reverse=True)

    for idx in range(5):
        yvals = res[idx][0][0,:]
        xvals = labels
        xlabel = 'Responsibility index value'
        ylabel = 'features'
        basename = filename.split("/")[-1]
        file = "response_" + str(res[idx][2])
        plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

        yvals = res[idx][0][1,:]
        xvals = labels
        xlabel = 'Holler-packel index value'
        ylabel = 'features'
        basename = filename.split("/")[-1]
        file = "holler_" + str(res[idx][2])
        plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

        yvals = res[idx][0][2,:]
        xvals = labels
        xlabel = 'Deegan-packel index value'
        ylabel = 'features'
        basename = filename.split("/")[-1]
        file = "deegan_" + str(res[idx][2])
        plot_bar(xvals, yvals, xlabel, ylabel, basename, file)


    for idx in range(5):

        yvals = res[idx][0][1,:]
        xvals = labels
        xlabel = 'Holler-packel index value'
        ylabel = 'features'
        basename = filename.split("/")[-1]
        file = "x_holler_" + str(res_holler[idx][2])
        plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

        yvals = res[idx][0][2,:]
        xvals = labels
        xlabel = 'Deegan-packel index value'
        ylabel = 'features'
        basename = filename.split("/")[-1]
        file = "x_deegan_" + str(res_holler[idx][2])
        plot_bar(xvals, yvals, xlabel, ylabel, basename, file)

    return deegan, holler, resp, all_labels
def compute_lime(path,all_labels):
    f = open(path,'r+')
    res={}
    predictions = {}
    for line in f:
        try:
            words = line.split(",")
            index = int(words[0])
            y_pred = float(words[1])
            values = words[2].split(":")
            labels = words[3].split(":")

            labels_ = all_labels[index]
            scores = {}
            idx = 0
            for label in labels:
                actual_label = label
                scores[actual_label] = values[idx]
                idx += 1
            final_vals = []
            for label in labels_:
                actual_label = get_label(label)

                final_vals.append(float(scores[actual_label]))

            predictions[index] = y_pred
            indices = np.array(final_vals)
            res[index]= indices
        except:
            pass

    return res, predictions

def get_label(label):
    actual_label = label.split(" ")
    if len(actual_label) == 3:
        actual_label = actual_label[0].strip()
    elif len(actual_label) >= 5:
        actual_label = actual_label[2].strip()
    elif len(actual_label)==1:
        actual_label = actual_label[0]
    else:
        print("here")
        assert (0)

    return actual_label

def compute_shap(path,all_labels):
    f = open(path, 'r+')
    res = {}
    predictions = {}
    for line in f:
        try:
            words = line.split(",")
            index = int(words[0])
            y_pred = float(words[1])
            values = words[2].split(":")
            labels = words[3].split(":")

            labels_ = all_labels[index]
            scores = {}
            idx = 0
            for label in labels:
                actual_label = get_label(label)
                scores[actual_label] = values[idx]
                idx += 1
            final_vals = []
            for label in labels_:
                actual_label = get_label(label)
                final_vals.append(float(scores[actual_label]))

            predictions[index] = y_pred
            indices  = final_vals
            res[index] = indices
        except:
            pass
    return res, predictions

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



def compute_expl_ids(expls,labels):

    res={}
    idx=0
    for x in labels:
        actual_e = get_label(x)
        res[actual_e]=idx
        idx+=1
    all_exp=[]
    for exp in expls:
        expl_d=[]
        for e_ in exp:
            try:
                actual_e = get_label(e_)
                expl_d.append(res[actual_e])
            except:
                print("here")
        all_exp.append(expl_d)
    return all_exp

def get_top_k(ranks,k):
    lent = len(ranks)
    features=[]
    features = ranks[:k]
    return features


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



def compute_metric(res,path,k=5):
    path = path.split("_points.txt")[0]
    full_path = "data/" + path + "/expls.pkl"
    res_dir = "results/"
    if os.path.exists(res_dir) == False:
        os.mkdir(res_dir)
    res_path = res_dir + path + "_results_k=" + str(k) + ".txt"
    f = open(res_path,'w+')
    expls = joblib.load(full_path)
    # res.append((x, missed, labels, resp[x], holler[x], deegan[x], lime_scores[x], shap_scores[x]))

    results={}
    all_expls={}
    lent = len(expls)

    resp_ness=0
    holler_ness=0
    deegan_ness=0
    lime_ness=0
    shap_ness=0

    resp_suf = 0
    holler_suf = 0
    deegan_suf = 0
    lime_suf = 0
    shap_suf = 0



    for idx in range(lent):
        x, missed, labels, resp, holler, deegan, lime, shap = res[idx]
        results[x]=(missed, labels, resp, holler, deegan, lime, shap)

    for idx in range(lent):
        expl = expls[idx]
        all_expls[idx]=expl

    total = len(results.keys())
    for index, value in results.items():
        missed, labels, resp, holler, deegan, lime, shap = value
        lent = len(resp)
        expl_ids = compute_expl_ids(all_expls[index],labels)

        features = get_top_k(resp,k)
        if check_superset(features,expl_ids)== True:
            resp_suf+=1
        comp = get_complementary(features,lent)
        if check_superset(comp,expl_ids)== False:
            resp_ness+=1


        features = get_top_k(holler, k)
        if check_superset(features, expl_ids)== True:
            holler_suf+=1
        comp = get_complementary(features, lent)
        if check_superset(comp, expl_ids) == False:
            holler_ness += 1


        features = get_top_k(deegan, k)
        if check_superset(features, expl_ids) == True:
            deegan_suf +=1
        comp = get_complementary(features, lent)
        if check_superset(comp, expl_ids) == False:
            deegan_ness += 1

        features = get_top_k(lime, k)
        if check_superset(features, expl_ids)== True:
            lime_suf+=1
        comp = get_complementary(features, lent)
        if check_superset(comp, expl_ids) == False:
           lime_ness += 1


        features = get_top_k(shap, k)
        if check_superset(features, expl_ids)== True:
            shap_suf+=1
        comp = get_complementary(features, lent)
        if check_superset(comp, expl_ids) == False:
            shap_ness += 1

    resp_ness = resp_ness/total
    holler_ness = holler_ness/total
    deegan_ness = deegan_ness/total
    lime_ness = lime_ness/total
    shap_ness = shap_ness/total

    resp_suf = resp_suf/total
    holler_suf = holler_suf/total
    deegan_suf = deegan_suf/total
    lime_suf = lime_suf/total
    shap_suf = shap_suf/total

    arr= [resp_ness,holler_ness,deegan_ness,lime_ness,shap_ness,resp_suf,holler_suf,deegan_suf,lime_suf,shap_suf]
    arr = [str(np.round(x,3)) for x in arr]
    arr_str = ",".join(arr)
    f.write(arr_str+"\n")
    f.flush()
    f.close()














    print(path)



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



def compare_index(path):
    lime_dir = "lime_data/"
    mwc_dir = "mwc_data/"
    shap_dir = "shap_data/"
    deegan, holler, resp, all_labels =  extract_points(mwc_dir + path)
    lime_scores, predictions = compute_shap(lime_dir + path,all_labels)
    shap_scores, predictions = compute_lime(shap_dir + path,all_labels)

    dir = "plots_all"
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    res=[]
    res_indexes=[]
    imp_indices={"responsibility":{},"deegan-packel":{},"holler-packel":{}}
    top_indices={}


    results = open("plots_all/"+ path + "_results.txt",'w+')
    for x, val in lime_scores.items():
        try:
            s = predictions[x]
            # if s==0:
            #     continue
            labels = all_labels[x]
            top_deegan = find_top(deegan[x],s)
            top_holler = find_top(holler[x],s)
            top_resp = find_top(resp[x],s)

            imp_deegan = find_imp(deegan[x],s)
            imp_holler = find_imp(holler[x],s)
            imp_resp = find_imp(resp[x],s)



            imp_indices["responsibility"][x]=imp_resp
            imp_indices["holler-packel"][x]=imp_holler
            imp_indices["deegan-packel"][x]= imp_deegan

            top_indices[x]={}

            top_indices[x]["responsibility"] = top_resp
            top_indices[x]["holler-packel"] = top_holler
            top_indices[x]["deegan-packel"] = top_deegan


            top_lime = find_top(lime_scores[x],s)
            top_shap = find_top(shap_scores[x],s)

            top_indices[x]['lime'] = top_lime



            resp_str = ":".join([labels[x] for x in top_resp])
            deegan_str = ":".join([labels[x] for x in top_deegan])
            holler_str =   ":".join([labels[x] for x in top_holler])
            lime_str = ":".join([labels[x] for x in top_lime])
            shap_str = ":".join([labels[x] for x in top_shap])

            missed = np.sum(np.abs(top_lime-top_deegan)) + np.sum(np.abs(top_lime-top_holler)) +  np.sum(np.abs(top_lime-top_resp)) + np.sum(np.abs(top_shap-top_deegan)) + np.sum(np.abs(top_shap-top_holler))  + np.sum(np.abs(top_resp-top_shap))
            res.append((x,missed,labels,resp[x],holler[x],deegan[x],lime_scores[x],shap_scores[x],predictions[x]))
            res_indexes.append((x,missed,labels,top_resp,top_holler,top_deegan,top_lime,top_shap))
            results.write(str(x) + "," + str(
                missed) + "," + resp_str + "," + holler_str + "," + deegan_str + "," + lime_str + "," + shap_str + "\n"
                          )

        except:
            pass

    results.close()
    joblib.dump(res,"plots_all/" + path + "_ranks.pkl")
    path1 = path.split("_points.txt")[0]
    joblib.dump(imp_indices,"data/"+path1+"/imp.pkl")

    # compute_metric(res_indexes, path,k=9)
    # compute_metric(res_indexes, path,k=7)
    # compute_metric(res_indexes, path,k=5)
    # compute_metric(res_indexes, path,k=3)
    #
    # compute_metric(res_indexes,path,k=1)

    compute_ranks(top_indices,all_labels)


    print("here")

    res.sort(key=lambda a: a[1], reverse=True)
    for idx in range(10):
        yvals = res[idx][3]
        xvals = res[idx][2]
        pred= res[idx][-1]
        xlabel = 'Responsibility index value'
        ylabel = 'features'
        file = "resp_" + str(res[idx][0])
        plot_bar(xvals, yvals, xlabel, ylabel, path, file,dir="plots_all",pred=pred)

        plt.clf()

        yvals = res[idx][4]
        xvals = res[idx][2]
        xlabel = 'Holler-Packel index value'
        ylabel = 'features'
        file = "holler_" + str(res[idx][0])
        plot_bar(xvals, yvals, xlabel, ylabel, path, file, dir="plots_all",pred=pred)

        plt.clf()

        yvals = res[idx][5]
        xvals = res[idx][2]
        xlabel = 'Deegan-Packel index value'
        ylabel = 'features'
        file = "deegan_" + str(res[idx][0])
        plot_bar(xvals, yvals, xlabel, ylabel, path, file, dir="plots_all",pred=pred)

        plt.clf()

        yvals = res[idx][6]
        xvals = res[idx][2]
        xlabel = 'Lime index value'
        ylabel = 'features'
        file = "lime_" + str(res[idx][0])
        plot_bar(xvals, yvals, xlabel, ylabel, path, file, dir="plots_all",pred=pred)

        plt.clf()

        yvals = res[idx][7]
        xvals = res[idx][2]
        xlabel = 'Shap  index value'
        ylabel = 'features'
        file = "shap_" + str(res[idx][0])
        plot_bar(xvals, yvals, xlabel, ylabel, path, file, dir="plots_all",pred=pred)

        plt.clf()






if __name__=='__main__':
    # compare_index("new_compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    compare_index("compas_ood_data_nbestim_50_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # compare_index("lending_data_nbestim_60_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
    # # extract_points(filet)
    # extract_points("../mwc_data/adult_data_nbestim_30_maxdepth_3_testsplit_0.2.mod_nbestim_100_maxdepth_3_testsplit_0.2_points.txt")
