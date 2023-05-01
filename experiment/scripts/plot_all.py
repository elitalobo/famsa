import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.patches as mpatches
plt.style.use("scripts/plot_style.txt")
import pandas as pd
def plot_bar(all_res,fname ,dir="plots/"):
    plt.clf()
    plt.figure(figsize=(12,10))

    X = np.arange(3)
    sns.set_style({'axes.grid': False})

    keys = list(all_res.keys())
    lent = len(all_res.keys())
    fig, ax = plt.subplots(1, lent, sharey=True)
    # plt.show()

    fig.set_size_inches(8,8)

    colors=['red','blue','green','magenta']
    # plotting columns
    idx=0
    handles=[]
    for key in keys:
        jdx=0
        df = pd.DataFrame(all_res[key],index=np.array(X))
        # for subkey, value in all_res[key].items():
            # sns.barplot(x=np.array(X), y=np.array(value)*100, color=colors[jdx],ax=ax[idx],label=subkey)

            # print("plotted")
            # if jdx==0:
            #
            # jdx+=1
        # create stacked bar chart for monthly temperatures


        df.plot(kind='bar', stacked=True, color=colors,ax=ax[idx],legend=False,title=key,rot=0)
        if idx!=0:
            ax[idx].yaxis.set_tick_params(left=False)
        ax[idx].set_xticklabels(np.array([1,2,3]))

        sns.despine(left=True,ax=ax[idx])

        # if idx!=0:
        #     ax[idx].set_yticks([])

        # plt.xlabel('Ranks')
        # plt.ylabel('% Ocurrences')
        # ax[idx].invert_yaxis()
        fig.supxlabel('Ranks')
        fig.supylabel('occurences')

        idx+=1
        # ax[0].set(ylabel='% of occurences')
    # plt.legend()

    lines_labels = [ax[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels,loc='upper center',
          fancybox=True, shadow=True, ncol=5)
    # plt.xlabel("Ranks")

    plt.savefig(dir+fname+".png")


    # renaming the axes
    # ax.set(xlabel="x-axis", ylabel="y-axis")


def load_data(path, filename,name="compas"):
    labels = ["shap","lime","resp","holler","deegan"]
    # features = ['race','unrelated_column_one','unrelated_column_two','other']
    if name=="compas":
        features = ['race','unrelated_column_one','unrelated_column_two']
    else:
        features = ['gender', 'unrelated_column_one', 'unrelated_column_two']


    all_res = {}
    fname=None
    for label in labels:
        try:
            fname = path + label + ".pkl"
            res = joblib.load(fname)
        except:
            print("here")
            continue
        results={}

        for f in features:
            if results.get(f) is None:
                results[f] = np.zeros(3)
        
        for idx in range(3):
            tuples = res[idx+1]
            sum=0
            for tuple in tuples:
                if tuple[0] in features:
                    if results.get(tuple[0])is None:
                        results[tuple[0]]=np.zeros(3)
                    results[tuple[0]][idx]=tuple[1]
                else:
                    sum+= tuple[1]


            # results['other'][idx]=sum
        
        all_res[label]=results
    
    plot_bar(all_res,filename)
                    
                    







if __name__=='__main__':
    path = "../bench/fairml/compas_ood/"
    load_data(path,"compas_ood")

    path = "../bench/fairml/compas_ood1/"
    load_data(path, "compas_ood1")

    path = "../bench/fairml/german_ood/"
    load_data(path, "german_ood",name="german")

    # path = "../bench/fairml/compas_shapood/"
    # load_data(path, "compas_shapood")
    #
    # path = "../bench/fairml/compas_shapood1/"
    # load_data(path, "compas_shapood1")
