import pandas as pd
import glob
import numpy as np
import itertools
import plotly.express as px
import matplotlib.pyplot as plt

PATH="/Users/webermac/Projekte/Mazurka/results_mac/2025/proteins/RepA/elastic_net/run13_06/"#/Users/webermac/Projekte/Mazurka/results_mac/upsampled_100/cohen_01/run1/"

def get_permu_rank(path):

    permu_values = []
    for permus in glob.glob(path + "permu*.csv"):
      #  print(permus)
        data = pd.read_csv(permus)

        data["index"] = data.index
        print(data.columns)
        test = pd.Series(data.index.values, index=data.col_name).to_dict()
        permu_values.append(test)

  #  print(permu_values)
    dd = {}
    for d in permu_values:
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
   # print("permu dd :", dd)

    dd_small = dict(itertools.islice(dd.items(), 40))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
   # median_df.to_csv(PATH + "permu_median.csv")
    #sorts the dataframe by median
    df2 = dd_df[median_df.index]

        
    dd_df = pd.DataFrame(dd)


    df2.boxplot()
    plt.xticks(rotation='vertical')
    plt.ylabel("feature importance rank")
    plt.title("boxplot of ranks for PFI")
    plt.savefig(PATH + "permu_box_sorted.svg", bbox_inches="tight",format="svg")
    plt.close()
    dd_vars = {}
    for key,  value in  dd.items():
        dd_vars[key] = np.std(value)

    return(dd_vars)

def get_shap_rank(path):

    shap_values = []
    for shappies in glob.glob(path + "shap_values*.csv"):
       # print(shappies)
        data = pd.read_csv(shappies)
        data["index"] = data.index
        test = pd.Series(data.index.values, index=data.col_name).to_dict()
        shap_values.append(test)

   # print(shap_values)
    dd = {}
    for d in shap_values:
        print(d)
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
   # print("shap dd: ", dd)
    dd_small = dict(itertools.islice(dd.items(), 400))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
    median_df.to_csv(PATH + "shap_median_all.csv")
    df2 = dd_df[median_df.index]
    df2.boxplot()
    #dd_df.plot(kind="box")
    plt.xticks(rotation='vertical')
    plt.ylabel("feature importance rank")
    plt.title("boxplot of ranks for SHAP")
    plt.savefig(PATH + "shap_box_sorted.svg", bbox_inches="tight",format="svg")
    plt.close()
    dd_vars = {}
    for key,  value in  dd.items():
        dd_vars[key] = np.std(value)

    return(dd_vars)


def get_tree_rank(path):

    forest_values = []
    for forries in glob.glob(path + "forest*.csv"):
      #  print(forries)
        data = pd.read_csv(forries)
      #  print(data.columns)
        data["index"] = data.index
        test = pd.Series(data.index.values, index=data["Unnamed: 0"]).to_dict()
        forest_values.append(test)

   # print(forest_values)
    dd = {}
    for d in forest_values:
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
    #print("tree dd :",  dd)
    dd_small = dict(itertools.islice(dd.items(), 40))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
    print(median_df)
   # median_df.to_csv(PATH + "tree_median.csv")
    df2 = dd_df[median_df.index]
    dd_df = pd.DataFrame(dd)
    print(dd_df)
    #dd_df.boxplot()
    df2.boxplot()
    plt.xticks(rotation='vertical')
    plt.ylabel("feature importance rank")
    plt.title("boxplot of ranks for MDI")
    plt.savefig(PATH + "tree_box_sorted.svg", bbox_inches="tight",format="svg")
    plt.close()
    dd_vars = {}
    for key,  value in  dd.items():
        dd_vars[key] = np.std(value)
  #  print(dd_vars)
    return(dd_vars)

#my_shap  =  get_shap_rank(PATH)
#my_permu  =  get_permu_rank(PATH)
#my_tree=  get_tree_rank(PATH)

#print(my_shap)
#fig  =  px.histogram(x=my_shap.keys(),  y=my_shap.values())
#fig.show()

#fig  =  px.histogram(x=my_permu.keys(),  y=my_permu.values())
#fig.show()
def plot_shap_box(my_shap):
    plt.bar(my_shap.keys(), my_shap.values())
    plt.xticks(rotation='vertical')
    plt.ylabel("standard deviation of ranks")
    plt.title("shap")
    plt.ylim(0, 7.5)
    #plt.savefig(PATH + "shap.svg", bbox_inches="tight", format="svg")
    plt.close()

def plot_permu_box(my_permu):
    plt.bar(my_permu.keys(), my_permu.values())
    plt.xticks(rotation='vertical')
    plt.ylabel("standard deviation of ranks")
    plt.title("PFI")
    plt.ylim(0, 12)
    plt.savefig(PATH + "permu.svg", bbox_inches="tight",format="svg")
    plt.close()

def plot_tree_box(my_tree):
    plt.bar(my_tree.keys(), my_tree.values())
    plt.xticks(rotation='vertical')
    plt.ylabel("standard deviation of ranks")
    plt.title("MDI")
    plt.ylim(0, 6)
    plt.savefig(PATH + "tree.svg", bbox_inches="tight",format="svg")
    plt.close()


def get_elasticnet_rank(path):

    shap_values = []
    for shappies in glob.glob(path + "elastic_importance*.csv"):
       # print(shappies)
        data = pd.read_csv(shappies)
        data = data.T
        print(data)
        data = data.rename(columns={0: "importances"})
        print(data)
        
        test = pd.Series(data.importances, index=data.index.values).to_dict()
        shap_values.append(test)

   # print(shap_values)
    dd = {}
    for d in shap_values:
        print(d)
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
   # print("shap dd: ", dd)
    dd_small = dict(itertools.islice(dd.items(), 200))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values(ascending=False)
    median_df.to_csv(PATH + "elastic_median.csv")
    df2 = dd_df[median_df.index]
    df2.boxplot()
    #dd_df.plot(kind="box")
    plt.xticks(rotation='vertical')
    plt.ylabel("feature importance rank")
    plt.title("boxplot of ranks for SHAP")
    plt.savefig(PATH + "elastic_box_sorted.pdf", bbox_inches="tight",format="pdf")
    plt.close()
    dd_vars = {}
    for key,  value in  dd.items():
        dd_vars[key] = np.std(value)

    return(median_df)

#get_permu_rank(PATH)
#get_shap_rank(PATH)
elastic_df = get_elasticnet_rank(PATH)
#get_tree_rank(PATH)