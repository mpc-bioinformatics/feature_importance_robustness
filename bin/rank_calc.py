import pandas as pd
import glob
import numpy as np
import itertools
import plotly.express as px
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to result folder of a training run")
args = parser.parse_args()

PATH = args.input

def get_permu_rank(path):

    permu_values = []
    for permus in glob.glob(path + "permu*.csv"):
        data = pd.read_csv(permus)
        data["index"] = data.index
        print(data.columns)
        test = pd.Series(data.index.values, index=data.col_name).to_dict()
        permu_values.append(test)

    dd = {}
    for d in permu_values:
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]

    dd_small = dict(itertools.islice(dd.items(), 100))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
    median_df.to_csv(PATH + "permu_median.csv")
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
        data = pd.read_csv(shappies)
        data["index"] = data.index
        test = pd.Series(data.index.values, index=data.col_name).to_dict()
        shap_values.append(test)

    dd = {}
    for d in shap_values:
        print(d)
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
    dd_small = dict(itertools.islice(dd.items(), 100))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
    median_df.to_csv(PATH + "shap_median_all.csv")
    df2 = dd_df[median_df.index]
    df2.boxplot()
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
        data = pd.read_csv(forries)
        data["index"] = data.index
        test = pd.Series(data.index.values, index=data["Unnamed: 0"]).to_dict()
        forest_values.append(test)
    dd = {}
    for d in forest_values:
        for key, value in d.items():
            if key in dd:
                dd[key].append(value)
            else:
                dd[key] = [value]
    dd_small = dict(itertools.islice(dd.items(), 100))
    dd_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dd_small.items() ]))
    
    median_df = dd_df.median().sort_values()
    print(median_df)
    median_df.to_csv(PATH + "tree_median.csv")
    df2 = dd_df[median_df.index]
    dd_df = pd.DataFrame(dd)
    df2.boxplot()
    plt.xticks(rotation='vertical')
    plt.ylabel("feature importance rank")
    plt.title("boxplot of ranks for MDI")
    plt.savefig(PATH + "tree_box_sorted.svg", bbox_inches="tight",format="svg")
    plt.close()
    dd_vars = {}
    for key,  value in  dd.items():
        dd_vars[key] = np.std(value)
    return(dd_vars)


get_permu_rank(PATH)
get_shap_rank(PATH)
get_tree_rank(PATH)