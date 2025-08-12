import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import plotly.io as pio
import researchpy as rp
import plotly.express as px
import sklearn as sk 

print(sk.__version__)


PATH = "/Users/webermac/Projekte/feature_importance/results_2025/kkb/rf/shuffled/run1/"
data = pd.read_json(PATH + "small_meta.json")
#model_data = pd.read_json("/home/maike/Covid/Covid_results/RandomForest/full_bochum/wilcoxon/1000runs/50thres/run1_balanced/meta.json")
print(data)
print(data.mean())
data_mean = data.mean()
#data_mean.to_csv(PATH + "mean_performance.csv")
#print(model_data)
#model_data.boxplot(fontsize=20)
plt.show()
columns = data.columns
pio.renderers.default= "browser"
fig = px.box(data)
#fig.update_layout(font_size=35)
#fig.write_image(PATH + "performance.svg")
fig.show()

def plot_hist(data):
    for column in columns:
        ax = data.plot.hist(column=column)
        plt.show()

def plot_boxplot(data):
    for column in columns:
        data.boxplot(column=column, fontsize=1000.0)
        plt.show()



def wilcoxon():
    print(stats.wilcoxon(data["auc"], model_data["auc"]))

def ttest():
    print(rp.ttest(data["auc"], model_data["auc"]))
    print(stats.ttest_ind(data["auc"], model_data["auc"]))


#plot_hist(data)
#plot_boxplot(data)