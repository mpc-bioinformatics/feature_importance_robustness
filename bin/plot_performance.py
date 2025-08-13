import pandas as pd
import plotly.io as pio
import plotly.express as px
import sklearn as sk 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to result folder of a training run")
args = parser.parse_args()

PATH = args.input
data = pd.read_json(PATH + "small_meta.json")

data_mean = data.mean()
data_mean.to_csv(PATH + "mean_performance.csv")

columns = data.columns
pio.renderers.default= "browser"
fig = px.box(data)
fig.update_layout(font_size=35)
fig.write_image(PATH + "performance.svg")
fig.show()
