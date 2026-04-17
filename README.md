# feature_importance_robustness

Background:
Machine learning or artificial intelligence (AI) models are on the rise in medical research fields yet are rarely implemented in the everyday clinical practice. This is mostly due to a lack of trust of humans towards the “black box” that provides answers without their derivation. An approach to solve this issue are feature importance methods, falling under the umbrella term eXplainable Artifical Intelligence (XAI), which try to provide human-interpretable evidence for the model's decision making. There are several methods for determining feature importance, as in post hoc methods like permutation feature importance (PFI) and SHapley Additive exPlanations (SHAP) as well as intrinsic methods like mean decrease in impurity (MDI). However, it is not well tested how similar the results of those methods are.
Methods:
Here, we train random forest models to predict 30 day survival of sepsis patients in the ICU and evaluate the behavior three feature importance methods (SHAP, PFI, MDI) on a real-world data set. This is assessed by three scenarios: First, models are trained on a fixed data split to isolate the effect of randomness within the forest itself. Secondly, varying data splits are used to capture both model randomness and dataset heterogeneity. Thirdly, labels are randomly shuffled as a negative control, where no real signal exists and all features should appear equally important.
Findings:
Our results show that PFI is more sensible and displays a higher variance, but is not affected by the cardinality of features or categorical features. Our nonsense models revealed tendencies for SHAP and MDI to favor high cardinal features and disfavor categorical features. Using a ranked approach for feature importance in cross-validation achieved the most overlap of key features for all three methods. 
Interpretation:
Our suggested approach is to apply several feature importance methods and apply them to a cross-validation to acquire more robust key features, instead of using a single split of the data set and therefore improve the trust clinicians have in those methods.




## Training of the random forest for classification of 30 day survival in Sepsis

### Overview

Trains random forest models via a 1000 times Monte Carlo Cross-Validation (MCCV) to predict whether or not a patient in the ICU would survive 30 days. 
Performs three feature importance methods (PFI, SHAP, MDI). 

#### Input

file containing the clinical information of patients and the target variable (survival). 

#### Output

results folder containing:
- k files for each method containing the feature importance values (in our scenario, 3 x 1000 csv files)
- 'meta.json' - a file containing meta information, e.g. hyperparameters used, which input file, target variable, etc. 
- 'small_meta.json' - a json file containing evaluation metrics of the model performance

#### Usage

'python random_forest.py --input path/to/clinical_data.csv --output /path/to/output/'

## Performance Summary & Visualization Script ("plot_performance.py")

### Overview
Reads performance metrics from `small_meta.json`, computes mean values, and generates a box plot.  


#### Input 
folder containing the small_meta.json created by the "random_forest.py" script.

#### Output 
- 'mean_performance.csv' - mean of each evaluation metric. 
- 'performance.svg' - boxplot of the evaluation metrics. 

#### Usage: 

' python plot_performance.py --input path/to/result/folder/'

Note that the path must end with /. (e.g. ./results/). 

## Calculation of feature importance ranks ("rank_calc.py)

### Overview
Reads the feature importance values from the k-fold MCCV and turns them into ranks and calculates the median 
rank of each feature. Should contain all three methods (permutation feature importance, mean decrease in impurity, 
SHAP)

#### Input
folder containing the k (default 1000)  feature importance value files (.csv) created by the "random_forest.py" script. 

#### Output
For all three feature importance methods:
- '*_median.csv' - median rank of each feature
- '*_box_sorted.svg' - box plot sorted by median rank

#### Usage

'python rank_calc.py --input path/to/result/folder/'
