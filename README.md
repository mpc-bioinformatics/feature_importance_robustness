# feature_importance_robustness

Machine learning models are on the rise in medical research fields yet are rarely implemented in the everyday clinical practice. This is mostly due to a lack of trust of humans towards the artificial intelligence. An approach to solve this issue is performed by so called feature importance methods, which try to explain the model decision making. There are several methods for feature importance, post hoc ones like permutation feature importance (PFI) and SHapley Additive exPlanations (SHAP) and intrinsic ones like mean decrease in impurity (MDI). Here, we train random forest models to predict 30 day survival and compare those feature importance methods on a real-world Sepsis dataset and check how similar their results are and how robust the most important features are in three different scenarios. The first explores the intrinsic randomness of a random forest model, the second variations introduced by different training set splits, and the third explores hidden patterns in a nonsense model.
Our results show that SHAP and MDI perform in a similar matter, whilst PFI shows the greatest variance in feature ranks. All methods differ in their 10 key features based on the chosen test set. Our suggested approach is to combine several feature importance methods and to keep in mind that a feature might not stay important if another slice of the data is used for training.  



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
