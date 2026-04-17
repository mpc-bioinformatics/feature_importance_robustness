import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/webermac/Projekte/feature_importance_all/data_2024/KKB_only.csv")


# overall missingness
overall = df.isna().mean() * 100

# missingness when DEATH == 0
death0 = df[df["DEATH"] == 0].isna().mean() * 100

# missingness when DEATH == 1
death1 = df[df["DEATH"] == 1].isna().mean() * 100

# combine into one dataframe
missing_df = pd.DataFrame({
    # The line `"column_name": df.columns` is creating a new column in the `missing_df` DataFrame
    # called "column_name" and populating it with the column names from the original DataFrame `df`.
    # This line is essentially extracting the column names from the `df` DataFrame and storing them in
    # the "column_name" column of the `missing_df` DataFrame for further analysis or visualization.
    "column_name": df.columns,
    "percent_missing_overall": overall.values,
    "percent_missing_DEATH0": death0.values,
    "percent_missing_DEATH1": death1.values
})

#missing_df.to_csv("/Users/webermac/Projekte/feature_importance_all/data_2024/missingness_KKB_only.csv")


overall = df.isna().mean() * 100
death0 = df[df["DEATH"] == 0].isna().mean() * 100
death1 = df[df["DEATH"] == 1].isna().mean() * 100

cols = df.columns
x = np.arange(len(cols))
width = 0.35

plt.figure()
plt.bar(x - width/2, death0, width, label="DEATH=0")
plt.bar(x + width/2, death1, width, label="DEATH=1")

plt.xticks(x, cols, rotation=90)
plt.ylabel("Percent Missing")
plt.title("Missingness by DEATH Group")
plt.legend()
plt.xlim(-0.5, len(cols) - 0.5)
plt.tight_layout()
plt.show()



# Percentage missing per column
missing_pct = df.isna().mean() * 100

# Define bins
bins = np.arange(0, 110, 10)  # 0,10,20,...,100

# Bin the missing percentages
binned = pd.cut(missing_pct, bins=bins, right=False)

# Count how many features fall into each bin
bin_counts = binned.value_counts().sort_index()

# Plot
plt.figure()
plt.bar(bin_counts.index.astype(str), bin_counts.values)

plt.xticks(rotation=45)
plt.xlabel("Missingness (%)")
plt.ylabel("Number of Features")
plt.title("Distribution of Feature Missingness")
plt.tight_layout()
plt.show()