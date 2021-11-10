import matplotlib.pyplot as plt
import pandas as pd

file = "../3_results/rulematcher/rm_preds/rulematcher_predictions_sample_50k_balanced_test_clean_blocking.csv"

df = pd.read_csv(file, index_col="id")
print(df.dtypes)
filtered_df = df[(df["match_score"] > 0) & (df["label"] == 0)]

print(filtered_df.describe())
