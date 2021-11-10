import pandas as pd

scores = pd.read_csv("deepmatcher_data/logs/scores_sample_50k_balanced.csv")
scores["avg"] = 0.0

for i, row in scores.iterrows():
    scores.at[i, "avg"] = (scores.at[i, "f1_val"] + scores.at[i, "f1_test"] + scores.at[i, "f1_test_dirty"]) / 3

scores = scores.sort_values(by="avg", ascending="False")
print(scores)
