import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

df = pd.read_csv("../../results/_predictions/sample_5k_balanced_16_clean/predictions_sample_5k_balanced_0.csv")
df["prediction"] = df["match_score"].apply(lambda x: 1 if x >= 0.5 else 0)

y_true = df["label"]
y_scores = df["match_score"]
y_pred = df["prediction"]

print("Precision:", precision_score(y_true=y_true, y_pred=y_pred))
print("Recall:", recall_score(y_true=y_true, y_pred=y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print("Plotting P-R curve...")
print("Number of probability thresholds:", len(thresholds))

plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
