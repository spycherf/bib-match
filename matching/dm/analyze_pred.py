import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

df = pd.read_csv("../../results/predictions_example.csv")

y_true = df["label"]
y_scores = df["match_score"]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print("Number of thresholds:", len(thresholds))

plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
