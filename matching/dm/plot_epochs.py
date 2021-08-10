import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../results/epochs/1k_balanced_30_epochs.csv")
eval_f1 = df[df["data"] == "EVAL"]["f1"].reset_index(drop=True)
train_f1 = df[df["data"] == "TRAIN"]["f1"].reset_index(drop=True)


df = pd.DataFrame({"train": train_f1, "eval": eval_f1})
print(df)
plt.plot(train_f1, marker=".")
plt.plot(eval_f1, marker=".")
plt.title("1k-balanced")
plt.legend(["TRAIN", "EVAL"])
plt.xlabel("Epochs")
plt.ylabel("F1")
plt.show()