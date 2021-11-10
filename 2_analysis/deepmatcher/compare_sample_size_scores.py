import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame({
    "sample": ["1k", "5k", "10k", "50k"],
    "precision": [68.37, 75.69, 72.84, 79.79],
    "recall": [87.13, 75.48, 79.03, 82.02],
    "f1": [76.43, 75.5, 75.68, 80.87],
    "accuracy": [73.13, 75.5, 74.17, 80.12],
})

df.plot(x="sample", marker="o")
# plt.title("DeepMatcher: Performance across sample sizes")
plt.legend(["PRECISION", "RECALL", "F1", "ACCURACY"])
plt.xlabel("Sample size", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.tight_layout()
plt.show()
