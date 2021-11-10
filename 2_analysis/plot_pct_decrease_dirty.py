import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame({
    "metric": ["Precision", "Recall", "F1", "Accuracy"],
    "rulematcher": [5, 3, 4, 4],
    "deepmatcher": [6, 22, 15, 12]
})

ax = df.plot.bar(x="metric")
ax.set_axisbelow(True)
plt.xlabel("Metric", fontsize=14)
plt.ylabel("% decrease", fontsize=14)
plt.ylim([0, 25])
plt.grid(which="major", axis="y", linestyle="--")
plt.tight_layout()
plt.legend(["RULEMATCHER", "DEEPMATCHER"], loc="upper right")
plt.savefig("rm_dm_dirty_decrease.png", dpi=150)
plt.show()
