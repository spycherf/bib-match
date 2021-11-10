import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"


def normalize(scores: list):
    return [1, scores[1] / scores[0]]  # 0 = clean, 1 = dirty


def main():
    # TODO: plug in the actual numbers
    scores = {
        "rm": {
            "auc": normalize([0.763, 0.692]),
            "recall": normalize([0.39, 0.128])
        },
        "dm": {
            "auc": normalize([0.879, 0.808]),
            "recall": normalize([0.812, 0.55])
        }
    }

    clean_df = pd.DataFrame({
        "metric": ["AUC", "Recall @ p ≈ 0.8"],
        "RuleMatcher": [scores["rm"]["auc"][0], scores["rm"]["recall"][0]],
        "DeepMatcher": [scores["dm"]["auc"][0], scores["dm"]["recall"][0]]
    })
    dirty_df = pd.DataFrame({
        "metric": ["AUC", "Recall @ p ≈ 0.8"],
        "RuleMatcher": [scores["rm"]["auc"][1], scores["rm"]["recall"][1]],
        "DeepMatcher": [scores["dm"]["auc"][1], scores["dm"]["recall"][1]]
    })

    fig, ax = plt.subplots(sharex="all", sharey="all")
    fig.set_size_inches(8, 4)
    ax.set_axisbelow(True)
    dirty_df.plot.bar(x="metric", ax=ax)
    clean_df.plot.bar(x="metric", ax=ax, alpha=0.4)
    plt.xticks(rotation='horizontal', fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.ylabel("Normalized score", fontsize=12)
    plt.legend(["RuleMatcher", "DeepMatcher"])
    plt.grid(axis="y", linestyle="--", zorder=0)
    plt.tight_layout()
    plt.savefig("rm_dm_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
