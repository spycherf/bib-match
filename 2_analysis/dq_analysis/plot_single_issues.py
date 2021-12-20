import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"


def main():
    df = pd.read_csv("../../3_results/dq_analysis/single_sample_scores.csv").sort_values(by="dm_score", ascending=False)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    plt.xticks(rotation=90)
    plt.grid()

    for method in ["rm_score", "dm_score"]:
        ax.plot(df["id"].to_list(), df[method].to_list(), linewidth=3)
        ax.set_ylim([0.7, 1.0])
        ax.legend(["RuleMatcher", "DeepMatcher"], fontsize="large")

    plt.tight_layout()
    plt.savefig("issues_single_sample.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
