import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"


def main():
    df = pd.read_csv("../../3_results/dq_analysis/dq_issues_impact.csv")
    df.fillna(value=0, inplace=True)
    print(df.describe())

    # Score decrease boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
    fig.set_size_inches(8, 4)
    ax1.boxplot(df[["rm_score_diff"]])
    ax1.set_title("RuleMatcher", fontsize=14)
    ax1.set_ylim([0.0, 1.0])
    ax2.boxplot(df[["dm_score_diff"]])
    ax2.set_title("DeepMatcher", fontsize=14)
    plt.xticks([])

    # plt.tight_layout()
    # plt.show()
    # fig.savefig("score_decrease_boxplots.png", dpi=150)

    # Correlation between score difference (RM/DM) and the number of errors
    df1 = df.drop(["id", "rm_score_clean", "rm_score_dirty", "dm_prob_clean", "dm_prob_dirty"], axis=1)
    # df1 = df[["nb_errors", "rm_score_diff", "dm_score_diff"]]
    df1_corr = df1.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    ax.set_xticklabels(xlabels, size=13)
    ax.set_yticklabels(ylabels, size=13)
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(df1_corr, cmap=cmap, center=0, vmin=-1.0, xticklabels=df1_corr.columns, yticklabels=df1_corr.columns)

    # plt.tight_layout()
    # plt.show()
    # fig.savefig("correlation_heatmap.png", dpi=150)

    # Correlations for records pairs with 2 errors
    df2 = df1[df1["nb_errors"] == 2]
    df2.drop(["nb_errors", "rm_score_diff"], axis=1, inplace=True)
    df2 = df2.loc[:, (df2 != 0).any(axis=0)]
    df2_corr = df2.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    ax.set_xticklabels(xlabels, size=13)
    ax.set_yticklabels(ylabels, size=13)
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(df2_corr, cmap=cmap, center=0, vmin=-1.0, xticklabels=df2_corr.columns, yticklabels=df2_corr.columns)

    plt.tight_layout()
    plt.show()
    fig.savefig("two_errors_correlation.png", dpi=150)


if __name__ == "__main__":
    main()
