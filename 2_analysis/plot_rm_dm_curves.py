import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

plt.rcParams["font.family"] = "Times New Roman"

PATH_TO_RM_PREDS = "../3_results/rulematcher/rm_preds/"
PATH_TO_DM_PREDS = "../3_results/deepmatcher/50k_balanced/dm_preds/"
BEST_DM_MODEL = 17
PRECISION_CONSTRAINT = 0.8


def maximize_r_under_p_constr(p, r, constr: float):
    r_best = 0
    t_ix = 0
    for i in range(0, len(p)):
        if p[i] >= constr:
            if r[i] > r_best:
                r_best = r[i]
                t_ix = i

    return t_ix


def main():
    # Setting figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
    fig.set_size_inches(8, 4)
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax2.set_xlabel("Recall", fontsize=12)

    for test_type in ["test_clean", "test_dirty"]:
        title = "Clean test data" if test_type == "test_clean" else "Dirty test data"

        rm_df = pd.read_csv(os.path.join(PATH_TO_RM_PREDS,
                                         "rulematcher_predictions_sample_50k_balanced_{}_noblocking.csv"
                                         .format(test_type)))
        dm_df = pd.read_csv(os.path.join(PATH_TO_DM_PREDS,
                                         "{0}_predictions_sample_50k_balanced_{1}.csv"
                                         .format("test" if test_type == "test_clean" else test_type, BEST_DM_MODEL)))

        for method in ["rm", "dm"]:
            df = rm_df if method == "rm" else dm_df
            legend_label = "RuleMatcher" if method == "rm" else "DeepMatcher"

            y_true = df["label"]
            y_score = df["match_score"]
            p, r, thresholds = precision_recall_curve(y_true, y_score)

            # Get index at optimal threshold
            ix = maximize_r_under_p_constr(p, r, constr=PRECISION_CONSTRAINT)

            # Plot superimposed P-R curves
            ax = ax1 if test_type == "test_clean" else ax2
            ax.plot(r, p, label=legend_label)
            ax.scatter(r[ix], p[ix], 10, zorder=999, marker="o", color="black")
            ax.plot([r[ix], r[ix]], [0, p[ix]], "k--", alpha=0.5)  # line from x to optimal threshold
            if method == "dm":
                ax.plot([0, r[ix]], [p[ix], p[ix]], "k--", alpha=0.5)  # line from y to optimal threshold
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_title(title, fontsize=14)
            ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    plt.savefig("dm_rm_pr_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
