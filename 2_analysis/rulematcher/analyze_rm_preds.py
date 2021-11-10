import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

plt.rcParams["font.family"] = "Times New Roman"

PATH_TO_PREDS = "../../3_results/rulematcher/rm_preds/"
PRECISION_CONSTRAINT = 0.8


def make_title(filename: str):
    parts = filename.split("_")
    test_type = "Clean test data" if parts[-2] == "clean" else "Dirty test data"
    blocking = " (w/ blocking)" if parts[-1] == "blocking.csv" else " (w/o blocking)"
    return test_type + blocking


def locate_ax(a: int, b: int):
    if a == -1 and b == -1:
        a = 0
        b = 0
    elif a == 0 and b == 0:
        a = 1
    elif a == 1 and b == 0:
        a = 0
        b = 1
    else:
        a = 1
        b = 1

    return a, b


def maximize_f1(p, r):
    f = (2 * p * r) / (p + r)

    return np.argmax(f)


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
    fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    fig.set_size_inches(8, 8)
    axs[0, 0].set_ylabel("Precision", fontsize=12)
    axs[1, 0].set_ylabel("Precision", fontsize=12)
    axs[1, 0].set_xlabel("Recall", fontsize=12)
    axs[1, 1].set_xlabel("Recall", fontsize=12)
    a = -1  # to keep track of axis location
    b = -1

    for file in sorted(os.listdir(PATH_TO_PREDS)):
        print(file.upper())

        # Get data
        df = pd.read_csv(os.path.join(PATH_TO_PREDS, file))
        y_true = df["label"]
        y_score = df["match_score"]

        # Compute precision-recall data & AUC
        p, r, thresholds = precision_recall_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        print("AUC score:", round(auc, 3))

        # Select optimal threshold by maximizing F1
        ix = maximize_f1(p, r)
        t_max_f1 = thresholds[ix]

        # Select optimal threshold by maximizing recall under precision constraint
        ix = maximize_r_under_p_constr(p, r, constr=PRECISION_CONSTRAINT)
        t_max_r = thresholds[ix]

        # Plot P-R curve
        a, b = locate_ax(a, b)
        ax = axs[a, b]
        ax.plot(r, p, label="AUC = {:.3f}".format(auc))
        ax.fill_between(r, p, 0, alpha=0.2)
        ax.scatter(r[ix], p[ix], 10, zorder=999, marker="o", color="black",
                   label="optimal (t = {:.3f})".format(t_max_r))
        ax.plot([r[ix], r[ix]], [0, p[ix]], "k--", alpha=0.5)  # line from x to optimal threshold
        ax.plot([0, r[ix]], [p[ix], p[ix]], "k--", alpha=0.5)  # line from y to optimal threshold
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(make_title(file), fontsize=14)
        ax.legend(loc="lower right")

        # Metrics
        for t in [t_max_f1, t_max_r]:
            if t == t_max_f1:
                print("\nF1 MAXIMIZATION\n")
            else:
                print("\nRECALL MAXIMIZATION UNDER PRECISION CONSTRAINT\n")

            y_pred = y_score.apply(lambda x: 1 if x >= t else 0)  # predict again with selected threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            print("Threshold:", round(t, 3))
            print("TP:", tp)
            print("FP:", fp)
            print("TN:", tn)
            print("FN:", fn)
            print("Precision:", round(precision, 3))
            print("Recall:", round(recall, 3))
            print("F1:", round(f1, 3))
            print("Accuracy:", round(accuracy, 3))
            print("\n   ----------   \n")

    plt.tight_layout()
    plt.savefig("rm_pr_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
