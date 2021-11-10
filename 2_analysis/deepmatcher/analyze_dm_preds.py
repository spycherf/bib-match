import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

plt.rcParams["font.family"] = "Times New Roman"

PATH_TO_PREDS = "../../3_results/deepmatcher/50k_balanced/dm_preds/"
BEST_MODEL = 17
PRECISION_CONSTRAINT = 0.8


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
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
    fig.set_size_inches(8, 4)
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax2.set_xlabel("Recall", fontsize=12)

    # Get results from the best model
    for test_type in ["test", "test_dirty"]:
        title = "Clean test data" if test_type == "test" else "Dirty test data"
        print(title.upper())

        # Loading probability scores from the best model
        df = pd.read_csv(os.path.join(PATH_TO_PREDS,
                                      "{0}_predictions_sample_50k_balanced_{1}.csv".format(test_type, BEST_MODEL)))
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

        # Plot P-R curves
        cmap = plt.cm.get_cmap("tab10")
        ax = ax1 if test_type == "test" else ax2
        ax.plot(r, p, color=cmap(0.1), label="AUC = {:.3f}".format(auc))
        ax.fill_between(r, p, 0, alpha=0.2, color=cmap(0.1))
        ax.scatter(r[ix], p[ix], 10, zorder=999, marker="o", color="black",
                   label="optimal (t = {:.3f})".format(t_max_r))
        ax.plot([r[ix], r[ix]], [0, p[ix]], "k--", alpha=0.5)  # line from x to optimal threshold
        ax.plot([0, r[ix]], [p[ix], p[ix]], "k--", alpha=0.5)  # line from y to optimal threshold
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower left")

        # Metrics
        for t in [t_max_f1, t_max_r]:
            if t == t_max_f1:
                print("\nF1 MAXIMIZATION\n")
            else:
                print("\nRECALL MAXIMIZATION UNDER PRECISION CONSTRAINT\n")

            y_pred = y_score.apply(lambda x: 1 if x >= 0.583 else 0)  # predict again with selected threshold
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
    plt.savefig("dm_pr_curves.png", dpi=150)
    plt.show()

    # Get results averaged from all 30 runs (need to specify threshold)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    dirty_precision_scores = []
    dirty_recall_scores = []
    dirty_f1_scores = []
    dirty_accuracy_scores = []

    for file in sorted(os.listdir(PATH_TO_PREDS)):
        df = pd.read_csv(os.path.join(PATH_TO_PREDS, file))
        y_true = df["label"]
        threshold = 0.705 if file.startswith("test_dirty") else 0.583
        y_pred = df["match_score"].apply(lambda x: 1 if x >= threshold else 0)

        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="binary", zero_division=0)
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        if file.startswith("test_dirty"):
            dirty_precision_scores.append(precision)
            dirty_recall_scores.append(recall)
            dirty_f1_scores.append(f1)
            dirty_accuracy_scores.append(accuracy)
        elif file.startswith("test_pred"):
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
        else:
            pass

    # Get averages
    precision_avg = sum(precision_scores) / len(precision_scores)
    recall_avg = sum(recall_scores) / len(recall_scores)
    f1_avg = sum(f1_scores) / len(f1_scores)
    accuracy_avg = sum(accuracy_scores) / len(accuracy_scores)

    dirty_precision_avg = sum(dirty_precision_scores) / len(dirty_precision_scores)
    dirty_recall_avg = sum(dirty_recall_scores) / len(dirty_recall_scores)
    dirty_f1_avg = sum(dirty_f1_scores) / len(dirty_f1_scores)
    dirty_accuracy_avg = sum(dirty_accuracy_scores) / len(dirty_accuracy_scores)

    print("\n--- METRICS AVERAGED FROM 30 RUNS (threshold picked with the max R under P constraint) ---")
    print("\n--- CLEAN TEST DATA ---")
    print("Precision: " + str(round(100 * precision_avg, 1)))
    print("Recall: " + str(round(100 * recall_avg, 1)))
    print("F1: " + str(round(100 * f1_avg, 1)))
    print("Accuracy: " + str(round(100 * accuracy_avg, 1)))

    print("\n--- DIRTY TEST DATA ---")
    print("Precision: " + str(round(100 * dirty_precision_avg, 1)))
    print("Recall: " + str(round(100 * dirty_recall_avg, 1)))
    print("F1: " + str(round(100 * dirty_f1_avg, 1)))
    print("Accuracy: " + str(round(100 * dirty_accuracy_avg, 1)))

    print("\n--- % DECREASE FROM CLEAN ---")
    print("Precision: " + str(round((precision_avg - dirty_precision_avg) / precision_avg * 100, 0)))
    print("Recall: " + str(round((recall_avg - dirty_recall_avg) / recall_avg * 100, 0)))
    print("F1: " + str(round((f1_avg - dirty_f1_avg) / f1_avg * 100, 0)))
    print("Accuracy: " + str(round((accuracy_avg - dirty_accuracy_avg) / accuracy_avg * 100, 0)))


if __name__ == "__main__":
    main()
