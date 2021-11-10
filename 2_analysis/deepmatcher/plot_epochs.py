import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"


def main():
    # Setting figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
    fig.set_size_inches(8, 4)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("F1-score", fontsize=12)
    ax2.set_xlabel("Epochs", fontsize=12)

    for sample in ["1k", "5k"]:
        ax = ax1 if sample == "1k" else ax2
        df = pd.read_csv("../../3_results/deepmatcher/parameter_tuning/epochs/{}_balanced_30_epochs.csv".format(sample))
        eval_f1 = df[df["data"] == "EVAL"]["f1"].reset_index(drop=True)
        train_f1 = df[df["data"] == "TRAIN"]["f1"].reset_index(drop=True)

        ax.plot(train_f1, marker=".")
        ax.plot(eval_f1, marker=".")
        ax.set_title("{}-balanced".format(sample), fontsize=14)
        ax.legend(["TRAIN", "EVAL"], fontsize="small")

    plt.tight_layout()
    plt.savefig("epochs.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
