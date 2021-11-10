import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def main():
    for method in ["rm", "dm"]:
        if method == "rm":
            clean_cm = [[2237, 1260],  # tp fp
                        [325, 1178]]   # fn tn
            dirty_cm = [[1711, 867],
                        [851, 1571]]
        else:
            clean_cm = [[2212, 678],
                        [350, 1760]]
            dirty_cm = [[1913, 725],
                        [649, 1713]]

        clean_cm_df = pd.DataFrame(clean_cm, index=["Match", "No match"], columns=["Match", "No match"])
        dirty_cm_df = pd.DataFrame(dirty_cm, index=["Match", "No match"], columns=["Match", "No match"])

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex="all", sharey="all")
        fig.set_size_inches(8, 4)
        cm1 = sn.heatmap(clean_cm_df, ax=ax1, vmin=0, vmax=2500, annot=True, fmt="g", cmap="Blues", cbar=False)
        cm1.set_title("Clean test data", fontsize=14)
        cm1.set_xlabel("Actual", fontsize=12)
        cm1.set_ylabel("Predicted", fontsize=12)
        cm1.set_yticklabels(["Match", "No match"], va="center")  # Fix to center the y-labels
        cm2 = sn.heatmap(dirty_cm_df, ax=ax2, vmin=0, vmax=2500, annot=True, fmt="g", cmap="Blues", cbar=False)
        cm2.set_title("Dirty test data", fontsize=14)
        cm2.set_xlabel("Actual", fontsize=12)
        plt.tight_layout()
        plt.savefig("{}_cm_dirty_clean.png".format(method), dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
