import numpy as np
import pandas as pd

MODE = "balance"
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2


def main():
    if MODE == "shuffle":
        df = pd.read_csv("../data/match/ground_truth/ground_truth.csv",
                         index_col="id",
                         dtype="str")
    else:
        df = pd.read_csv("../data/match/ground_truth/G_1m.csv",
                         # sep="\t",
                         index_col="id",
                         dtype="str")

    for n in [1000, 5000, 10000, 50000, 100000, 500000]:
        if MODE == "shuffle":
            df_sample = df.sample(n=n, random_state=SEED)
        else:
            df_sample = df.head(n)

        train, val, test = np.split(df_sample, [int(TRAIN_RATIO * len(df_sample)),
                                                int((TRAIN_RATIO + VAL_RATIO) * len(df_sample))])

        print("Size of train set:", train.shape[0])
        print("Size of validation set:", val.shape[0])
        print("Size of test set:", test.shape[0])

        file_prefix = "sample_{}_".format(str(n))
        train.to_csv("../out/{}train.csv".format(file_prefix), index_label="id")
        val.to_csv("../out/{}val.csv".format(file_prefix), index_label="id")
        test.to_csv("../out/{}test.csv".format(file_prefix), index_label="id")


if __name__ == "__main__":
    main()
