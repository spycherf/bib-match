import pandas as pd


def main():
    df = pd.read_csv("../data/match/ground_truth/first_1m.tsv", sep="\t", index_col="id", dtype="str")
    df.to_csv("../data/match/ground_truth/first_1m.csv", index_label="id")


if __name__ == "__main__":
    main()
