import csv
import os

import pandas as pd

TOOL = "deepmatcher"
PATH_TO_OUT = "../out"

if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)


def get_chunk(dataframe: pd.DataFrame, chunk_size: int, start_row: int = 0):
    end_row  = min(start_row + chunk_size, dataframe.shape[0])

    return dataframe.iloc[start_row:end_row, :]


def main():
    print("Loading BIB data...")
    bib_df = pd.read_csv("../data/bib/bib.tsv", sep="\t", encoding="utf-8")

    print("Loading labels...")
    labels_df = pd.read_csv("../data/match/labels.tsv", sep="\t")

    for n in range(4800000, 5000000, 100000):
        start = n
        to_process = 100000
        labels_chunk = get_chunk(labels_df, chunk_size=to_process, start_row=start)

        output_file = os.path.join(PATH_TO_OUT, "merged_{}-{}.tsv".format(str(start), str(start + to_process)))
        header = None

        if TOOL == "deepmatcher":
            print("Processing {0} record pairs, starting at {1}".format(str(to_process), str(start)))

            for i, row in labels_chunk.iterrows():
                if i % 1000 == 0:
                    print("Processing record pair #{}".format(str(i)))

                # Retrieve both records
                ocn_l = row["ocn_l"]
                record_l = bib_df[bib_df["ocn"] == ocn_l]
                record_l = record_l.add_prefix("l_")
                record_l.reset_index(drop=True, inplace=True)

                ocn_r = row["ocn_r"]
                record_r = bib_df[bib_df["ocn"] == ocn_r]
                record_r = record_r.add_prefix("r_")
                record_r.reset_index(drop=True, inplace=True)

                # Merge
                new_row = pd.concat([record_l, record_r], axis=1)
                new_row.insert(0, "id", str(i))
                new_row.insert(1, "label", row["match"])

                # Save
                if header is None:
                    header = list(new_row.columns)

                with open(output_file, "a", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, delimiter="\t", fieldnames=header)
                    if os.stat(output_file).st_size == 0:
                        writer.writeheader()
                    writer.writerow(new_row.iloc[0].to_dict())


if __name__ == "__main__":
    main()
