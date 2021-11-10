import argparse
import itertools
import os
import random

from fuzzywuzzy import fuzz
import pandas as pd

PID = str(os.getpid())
PATH_TO_IN = "workid_ocn_isbn.csv"
PATH_TO_OUT = "out"
PATH_TO_RECORDS = "bib"
PATH_TO_LOGS = "logs"

if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)
if not os.path.exists(PATH_TO_LOGS):
    os.makedirs(PATH_TO_LOGS)


def main():
    parser = argparse.ArgumentParser(description="Generate ground-truth data based on workID and ISBN")
    parser.add_argument("-r", "--range", action="store",
                        help="at which cluster index to start/end, e.g. 0-2")
    args = parser.parse_args()
    cluster_range = args.range

    if cluster_range:
        start, end = cluster_range.split("-")
        suffix = "_{0}-{1}".format(start, end)
    else:
        start = 0
        end = 0
        suffix = ""

    # Create paths
    output_file = os.path.join(PATH_TO_OUT, "labels{}.tsv".format(suffix))
    uncertain_log = os.path.join(PATH_TO_LOGS, "uncertain{}.log".format(suffix))
    processed_log = os.path.join(PATH_TO_LOGS, "processed{}.log".format(suffix))
    skipped_log = os.path.join(PATH_TO_LOGS, "skipped{}.log".format(suffix))

    # Retrieve already processed workIDs
    processed_workids = []
    if os.path.isfile(processed_log):
        with open(processed_log, "r") as log:
            for line in log.readlines():
                workid = line.split(",")[0]
                processed_workids.append(workid)

    print("Loading CSV...")
    df = pd.read_csv(PATH_TO_IN, dtype="str")

    header = ["ocn_l", "ocn_r", "match"]

    # Write header once
    if not os.path.isfile(output_file):
        labels = pd.DataFrame(columns=header, dtype="str")
        labels.to_csv(output_file, sep="\t", index=False, header=header)

    # Generate clusters
    print("Clustering...")
    clusters = df.groupby(["workid"])
    n_samples = 5  # number of samples per label within a workID cluster

    if cluster_range:
        start = int(start)
        end = int(end)
        nb_clusters = end - start
    else:
        nb_clusters = len(clusters)
        start = 0
        end = nb_clusters

    # Track progress
    processed = 0
    steps = [int(nb_clusters / 100 * pct) for pct in range(0, 100, 10)]

    # Matching
    for key, cluster in list(clusters)[start:end]:
        nb_records = cluster.shape[0]

        if processed in steps:
            print("Processed (PID {0}): {1} clusters out of {2}".format(PID, processed, nb_clusters))

        if key in processed_workids:
            processed += 1
            continue
        elif nb_records > 300:
            print("Skipping cluster {0} ({1} records)".format(key, str(nb_records)))
            with open(skipped_log, "a") as log:
                log.write("{0},{1}\n".format(key, str(nb_records)))
        else:
            pairs = cluster[["ocn"]].apply(lambda x: list(itertools.combinations(x.values, 2)))
            cluster_labels = pd.DataFrame(columns=header, dtype="str")

            for ocn_l, ocn_r in pairs["ocn"]:
                isbn_l = cluster[cluster["ocn"] == ocn_l]["isbn"].values[0]
                isbn_r = cluster[cluster["ocn"] == ocn_r]["isbn"].values[0]
                if isbn_l == isbn_r:
                    cluster_labels.loc[len(cluster_labels)] = [ocn_l, ocn_r, 1]
                else:
                    cluster_labels.loc[len(cluster_labels)] = [ocn_l, ocn_r, 0]

            # Select random matches
            match = cluster_labels[cluster_labels["match"] == 1].reset_index()
            match_sample = pd.DataFrame(columns=header, dtype="str")
            n = len(match) if len(match) < n_samples else n_samples
            for i in random.sample(range(0, len(match)), n):

                # Get records to compare title statements (fuzzy matching)
                ocn_l = match.at[i, "ocn_l"]
                ocn_r = match.at[i, "ocn_r"]
                filename_l = cluster[cluster["ocn"] == ocn_l]["file"].values[0].replace(".gz", "_bib_data.tsv")
                filename_r = cluster[cluster["ocn"] == ocn_r]["file"].values[0].replace(".gz", "_bib_data.tsv")
                file_l_df = pd.read_csv(os.path.join(PATH_TO_RECORDS, filename_l), sep="\t", dtype="str")
                file_r_df = pd.read_csv(os.path.join(PATH_TO_RECORDS, filename_r), sep="\t", dtype="str")
                record_l = file_l_df[file_l_df["ocn"] == ocn_l]
                record_r = file_r_df[file_r_df["ocn"] == ocn_r]
                title_l = record_l["title_statement"].values[0]
                title_r = record_r["title_statement"].values[0]

                similarity = fuzz.token_set_ratio(title_l, title_r)

                if similarity < 50:
                    with open(uncertain_log, "a", encoding="utf-8") as log:
                        log.write("{0},{1},{2},{3}\n{4}\n{5}\n".format(
                            filename_l, ocn_l, filename_r, ocn_r,
                            title_l, title_r)
                        )
                else:
                    match_sample.loc[len(match_sample)] = match.iloc[i]

            # Select random non-matches
            non_match = cluster_labels[cluster_labels["match"] == 0].reset_index()
            non_match_sample = pd.DataFrame(columns=header, dtype="str")
            n = len(non_match) if len(non_match) < n_samples else n_samples
            for i in random.sample(range(0, len(non_match)), n):
                non_match_sample.loc[len(non_match_sample)] = non_match.iloc[i]

            match_sample.to_csv(output_file, sep="\t", index=False, header=False, mode="a")
            non_match_sample.to_csv(output_file, sep="\t", index=False, header=False, mode="a")

        processed += 1
        with open(processed_log, "a") as log:
            log.write("{0},{1}\n".format(key, str(nb_records)))


if __name__ == "__main__":
    main()
