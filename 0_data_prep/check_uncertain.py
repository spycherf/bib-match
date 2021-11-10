import os

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

PATH_TO_BIB = "../_results/bib"
PATH_TO_OUT = "../out"
PATH_TO_LOGS = "../logs"

if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)
if not os.path.exists(PATH_TO_LOGS):
    os.makedirs(PATH_TO_LOGS)


def main():
    df = pd.read_csv("../_results/match/uncertain.csv", dtype="str")
    output_file = os.path.join(PATH_TO_OUT, "labels_reviewed.tsv")
    processed_log = os.path.join(PATH_TO_LOGS, "processed.log")
    still_uncertain_log = os.path.join(PATH_TO_LOGS, "uncertain_final.csv")

    # Write headers once
    if not os.path.isfile(output_file):
        header = ["ocn_l", "ocn_r", "match"]
        labels_df = pd.DataFrame(columns=header, dtype="str")
        labels_df.to_csv(output_file, sep="\t", index=False, header=header)

    if not os.path.isfile(still_uncertain_log):
        header = ["file_l", "ocn_l", "file_r", "ocn_r"]
        uncertain_df = pd.DataFrame(columns=header, dtype="str")
        uncertain_df.to_csv(still_uncertain_log, index=False, header=header)

    # Load processed IDs
    if os.path.isfile(processed_log):
        with open(processed_log, "r") as log:
            processed = [line.strip("\n") for line in log.readlines()]
    else:
        processed = []

    for i, row in df.iterrows():
        if str(i) in processed:
            continue

        print("Processing index {}...".format(str(i)))

        # Retrieve records
        ocn_l = row["ocn_l"]
        filepath_l = os.path.join(PATH_TO_BIB, row["file_l"])
        df_l = pd.read_csv(filepath_l, sep="\t", dtype="str")
        record_l = df_l[df_l["ocn"] == ocn_l]

        ocn_r = row["ocn_r"]
        filepath_r = os.path.join(PATH_TO_BIB, row["file_r"])
        df_r = pd.read_csv(filepath_r, sep="\t", dtype="str")
        record_r = df_r[df_r["ocn"] == ocn_r]

        # Display records
        print("rtyp:\t\t{0}\t\t{1}"
              .format(record_l["rec_type"].values[0], record_r["rec_type"].values[0]))
        print("blvl:\t\t{0}\t\t{1}"
              .format(record_l["bib_lvl"].values[0], record_r["bib_lvl"].values[0]))
        print("form:\t\t{0}\t\t{1}"
              .format(record_l["form"].values[0], record_r["form"].values[0]))
        print("dat1:\t\t{0}\t\t{1}"
              .format(record_l["date_1"].values[0], record_r["date_1"].values[0]))
        print("dat2:\t\t{0}\t\t{1}"
              .format(record_l["date_2"].values[0], record_r["date_2"].values[0]))
        print("lang:\t\t{0}\t\t{1}"
              .format(record_l["language"].values[0], record_r["language"].values[0]))
        print("ctry:\t\t{0}\t\t{1}"
              .format(record_l["country"].values[0], record_r["country"].values[0]))
        print("lcat:\t\t{0}\t\t{1}"
              .format(record_l["language_cataloging"].values[0], record_r["language_cataloging"].values[0]))
        print("aut1:\t\t{0}\t\t{1}"
              .format(record_l["main_author"].values[0], record_r["main_author"].values[0]))
        print("aut2:\t\t{0}\t\t{1}"
              .format(record_l["added_authors"].values[0], record_r["added_authors"].values[0]))
        print("titl:\t\t{0}\t\t{1}"
              .format(record_l["title_statement"].values[0], record_r["title_statement"].values[0]))
        print("edit:\t\t{0}\t\t{1}"
              .format(record_l["edition_statement"].values[0], record_r["edition_statement"].values[0]))
        print("publ:\t\t{0}\t\t{1}"
              .format(record_l["publishing_info"].values[0], record_r["publishing_info"].values[0]))
        print("desc:\t\t{0}\t\t{1}"
              .format(record_l["physical_description"].values[0], record_r["physical_description"].values[0]))

        # Prompt user
        user_feedback = ""
        while user_feedback not in ["y", "n", "u", "q"]:
            user_feedback = input("Match?\n\nY(es)\nN(o)\nU(ncertain)\nQ(uit)\n\n").lower()

        if user_feedback == "y":  # match
            with open(output_file, "a") as out:
                out.write("{0}\t{1}\t{2}\n".format(ocn_l, ocn_r, "1"))
        elif user_feedback == "n":  # not match
            with open(output_file, "a") as out:
                out.write("{0}\t{1}\t{2}\n".format(ocn_l, ocn_r, "0"))
        elif user_feedback == "u":  # uncertain
            with open(still_uncertain_log, "a") as log:
                log.write("{0},{1},{2},{3}\n".format(row["file_l"], ocn_l, row["file_r"], ocn_r))
        elif user_feedback == "q":  # quit
            break

        # Log process
        with open(processed_log, "a", encoding="utf-8") as log:
            log.write(str(i) + "\n")


if __name__ == "__main__":
    main()
