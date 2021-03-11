import gzip
import os
import re
import shutil

import pandas as pd

PATH_TO_XWC = "/data/users/koopmanr/xwc"
PATH_TO_TEMP = "temp"
PATH_TO_IN = "isbn.csv"
PATH_TO_OUT = "out"

if not os.path.exists(PATH_TO_TEMP):
    os.makedirs(PATH_TO_TEMP)
if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)


def main():
    print("Loading CSV...")
    df = pd.read_csv(PATH_TO_IN, dtype="str")
    groups = df.groupby(["file"])

    for filename, group in groups:
        # Temp folder cleanup
        temp_files = [f for f in os.listdir(PATH_TO_TEMP)]
        for temp in temp_files:
            os.remove(os.path.join(PATH_TO_TEMP, temp))

        # Uncompress gzip
        print("Extracting {}...".format(filename))
        filepath = os.path.join(PATH_TO_XWC, filename)
        temp_file = os.path.join(PATH_TO_TEMP, filename.replace(".gz", ""))
        with gzip.open(filepath, "rb") as f_in:
            with open(temp_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Create empty DataFrame for the bibliographic data
        bib_data = pd.DataFrame(columns=["ocn",
                                         "workid",
                                         "isbn",
                                         "rec_type",
                                         "bib_lvl",
                                         "form",
                                         "date_1",
                                         "date_2",
                                         "language",
                                         "country",
                                         "language_cataloging",
                                         "main_author",
                                         "added_authors",
                                         "title_statement",
                                         "edition_statement",
                                         "publishing_info",
                                         "physical_description"],
                                dtype="str")

        # Track progress
        nb_records = group.shape[0]
        processed = 0
        last_offset = 0
        steps = [int(nb_records / 100 * pct) for pct in range(0, 100, 10)]

        # Retrieve bibliographic data
        remove_tags = r"</d>(<Authority>.*?</Authority>)?</.*?>(<.*?>)?(<d>)?"

        with open(temp_file, "rb") as f:
            for i, row in group.iterrows():
                if processed in steps:
                    print("Processed: {0} records out of {1}".format(processed, nb_records))

                offset = int(row["offset"])
                f.seek(offset - last_offset, 1)
                record = f.readline().decode("utf-8", "strict")
                last_offset = f.tell()

                m = re.search(r"<CDFRec>(?P<cdfrec>.*)</CDFRec>", record)

                if m:
                    record = m.group("cdfrec")

                    # OCN, work ID & ISBN
                    bib_data.loc[i, "ocn"] = row["ocn"]
                    bib_data.loc[i, "workid"] = row["workid"]
                    bib_data.loc[i, "isbn"] = row["isbn"]

                    # Record type & bibliographic level
                    a = re.search(r"<a>.(?P<rec_type>.)(?P<bib_lvl>.).*?</a>", record)
                    if a:
                        bib_data.loc[i, "rec_type"] = a.group("rec_type")
                        bib_data.loc[i, "bib_lvl"] = a.group("bib_lvl")

                    # Form, date(s), country & language
                    c008 = re.search(
                        r"<c008>.{7}(?P<date_1>.{4})(?P<date_2>.{4})(?P<country>.{3}).{5}(?P<form>.).{11}(?P<language>.{3})",
                        record)
                    if c008:
                        bib_data.loc[i, "form"] = c008.group("form")
                        bib_data.loc[i, "date_1"] = c008.group("date_1")
                        bib_data.loc[i, "date_2"] = c008.group("date_2")
                        bib_data.loc[i, "country"] = c008.group("country")
                        bib_data.loc[i, "language"] = c008.group("language")

                    # Language of cataloging
                    v040 = re.search(r"<v040.*?<sb><d>(?P<language_cataloging>.*?)</d>", record)
                    if v040:
                        bib_data.loc[i, "language_cataloging"] = v040.group("language_cataloging")

                    # Authors (main & added)
                    v1xx = re.findall(r"<v1[01]0.*?<sa><d>(.*?)</d>", record)
                    if v1xx:
                        main_author = ""
                        for entry in v1xx:
                            if len(main_author) > 0:
                                main_author += " | "
                            main_author += entry
                        bib_data.loc[i, "main_author"] = main_author

                    v7xx = re.findall(r"<v7[01]0.*?<sa><d>(.*?)</d>", record)
                    if v7xx:
                        added_authors = ""
                        for entry in v7xx:
                            if len(added_authors) > 0:
                                added_authors += " | "
                            added_authors += entry
                        bib_data.loc[i, "added_authors"] = added_authors

                    # Title statement
                    v245 = re.search(r"<v245.*?<d>(?P<title_statement>.*?)</v245>", record)
                    if v245:
                        title_statement = re.sub(remove_tags, " ", v245.group("title_statement"))
                        bib_data.loc[i, "title_statement"] = title_statement

                    # Edition statement
                    v250 = re.findall(r"<v250.*?<d>(.*?)</v250>", record)
                    if v250:
                        edition_statement = ""
                        for entry in v250:
                            if len(edition_statement) > 0:
                                edition_statement += "| "
                            edition_statement += re.sub(remove_tags, " ", entry)
                        bib_data.loc[i, "edition_statement"] = edition_statement

                    # Publishing information
                    v26x = re.findall(r"<v26[04].*?<d>(.*?)</v26[04]>", record)
                    if v26x:
                        publishing_info = ""
                        for entry in v26x:
                            if len(publishing_info) > 0:
                                publishing_info += "| "
                            publishing_info += re.sub(remove_tags, " ", entry)
                        bib_data.loc[i, "publishing_info"] = publishing_info

                    # Physical description
                    v300 = re.findall(r"<v300.*?<d>(.*?)</v300>", record)
                    physical_description = ""

                    if v300:
                        for entry in v300:
                            if len(physical_description) > 0:
                                physical_description += "| "
                            physical_description += re.sub(remove_tags, " ", entry)
                        bib_data.loc[i, "physical_description"] = physical_description

                processed += 1

        print("Exporting bibliographic data...")
        export_file = os.path.join(PATH_TO_OUT, filename.replace(".gz", "") + "_bib_data.tsv")
        bib_data.to_csv(export_file, sep="\t", index=False)


if __name__ == "__main__":
    main()
