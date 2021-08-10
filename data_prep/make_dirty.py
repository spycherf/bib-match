import os
import random
import string

import numpy as np
import pandas as pd

SAMPLE = "sample_1k_shuffled"
PATH = os.path.join("../data/ml", SAMPLE)


def add_typo(s: str):
    rand = random.randrange(0, len(s))
    while not s[rand].isalpha() and rand > 0:
        rand -= 1
    letter = "Z" if s[rand].lower() != "z" else "X"
    return s[0:rand] + letter + s[rand+1:]


def remove_n_tokens(s: str, n: int):
    tokens = s.split()
    if len(tokens) > 2:
        for _ in range(0, n):
            rand = random.randrange(0, len(tokens))
            while tokens[rand] in string.punctuation and rand > 0:
                rand -= 1
            tokens.remove(tokens[rand])

    return " ".join([token for token in tokens])


def main():
    df = pd.read_csv(os.path.join(PATH, "test.csv"), dtype="str", encoding="utf-8", index_col="id")

    for i, row in df.iterrows():
        # Record type
        df.at[i, "l_rec_type"] = "a"

        # Bib level
        df.at[i, "l_bib_lvl"] = "m"

        # Form
        df.at[i, "l_form"] = " "

        # Date 1
        date = row["l_date_1"]
        if date != "    ":
            if random.random() < 0.7:
                pass
            else:
                if random.random() < 0.5:
                    df.at[i, "l_date_1"] = "    "
                else:
                    if date.isnumeric():
                        df.at[i, "l_date_1"] = str(int(date) - 1)

        # Date 2
        df.at[i, "l_date_2"] = "    "

        # Language
        if random.random() < 0.5:
            pass
        else:
            df.at[i, "l_language"] = "   "

        # Place of publication
        if random.random() < 0.5:
            pass
        else:
            df.at[i, "l_country"] = "   "

        # Main author entry
        s = row["l_main_author"]
        if s is not np.nan:
            if random.random() < 0.7:
                pass
            else:
                if random.random() < 0.5:
                    s = row["l_main_author"].split(",")
                    if len(s) > 1:
                        df.at[i, "l_main_author"] = s[1] + " " + s[0]
                else:
                    df.at[i, "l_main_author"] = add_typo(row["l_main_author"])

        # Added entries
        s = row["l_added_authors"]
        if s is not np.nan:
            if random.random() < 0.5:
                pass
            else:
                if random.random() < 0.5:
                    df.at[i, "l_added_authors"] = ""
                else:
                    s = s.split("|")
                    if len(s) > 1:
                        halved = s[0:int(len(s) / 2)]
                        df.at[i, "l_added_authors"] = " ".join([token for token in halved])

        # Title statement
        s = row["l_title_statement"]
        if s is not np.nan:
            if random.random() < 0.7:
                pass
            else:
                sub_start = s.find(":")
                resp_start = s.find("/")
                if resp_start != -1:
                    if sub_start != -1:
                        s = s[0:sub_start] + s[resp_start:]
                    else:
                        s = s[0:resp_start]
            if random.random() < 0.05:
                s = add_typo(s)

            df.at[i, "l_title_statement"] = s

        # Edition statement
        s = row["l_edition_statement"]
        if s is not np.nan:
            if random.random() < 0.5:
                pass
            else:
                df.at[i, "l_edition_statement"] = ""
                if random.random() < 0.5:
                    df.at[i, "l_title_statement"] += " " + s

        # Publishing information
        s = row["l_publishing_info"]
        if s is not np.nan:
            if random.random() < 0.5:
                pass
            else:
                if random.random() < 0.1:
                    s = "[S.l.] : [s.n.], [s.d.]"
                else:
                    n = 1 if random.random() < 0.5 else 2
                    s = remove_n_tokens(s, n=n)
            if random.random() < 0.05:
                s = add_typo(s)

            df.at[i, "l_publishing_info"] = s

        # Physical description
        s = row["l_physical_description"]
        if s is not np.nan:
            df.at[i, "l_physical_description"] = s.replace("pages", "p.")
            df.at[i, "l_physical_description"] = s.replace("Seiten", "S.")
            if random.random() < 0.3:
                pass
            else:
                detail_start = s.find(":")
                dim_start = s.find(";")

                if dim_start != -1:
                    df.at[i, "l_physical_description"] = s[0:dim_start]

                if detail_start != -1:
                    df.at[i, "l_physical_description"] = s[0:detail_start]

    df.to_csv(os.path.join(PATH, "test_dirty.csv"), index_label="id")


if __name__ == "__main__":
    main()
