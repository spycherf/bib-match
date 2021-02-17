import gzip
import os
import re
import shutil

from bs4 import BeautifulSoup
import pandas as pd

PATH_TO_XWC = "/data/users/koopmanr/xwc"
PATH_TO_TEMP = "../temp"
PATH_TO_OUT = "../out"

if not os.path.exists(PATH_TO_TEMP):
    os.makedirs(PATH_TO_TEMP)
if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)


def main():
    print("Loading CSV...")
    df = pd.read_csv("filtered.csv", dtype="str")
    df["isbn"] = None

    # Track progress
    nb_records = df.shape[0]
    processed = 0
    steps = [int(df.shape[0] / 100 * pct) for pct in range(0, 100)]

    current_file = ""

    for i, row in df.iterrows():
        if processed in steps:
            print("Processed: {0} out of {1}".format(processed, nb_records))

        filename = df.at[i, "file"]
        offset = int(df.at[i, "offset"])

        if filename == current_file:
            filepath = os.path.join(PATH_TO_TEMP, filename.replace(".gz", ""))
        else:
            current_file = filename
            filepath = os.path.join(PATH_TO_XWC, filename)

            # Empty temporary folder
            temp_files = [f for f in os.listdir(PATH_TO_TEMP)]
            for temp in temp_files:
                os.remove(os.path.join(PATH_TO_TEMP, temp))

            # Uncompress gzip
            print("Extracting {}...".format(filename))
            temp = os.path.join(PATH_TO_TEMP, filename.replace(".gz", ""))

            with gzip.open(filepath, "rb") as f_in:
                with open(temp, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            filepath = temp

        # Retrieve ISBN
        with open(filepath, "rb") as f:
            f.seek(offset)
            record = f.readline().decode("utf-8", "strict")
            m = re.search(r"<CDFRec>.*</CDFRec>", record)
            soup = BeautifulSoup(m.group(0), "lxml").find("cdfrec")
            isbn_fields = soup.find_all("v020")
            all_isbn = []

            for field in isbn_fields:
                correct_isbn = field.find("sa")
                if correct_isbn:
                    isbn = correct_isbn.findChild("d").contents[0].upper()
                    isbn = re.sub(r"[^\dX]", "", isbn)
                    all_isbn.append(isbn)

            all_isbn.sort(reverse=True)
            df.at[i, "isbn"] = all_isbn[0]

        processed += 1

    df.to_csv("isbn.csv", index=False)


if __name__ == "__main__":
    main()
