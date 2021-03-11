import csv
import os
import re

import pandas as pd

PATH_TO_JISC = "../qual-check/in/jisc"


def main():
    for file in sorted(os.listdir(PATH_TO_JISC)):
        print("Parsing {}...".format(file))
        filepath = os.path.join(PATH_TO_JISC, file)
        output_file = file + "_ocn.csv"
        jisc_ocn = pd.DataFrame(columns=["file", "offset", "ocn"], dtype="str")

        with open(filepath, "rb") as f:
            offset = 0

            for line in f:
                row = line.decode("utf-8", "strict").lower()

                if row.startswith("\x1d"):
                    offset = f.tell()

                # Look for OCN
                m = re.search(r"\x1e035.*?\x1fa\(ocolc\)(o[cn][mn]?)?(?P<ocn>\d+)", row)

                if m:
                    ocn = m.group("ocn")
                    ocn = str(int(ocn))  # remove leading zeros
                    jisc_ocn = jisc_ocn.append({"file": file, "offset": offset, "ocn": ocn}, ignore_index=True)

        jisc_ocn.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
