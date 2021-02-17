import csv
import gzip
import os
import re
import shutil

ISBN_ONLY = True

PATH_TO_XWC = "/data/users/koopmanr/xwc"
PATH_TO_TEMP = "../temp"
PATH_TO_OUT = "../out"

if not os.path.exists(PATH_TO_TEMP):
    os.makedirs(PATH_TO_TEMP)
if not os.path.exists(PATH_TO_OUT):
    os.makedirs(PATH_TO_OUT)


def main():
    header = ["file", "offset", "ocn", "workid"]

    for file in sorted(os.listdir(PATH_TO_XWC)):
        filepath = os.path.join(PATH_TO_XWC, file)
        output_file = os.path.join(PATH_TO_OUT, os.path.splitext(file)[0] + ".csv")

        if filepath.endswith(".gz"):

            # Empty temporary folder
            temp_files = [f for f in os.listdir(PATH_TO_TEMP)]
            for temp in temp_files:
                os.remove(os.path.join(PATH_TO_TEMP, temp))

            # Uncompress gzip
            print("Extracting {}...".format(file))
            temp = os.path.join(PATH_TO_TEMP, file.replace(".gz", ""))

            with gzip.open(filepath, "rb") as f_in:
                with open(temp, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            filepath = temp

            print("Parsing {0}...".format(file))

            with open(filepath, "rb") as f:
                offset = 0

                for line in f:
                    row = line.decode("utf-8", "strict")

                    if ISBN_ONLY:
                        m = re.search(r"^(\w+)\t.*<\/sa><\/v020>.*(?<=<workId>)(\w+)", row)
                    else:
                        m = re.search(r"^(\w+)\t.*(?<=<workId>)(\w+)", row)
                    if m:
                        data = {
                            "file": file,
                            "offset": offset,
                            "ocn": m[1],
                            "workid": m[2]
                        }

                        with open(output_file, "a", newline="", encoding="utf8") as out:
                            writer = csv.DictWriter(out, fieldnames=header)
                            if os.stat(output_file).st_size == 0:
                                writer.writeheader()
                            writer.writerow(data)

                    offset = f.tell()


if __name__ == "__main__":
    main()
