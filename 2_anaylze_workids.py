import pandas as pd
import matplotlib as plt
import gzip
import re
from bs4 import BeautifulSoup

print("reading csv")
df = pd.read_csv("../../csv/workids_isbn_only.csv", dtype="str")

print("getting value counts")
counts = df["workid"].value_counts()[:500000]

print("stats")
print(counts.describe())
total = 0
for val, cnt in counts.iteritems():
    total += int(cnt)

print("nb record:", total)
print("filtering")
most_frequent = counts.index.tolist()

filtered = df[df["workid"].isin(most_frequent)]
# filtered.reset_index(inplace=True)
filtered.to_csv("filtered.csv", index=False)

# print("quantiles")
# for x in range(1,101):
#     print(x, counts.quantile(x/100))


# print("getting list of most frequent")
#

# print(most_frequent)

# print("getting duplicate ocns")
# dupes = df["ocn"].duplicated(keep=False)
#
# dupe_stats = dupes.value_counts()
# print(dupe_stats)


def get_record(offset):
    filename = "part-00000"
    offset = int(offset)
    with open(filename, "rb") as f:
        f.seek(offset)
        r = f.readline().decode("utf-8").strip()
    return r


def parse_record(record):
    m = re.search(r"<CDFRec>.*</CDFRec>", record)
    if m:
        dict_form = {}
        soup = BeautifulSoup(m.group(0), "lxml").find("cdfrec")
        for child_lvl_1 in soup.find_all(recursive=False):
            if child_lvl_1.name != "admin":
                field_code = child_lvl_1.name
                for child_lvl_2 in child_lvl_1.find_all(recursive=False):  # subfields
                    if child_lvl_2.name.startswith("s"):
                        sbf_code = re.sub(r"s", "", child_lvl_2.name)
                        tag = field_code + "$" + sbf_code
                        data = child_lvl_2.findChild("d").contents[0]

                        if tag not in dict_form:
                            dict_form[tag] = data
                        else:
                            dict_form[tag] += ";" + data

    return dict_form


# for fre in most_frequent:
#     cluster = df.loc[df["workid"] == fre]
#
#     print("NEW CLUSTER")
#     p = []
#     for index, row in cluster.iterrows():
#         record = get_record(row["offset"])
#         record_dict = parse_record(record)
#         if "v020$a" in record_dict:
#             isbns = record_dict["v020$a"].split(";")
#             p.append(isbns)
#
#     # print(p)
#
#         elements_in_all = list(set.intersection(*map(set, p)))
#
#         print(elements_in_all)


