import os

import pandas as pd

SAMPLE = "sample_50k_balanced"
PATH = os.path.join("../data/ml", SAMPLE)

df = pd.read_csv(os.path.join(PATH, "test.csv"), dtype="str", encoding="utf-8")

print("Record type")
print(df["l_rec_type"].value_counts())

print("Bib level")
print(df["l_bib_lvl"].value_counts())
