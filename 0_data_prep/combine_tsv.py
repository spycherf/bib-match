import glob
import os

import pandas as pd

os.chdir("../data/match/ground_truth/parts")
extension = "tsv"
filenames = [f for f in sorted(glob.glob("*.{}".format(extension)))]

combined_tsv = pd.concat([pd.read_csv(f, sep="\t", encoding="utf-8") for f in filenames])
combined_tsv.to_csv("combined.tsv", sep="\t", index=False, encoding="utf-8")
