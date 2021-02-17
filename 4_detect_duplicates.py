import pandas as pd

print("Loading CSV...")
df = pd.read_csv("isbn.csv", dtype="str")
print("Nb of rows:")
print(df.shape[0])
dupes_unique = df.duplicated(subset="isbn", keep="first")
dupes_all = df.duplicated(subset="isbn", keep=False)
print("Nb of unique duplicated ISBN:")
print(dupes_unique.value_counts())
print("Total records with duplicated ISBN:")
print(dupes_all.value_counts())
print("Nb of unique values:")
print(df["isbn"].nunique())

# create new column for dupe/non-dupe
df["dpl"] = 1
dupes = df["dpl"].where(dupes_all, other=0)
df["dpl"] = dupes

print(df.tail(5))

grouped_df = df.groupby(["workid"])

for key, item in grouped_df:
    print(grouped_df.get_group(key), "\n\n")



