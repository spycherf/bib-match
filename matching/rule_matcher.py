import re

import nltk
import pandas as pd
import unidecode

SAMPLE = "sample_1k_shuffled"
PATH_TO_SAMPLE = "../data/ml/{}/train.csv".format(SAMPLE)
CHECK_FINGERPRINT = False
THRESHOLD = 0.7


def extract_date(string: str):
    string = "" if type(string) != str else string
    m = re.findall(r"[\d?]{4}", string)

    return m[-1] if m else "0000"


def roman_to_int(s: re.Match):
    s = s.group()
    rom_val = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]

    return str(int_val)


def normalize(string: str, field: str):
    if type(string) == str:
        string = string.lower()
        string = string.replace("&", "and")
        string = unidecode.unidecode(string)

        if field == "author":
            string = re.sub(r"(?<=\d)-", "", string)
            string = re.sub(r"[\d.,|]+", "", string)
            string = re.sub(r"[(<{\[].*?[)>}\]]", "", string)
        elif field == "title":
            string = re.match(r"(.*?)(?=/|$)", string).group()
            string = re.sub(r"[^\w\s]", "", string)
        elif field in ["edition", "description"]:
            string = re.sub(r"\b[ivxlcdm]+\b", roman_to_int, string)
            string = re.sub(r"[^\d\s]", "", string)
        elif field == "publisher":
            string = re.sub(r"[^\w\s]|\d", "", string)

        return re.sub(r"\s+", "_", string.strip())
    else:
        return ""


def get_initials(author: str):
    initials = []
    names = author.split("_")
    for name in names:
        i = name[0].lower()
        if not i.isnumeric():
            initials.append(name[0].lower())

    return initials


def ini_sim(author_1, author_2):
    a = get_initials(author_1)
    b = get_initials(author_2)
    m = len(a) - 1
    n = len(b) - 1

    if m == 0:
        a += "?"

    if n == 0:
        b += "?"

    if (a[0] == b[0] and a[m] == b[n]) \
            or (a[0] == b[1] and a[m] == b[0]) \
            or (a[0] == b[0] and a[1] == b[1]) \
            or (a[0] == b[n] and a[1] == b[0]):
        return True
    else:
        return False


def get_string_hash(string: str):
    null = ".0."

    if type(string) == str:
        field_hash = ""
        tokens = re.findall(r"[\w'-]+", string.lower())
        sorted_tokens = sorted(tokens, key=len, reverse=True)
        first_three = sorted_tokens[:3]
        missing = 3 - len(first_three)
        for token in first_three:
            field_hash += (token[0] + str(len(token) % 7) + token[-1])
        field_hash += missing * null

        return field_hash
    else:
        return 3 * null


def get_fingerprint(record_pair: pd.DataFrame):
    prefixes = ["l_", "r_"]
    for prefix in prefixes:
        fingerprint = ""

        # Title part
        title = re.match(r"(.*?)(?=[/:]|$)", record_pair["{}title_statement".format(prefix)]).group()
        fingerprint += get_string_hash(title)

        # Author part
        author = record_pair["{}main_author".format(prefix)]
        fingerprint += get_string_hash(author)

        # Date part
        fingerprint += extract_date(record_pair["{}publishing_info".format(prefix)])

        record_pair["{}fingerprint".format(prefix)] = fingerprint

    return record_pair


def is_fingerprint_match(record_pair: pd.DataFrame):
    l_fp = record_pair["l_fingerprint"]
    r_fp = record_pair["r_fingerprint"]
    if l_fp == r_fp:
        record_pair["fp_match"] = 1
    else:
        record_pair["fp_match"] = 0

    return record_pair


def is_match(record_pair: pd.DataFrame):
    score = 0
    total = 11

    # Compare fingerprints, then check matching rules
    if CHECK_FINGERPRINT:
        if record_pair["l_fingerprint"] != record_pair["r_fingerprint"]:
            record_pair["is_match"] = 0
            return record_pair

    # Record type
    if record_pair["l_rec_type"] == record_pair["r_rec_type"]:
        score += 1

    # Bibliographic level
    if record_pair["l_bib_lvl"] == record_pair["r_bib_lvl"]:
        score += 1

    # Form
    if record_pair["l_form"] == record_pair["r_form"]:
        score += 1

    # Date
    if record_pair["l_date_1"].isnumeric():
        l_date = record_pair["l_date_1"]
    else:
        l_date = extract_date(record_pair["l_publishing_info"])

    if record_pair["r_date_1"].isnumeric():
        r_date = record_pair["r_date_1"]
    else:
        r_date = extract_date(record_pair["r_publishing_info"])

    if l_date.isnumeric() and r_date.isnumeric():
        if (int(r_date) + 1) >= int(l_date) >= (int(r_date) - 1):
            score += 1

    # Place of publication
    if record_pair["l_country"] == record_pair["l_country"]:
        score += 1

    # Main author entry
    l_author = normalize(record_pair["l_main_author"], field="author")
    r_author = normalize(record_pair["r_main_author"], field="author")
    edit_dist = nltk.edit_distance(l_author, r_author)

    if len(l_author) > 0 and len(r_author) > 0:
        if edit_dist <= 0.3 or ini_sim(l_author, r_author):
            score += 1

    # Added author entries
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_added_authors"], field="author").split("_")),
        set(normalize(record_pair["r_added_authors"], field="author").split("_"))
    )
    if jac_dist <= 0.5:
        score += 1

    # Title
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_title_statement"], field="title").split("_")),
        set(normalize(record_pair["r_title_statement"], field="title").split("_"))
    )
    if jac_dist <= 0.5:
        score += 1

    # Edition statement
    l_edition = normalize(record_pair["l_edition_statement"], field="edition")
    r_edition = normalize(record_pair["r_edition_statement"], field="edition")

    if l_edition == r_edition:
        score += 1

    # Publishing information
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_publishing_info"], field="publisher").split("_")),
        set(normalize(record_pair["r_publishing_info"], field="publisher").split("_"))
    )
    if jac_dist <= 0.5:
        score += 1

    # Physical description
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_physical_description"], field="description").split("_")),
        set(normalize(record_pair["r_physical_description"], field="description").split("_"))
    )
    if jac_dist <= 0.5:
        score += 1

    if score / total >= THRESHOLD:
        record_pair["is_match"] = 1
    else:
        record_pair["is_match"] = 0
    
    return record_pair


def compute_metrics(dataframe: pd.DataFrame):
    true_pos = dataframe[(dataframe["label"].astype(int) == 1) & (dataframe["is_match"] == 1)].shape[0]
    true_neg = dataframe[(dataframe["label"].astype(int) == 0) & (dataframe["is_match"] == 0)].shape[0]
    false_pos = dataframe[(dataframe["label"].astype(int) == 0) & (dataframe["is_match"] == 1)].shape[0]
    false_neg = dataframe[(dataframe["label"].astype(int) == 1) & (dataframe["is_match"] == 0)].shape[0]
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("True positives:", true_pos)
    print("True negatives:", true_neg)
    print("False positives:", false_pos)
    print("False negatives:", false_neg)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)


def main():
    df = pd.read_csv(PATH_TO_SAMPLE, dtype=str)

    # Get fingerprints
    df = df.apply(get_fingerprint, axis=1)
    df = df.apply(is_fingerprint_match, axis=1)

    # Check matching rules
    df = df.apply(is_match, axis=1)

    # Results
    print("Fingerprint matches:")
    print(df["fp_match"].value_counts())
    print("Confirmed matches:")
    print(df["is_match"].value_counts())
    g_fp_match = df[df["label"] == df["fp_match"].astype(str)].shape[0]
    print("Number of record pairs where fingerprint matches ground-truth:\n{0} ({1}%)\n".format(
        g_fp_match,
        g_fp_match / len(df) * 100
    ))
    print("Metrics:")
    compute_metrics(df)


if __name__ == "__main__":
    main()




