import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import unidecode

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

SAMPLE = "sample_50k_balanced"


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


def make_normalized_score(jac_dist: float, threshold: float):
    return 0 if jac_dist > threshold else 1 - (jac_dist / threshold)


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


def compute_match_score(record_pair: pd.DataFrame, check_fingerprint: bool):
    score = 0
    total = 12

    # Compare fingerprints, then check matching rules
    if check_fingerprint:
        if record_pair["l_fingerprint"] != record_pair["r_fingerprint"]:
            record_pair["match_score"] = 0
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

    # Language of publication
    if record_pair["l_language"] == record_pair["r_language"]:
        score += 1

    # Place of publication
    if record_pair["l_country"] == record_pair["r_country"]:
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
    score += make_normalized_score(jac_dist, threshold=0.5)

    # Title
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_title_statement"], field="title").split("_")),
        set(normalize(record_pair["r_title_statement"], field="title").split("_"))
    )
    score += make_normalized_score(jac_dist, threshold=0.5)

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
    score += make_normalized_score(jac_dist, threshold=0.5)

    # Physical description
    jac_dist = nltk.jaccard_distance(
        set(normalize(record_pair["l_physical_description"], field="description").split("_")),
        set(normalize(record_pair["r_physical_description"], field="description").split("_"))
    )
    score += make_normalized_score(jac_dist, threshold=0.5)

    record_pair["match_score"] = score / total
    # print(record_pair["match_score"])

    return record_pair


def main():
    for file in ["test", "test_dirty"]:
        path = "../_data/ml/{0}/{1}.csv".format(SAMPLE, file)
        df = pd.read_csv(path, dtype=str)

        # Fingerprint analysis
        df = df.apply(get_fingerprint, axis=1)
        df = df.apply(is_fingerprint_match, axis=1)

        print("--- FINGERPRINT ANALYSIS ({}) ---\n".format(file))
        print("Fingerprint matches:")
        print(df["fp_match"].value_counts())

        g_fp_match = df[df["label"] == df["fp_match"].astype(str)].shape[0]
        cm = confusion_matrix(y_true=df["label"].astype(int),
                              y_pred=df["fp_match"].astype(int))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        print("Record pairs where fingerprint match label (0/1) matches the ground-truth label:\n{0} ({1}%)\n".format(
            g_fp_match,
            g_fp_match / len(df) * 100
        ))

        # Get matching scores
        for check_fp in (True, False):
            new_df = df.apply(compute_match_score, axis=1, args=(check_fp,))
            preds = new_df[["id", "label", "match_score"]]
            preds.to_csv(
                "rulematcher_predictions_{0}_{1}_{2}.csv".format(
                    SAMPLE,
                    "test_clean" if file == "test" else file,
                    "blocking" if check_fp else "noblocking"),
                index=False)


if __name__ == "__main__":
    main()
