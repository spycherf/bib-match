import pandas as pd


def get_tp(df: pd.DataFrame):
    tp = df[(df["label"] == 1) & (df["is_match"] == 1)]
    return tp.shape[0], tp.index.tolist()


def get_fp(df: pd.DataFrame):
    fp = df[(df["label"] == 0) & (df["is_match"] == 1)]
    return fp.shape[0], fp.index.tolist()


def get_tn(df: pd.DataFrame):
    tn = df[(df["label"] == 0) & (df["is_match"] == 0)]
    return tn.shape[0], tn.index.tolist()


def get_fn(df: pd.DataFrame):
    fn = df[(df["label"] == 1) & (df["is_match"] == 0)]
    return fn.shape[0], fn.index.tolist()


def main():
    # RULEMATCHER
    rm_preds = pd.read_csv("../../3_results/rulematcher/rm_preds/rulematcher_predictions_sample_50k_balanced_test_clean_noblocking.csv",
                           index_col="id")
    rm_preds_dirty = pd.read_csv("../../3_results/rulematcher/rm_preds/rulematcher_predictions_sample_50k_balanced_test_dirty_noblocking.csv",
                                 index_col="id")

    # Determine predicted labels for clean and dirty datasets
    rm_threshold_clean = 0.75
    rm_threshold_dirty = 0.75

    rm_preds["is_match"] = rm_preds["match_score"].apply(lambda x: 1 if x >= rm_threshold_clean else 0)
    rm_preds_dirty["is_match"] = rm_preds_dirty["match_score"].apply(lambda x: 1 if x >= rm_threshold_dirty else 0)

    rm_tp_count, rm_tp_list = get_tp(rm_preds)
    rm_fp_count, rm_fp_list = get_fp(rm_preds)
    rm_tn_count, rm_tn_list = get_tn(rm_preds)
    rm_fn_count, rm_fn_list = get_fn(rm_preds)
    rm_tp_count_dirty, rm_tp_list_dirty = get_tp(rm_preds_dirty)
    rm_fp_count_dirty, rm_fp_list_dirty = get_fp(rm_preds_dirty)
    rm_tn_count_dirty, rm_tn_list_dirty = get_tn(rm_preds_dirty)
    rm_fn_count_dirty, rm_fn_list_dirty = get_fn(rm_preds_dirty)

    # DEEPMATCHER
    dm_preds = pd.read_csv("../../3_results/deepmatcher/50k_balanced/dm_preds/test_predictions_sample_50k_balanced_17.csv",
                           index_col="id")
    dm_preds_dirty = pd.read_csv("../../3_results/deepmatcher/50k_balanced/dm_preds/test_dirty_predictions_sample_50k_balanced_17.csv",
                                 index_col="id")

    # Determine predicted labels for clean and dirty datasets
    dm_threshold_clean = 0.583
    dm_threshold_dirty = 0.583
    dm_preds["is_match"] = dm_preds["match_score"].apply(lambda x: 1 if x >= dm_threshold_clean else 0)
    dm_preds_dirty["is_match"] = dm_preds_dirty["match_score"].apply(lambda x: 1 if x >= dm_threshold_dirty else 0)

    dm_tp_count, dm_tp_list = get_tp(dm_preds)
    dm_fp_count, dm_fp_list = get_fp(dm_preds)
    dm_tn_count, dm_tn_list = get_tn(dm_preds)
    dm_fn_count, dm_fn_list = get_fn(dm_preds)
    dm_tp_count_dirty, dm_tp_list_dirty = get_tp(dm_preds_dirty)
    dm_fp_count_dirty, dm_fp_list_dirty = get_fp(dm_preds_dirty)
    dm_tn_count_dirty, dm_tn_list_dirty = get_tn(dm_preds_dirty)
    dm_fn_count_dirty, dm_fn_list_dirty = get_fn(dm_preds_dirty)

    # Count TP, FP, TN and FN
    counts_clean = pd.DataFrame(
        columns=["tp_rm", "tp_dm", "tp_shared",
                 "fp_rm", "fp_dm", "fp_shared",
                 "tn_rm", "tn_dm", "tn_shared",
                 "fn_rm", "fn_dm", "fn_shared"]
    )
    counts_dirty = pd.DataFrame(
        columns=["tp_rm", "tp_dm", "tp_shared",
                 "fp_rm", "fp_dm", "fp_shared",
                 "tn_rm", "tn_dm", "tn_shared",
                 "fn_rm", "fn_dm", "fn_shared"]
    )
    counts_clean.at[0, "tp_rm"] = rm_tp_count
    counts_clean.at[0, "fp_rm"] = rm_fp_count
    counts_clean.at[0, "tn_rm"] = rm_tn_count
    counts_clean.at[0, "fn_rm"] = rm_fn_count
    counts_clean.at[0, "tp_dm"] = dm_tp_count
    counts_clean.at[0, "fp_dm"] = dm_fp_count
    counts_clean.at[0, "tn_dm"] = dm_tn_count
    counts_clean.at[0, "fn_dm"] = dm_fn_count
    counts_clean.at[0, "tp_shared"] = len(list(set(rm_tp_list).intersection(dm_tp_list)))
    counts_clean.at[0, "fp_shared"] = len(list(set(rm_fp_list).intersection(dm_fp_list)))
    counts_clean.at[0, "tn_shared"] = len(list(set(rm_tn_list).intersection(dm_tn_list)))
    counts_clean.at[0, "fn_shared"] = len(list(set(rm_fn_list).intersection(dm_fn_list)))
    counts_dirty.at[0, "tp_rm"] = rm_tp_count_dirty
    counts_dirty.at[0, "fp_rm"] = rm_fp_count_dirty
    counts_dirty.at[0, "tn_rm"] = rm_tn_count_dirty
    counts_dirty.at[0, "fn_rm"] = rm_fn_count_dirty
    counts_dirty.at[0, "tp_dm"] = dm_tp_count_dirty
    counts_dirty.at[0, "fp_dm"] = dm_fp_count_dirty
    counts_dirty.at[0, "tn_dm"] = dm_tn_count_dirty
    counts_dirty.at[0, "fn_dm"] = dm_fn_count_dirty
    counts_dirty.at[0, "tp_shared"] = len(list(set(rm_tp_list_dirty).intersection(dm_tp_list_dirty)))
    counts_dirty.at[0, "fp_shared"] = len(list(set(rm_fp_list_dirty).intersection(dm_fp_list_dirty)))
    counts_dirty.at[0, "tn_shared"] = len(list(set(rm_tn_list_dirty).intersection(dm_tn_list_dirty)))
    counts_dirty.at[0, "fn_shared"] = len(list(set(rm_fn_list_dirty).intersection(dm_fn_list_dirty)))

    counts_clean.to_csv("classification_report_clean.csv", index=False)
    counts_dirty.to_csv("classification_report_dirty.csv", index=False)

    # Get indices TP that became FN with dirty data in both RuleMatcher and DeepMatcher
    rm_tp_to_fn = sorted(list(set(rm_tp_list).intersection(set(rm_fn_list_dirty))))
    dm_tp_to_fn = sorted(list(set(dm_tp_list).intersection(set(dm_fn_list_dirty))))
    shared_tp_to_fn = sorted(list(set(rm_tp_to_fn).intersection(set(dm_tp_to_fn))))
    print("RuleMatcher TP to FN\t\t{0}\nDeepMatcher TP to FN\t\t{1}\nShared TP to FN\t\t\t\t{2}"
          .format(len(rm_tp_to_fn), len(dm_tp_to_fn), len(shared_tp_to_fn)))

    df = pd.DataFrame(columns=["rm_score_clean", "rm_score_dirty", "dm_prob_clean", "dm_prob_dirty"],
                      index=shared_tp_to_fn)

    for i, row in df.iterrows():
        df.at[i, "rm_score_clean"] = rm_preds.loc[i, "match_score"]
        df.at[i, "rm_score_dirty"] = rm_preds_dirty.loc[i, "match_score"]
        df.at[i, "dm_prob_clean"] = dm_preds.loc[i, "match_score"]
        df.at[i, "dm_prob_dirty"] = dm_preds_dirty.loc[i, "match_score"]

    df.to_csv("shared_tp_to_fn.csv", index_label="id")


if __name__ == "__main__":
    main()
