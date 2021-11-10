import os
import contextlib
import random
import string

import numpy as np
import pandas as pd
import deepmatcher as dm

from mojito import Mojito, chart

PATH_TO_DATA = "dm/deepmatcher_data/training/sample_1k_balanced/test.csv"
PATH_TO_MODEL = "../results/dm_clean_vs_dirty/1k_balanced/best_model_sample_1k_balanced_17"


def wrap_dm(model, ignore_columns=("label", "id")):
    def wrapper(dataframe):
        data = dataframe.copy().drop([c for c in ignore_columns if c in dataframe.columns], axis=1)

        data["id"] = np.arange(len(dataframe))

        tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
        data.to_csv(tmp_name, index=False)

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                data_processed = dm.data.process_unlabeled(tmp_name, trained_model=model)
                out_proba = model.run_prediction(data_processed, output_attributes=True)
                out_proba = out_proba["match_score"].values.reshape(-1)

        multi_proba = np.dstack((1 - out_proba, out_proba)).squeeze()

        os.remove(tmp_name)
        return multi_proba

    return wrapper


df = pd.read_csv(PATH_TO_DATA, dtype=str)
model = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid())
model.load_state(PATH_TO_MODEL)
proba = wrap_dm(model)(df)
tp_group = df[(proba[:, 1] >= 0.5) & (df["label"] == "1")]
tn_group = df[(proba[:, 0] >= 0.5) & (df["label"] == "0")]

len(tp_group), len(tn_group)

###

mojito = Mojito(df.columns,
                attr_to_copy="left",
                split_expression=" ",
                class_names=['no_match', 'match'],
                feature_selection="lasso_path")

tp_result = mojito.drop(wrap_dm(model),
                        tp_group,
                        num_features=20,
                        num_perturbation=100)

tn_result = mojito.copy(wrap_dm(model),
                        tn_group,
                        num_features=20,
                        num_perturbation=100)

chart(tp_result,(1,1,1),(-1,1),title="match-true-positives")
chart(tn_result,(1,1,1),(-1,1),title="match-true-negatives")

tp_result.to_csv('dblp_acm_mojito_positives.csv', index=False)
tn_result.to_csv('dblp_acm_mojito_negatives.csv', index=False)
