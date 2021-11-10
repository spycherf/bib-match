import deepmatcher as dm
import pandas as pd

from sklearn.metrics import f1_score

SAMPLE = "sample_50k_balanced"
# NB_RUNS = 30

# Process clean data
train, val, test = dm.data.process(
    path="deepmatcher_data/training/{}".format(SAMPLE),
    train="train.csv",
    validation="val.csv",
    test="test.csv",
    ignore_columns=("l_ocn", "r_ocn", "l_workid", "r_workid", "l_isbn", "r_isbn"),
    left_prefix="l_",
    right_prefix="r_",
    label_attr="label",
    id_attr="id",
    lowercase=True,  # depends on whether you used cased or uncased embeddings
    embeddings="fasttext.en.bin",  # fasttext.en.bin (uncased) / fasttext.crawl.vec (cased)
    embeddings_cache_path="deepmatcher_data/embeddings",
    pca=False,
    cache="cache",
    check_cached_data=True,
    auto_rebuild_cache=True
)

scores = pd.DataFrame(columns=["f1_val", "f1_test", "f1_test_dirty"])

for n in range(29, 30):
    print("\nRun", n)
    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.Hybrid(
            # word_contextualizer="lstm",  # optional: gru, lstm, rnn, self-attention
            # word_comparator="dot-attention",  # optional: decomposable-, general-, dot-attention
            # word_aggregator="birnn-last-pool"  # one of the styles from Pool with suffix "-pool"
        ),
        # attr_comparator="concat-abs-diff",
        # classifier="1-layer-leaky_relu-residual"  # <n>-layer-<non_linearity>-[residual, highway]
    )
    model.initialize(train)

    # Train model
    print("Training model...")
    f1_val = model.run_train(
        train,
        val,
        epochs=6,
        batch_size=16,
        best_save_path="deepmatcher_data/models/best_model_{0}_{1}".format(SAMPLE, str(n))
    )

    f1_test = model.run_eval(test, batch_size=16)

    # Process dirty data
    test_dirty = dm.data.process_unlabeled(
        path="deepmatcher_data/training/{}/test_dirty.csv".format(SAMPLE),
        trained_model=model,
        ignore_columns=("id", "label", "l_ocn", "r_ocn", "l_workid", "r_workid", "l_isbn", "r_isbn")
    )

    # Get predictions and F1 score for dirty data
    test_dirty_pred = model.run_prediction(test_dirty, output_attributes=True)
    y_true = pd.read_csv("deepmatcher_data/training/{}/test_dirty.csv".format(SAMPLE), index_col="id")["label"]
    test_dirty_pred["label"] = y_true.values
    y_pred = test_dirty_pred["match_score"].apply(lambda score: 1 if score >= 0.5 else 0)
    f1_test_dirty = f1_score(y_true, y_pred) * 100

    # Export scores
    scores.at[n, "f1_val"] = f1_val.item()
    scores.at[n, "f1_test"] = f1_test.item()
    scores.at[n, "f1_test_dirty"] = f1_test_dirty

    scores.to_csv("deepmatcher_data/logs/scores_{}.csv".format(SAMPLE), index=False)

    # Export predictions
    val_pred = model.run_prediction(val, output_attributes=True)
    val_pred[["label", "match_score"]].to_csv(
        "deepmatcher_data/logs/val_predictions_{0}_{1}.csv".format(SAMPLE, str(n)),
        index_label="id")

    test_pred = model.run_prediction(test, output_attributes=True)
    test_pred[["label", "match_score"]].to_csv(
        "deepmatcher_data/logs/test_predictions_{0}_{1}.csv".format(SAMPLE, str(n)),
        index_label="id")

    test_dirty_pred[["label", "match_score"]].to_csv(
        "deepmatcher_data/logs/test_dirty_predictions_{0}_{1}.csv".format(SAMPLE, str(n)),
        index_label="id")
