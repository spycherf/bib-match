import deepmatcher as dm
import pandas as pd

SAMPLE = "sample_1k_balanced"
NB_RUNS = 30
TEST_TYPE = "test_dirty"  # "test" (clean data) or "test_dirty"

# Process data
train, val, test = dm.data.process(
    path="deepmatcher_data/training/{}".format(SAMPLE),
    train="train.csv",
    validation="val.csv",
    test="{}.csv".format(TEST_TYPE),
    ignore_columns=("l_ocn", "r_ocn", "l_workid", "r_workid", "l_isbn", "r_isbn"),
    left_prefix="l_",
    right_prefix="r_",
    label_attr="label",
    id_attr="id",
    lowercase=True,
    embeddings="fasttext.en.bin",  # glove.42B.300d
    embeddings_cache_path="deepmatcher_data/embeddings",
    pca=False,
    cache="cache",
    check_cached_data=True,
    auto_rebuild_cache=True
)

scores = pd.DataFrame(columns=["f1_val", "f1_{}".format(TEST_TYPE)])

# Train model n times
for n in range(0, NB_RUNS):
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

    # Train and get scores
    f1_val = model.run_train(
        train,
        val,
        epochs=6,
        batch_size=16,
        best_save_path="deepmatcher_data/models/best_model_{0}_{1}".format(SAMPLE, str(n))
    )

    f1_test = model.run_eval(test)

    scores.at[n, "f1_val"] = f1_val.item()
    scores.at[n, "f1_{}".format(TEST_TYPE)] = f1_test.item()
    scores.to_csv("deepmatcher_data/logs/scores_{}.csv".format(SAMPLE), index=False)

    # Save predictions
    val_pred = model.run_prediction(val, output_attributes=True)
    val_pred[["label", "match_score"]].to_csv(
        "deepmatcher_data/logs/val_predictions_{0}_{1}.csv".format(SAMPLE, str(n)),
        index_label="id")
    test_pred = model.run_prediction(test, output_attributes=True)
    test_pred[["label", "match_score"]].to_csv(
        "deepmatcher_data/logs/{0}_predictions_{1}_{2}.csv".format(TEST_TYPE, SAMPLE, str(n)),
        index_label="id")

    print("Prediction score sample (from validation set):\n",
          val_pred[["label", "match_score"]].head(15))
