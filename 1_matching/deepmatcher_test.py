import deepmatcher as dm
import pandas as pd

PATH_TO_SAMPLE = "../_data/ml/test_sample.csv"


def main():
    # Print records
    df = pd.read_csv(PATH_TO_SAMPLE)
    df.drop(["id", "label"], axis=1, inplace=True)

    print("Original incoming record:")
    print(df.iloc[0, :17].to_string())
    print("\nOriginal target record:")
    print(df.iloc[0, 17:].to_string())

    # Load model
    model = dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid())
    model.load_state("../_models/best_model_sample_50k_balanced_17")

    # Process data
    test = dm.data.process_unlabeled(
        path=PATH_TO_SAMPLE,
        trained_model=model,
        ignore_columns=("id", "label", "l_ocn", "r_ocn", "l_workid", "r_workid", "l_isbn", "r_isbn")
    )

    # Predict
    pred = model.run_prediction(test, output_attributes=True)
    for i, row in pred.iterrows():
        print("{0},{1}".format(i, row["match_score"]))


if __name__ == "__main__":
    main()
