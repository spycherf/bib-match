# Files and Usage

The original raw data is stored as a flat-file database of CDF XML records, split over 400 files containing about a million records each. The scripts below are used in sequence to process this data into labeled record pairs (ground truth). Please refer to the thesis for a detailed explanation of this process. 

**Note 1**: In some scripts, the data is processed file by file. `combine_tsv.py` can be used to merge the 400 outputs into one. `tsv_to_csv.py` is another utility tool that simply converts a TSV file into a CSV file.

**Note 2**: The first three scripts could be refactored into one.

1. `extract_workids.py`: Parses the original records and generates a list of their file location, offset, OCN, and WorkID. The user can specify whether or not to filter out records without ISBN.
2. `retrieve_isbn.py`: For each record, reverse-sorts all available ISBNs and adds the first one to the list from (1).
3. `retrieve_bib_by_offset.py`: Completes the list from (2) with selected bibliographic features from the CDF records.
4. `generate_labels_by_cluster.py`: Cluster WorkIDs then labels a specified number of random record pairs within each cluster. The label is 1 (= match) if the records share the same ISBN and if their titles are similar enough; else, 0 (= no match). The ouptut is a list of OCN pairs and their respective label. The script also outputs an `uncertain.csv` file for record pairs sharing the same ISBN but having dissimilar titles. If they so wish, the user can use the `check_uncertain.py` script to manually label the pairs.
5. `prepare_training_data.py`: Generates a CSV of record pairs to fit DeepMatcher's format requirements, using the outputs of (3) and (4).
6. `train_val_test_split.py`: Generates samples of different sizes and splits each of them into three sets for training, validation, and testing. Sampling is done either randomly ("shuffle" mode) drawing from the entire ground truth, or sequentially ("balanced" mode) taking the first *n* observations, where *n* is the desired sample size.
7. `make_dirty.py`: Transforms a test set into its dirty version.
