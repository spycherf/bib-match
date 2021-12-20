# Files and Usage

Once the ground truth is generated and formatted with the scripts from `0_data_prep`, the user can match record pairs found in the test sets using one of two methods: RuleMatcher or DeepMatcher. The first one is the ad hoc rule-based matching algorithm devised for the purpose of the thesis, whereas the second one is an open-source library available at [https://github.com/anhaidgroup/deepmatcher](https://github.com/anhaidgroup/deepmatcher).

* `rulematcher.py`: Predicts labels for the specified test set. The user should specify whether or not to use fingerprint blocking prior to matching.
* `deepmatcher_train.py`: Trains a model based on the specified sample and predicts labels for its respective test set. The user needs to specify the number of times the process should be run.
* `deepmatcher_test.py`: Can be used at a later stage to load a model and predict labels for other data.

For more information about DeepMatcher, please refer to the following article:

Mudgal, S., Li, H., Rekatsinas, T., Doan, A., Park, Y., Krishnan, G., Deep, R., Arcaute, E., & Raghavendra, V. (2018). Deep learning for entity matching: A design space exploration. *SIGMOD ’18: Proceedings of the 2018 International Conference on Management of Data*, 19–34. https://doi.org/10.1145/3183713.3196926
