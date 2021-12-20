# Files and Usage

The scripts found in this folder provide means to analyze the predictions made by RuleMatcher and DeepMatcher. The outputs of some of them are used in the Results section of the thesis. Some are specific to RuleMatcher or DeepMatcher, while others use the predictions from both approaches.

The most important are `rulematcher/analyze_rm_preds.py` and `deepmatcher/analyze_dm_preds.py`, which output precision/recall curves and select appropriate thresholds using either F1 maximization or recall maximization under a precision constraint. Various metrics are given for the selected threshold (precision, recall, F1 score, accuracy).
