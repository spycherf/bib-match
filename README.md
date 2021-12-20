# Rule-Based vs. Deep-Learning-Based Bibliographic Record Linkage

## Author

**Frederic Spycher**, Master student in Information Systems at University of Lausanne

## Context

This project was developed in relation to my Master's thesis:

*Towards a Learning-Based Approach of Improving Bibliographic Record Linkage*

## Thesis Abstract

Aggregators of library records like OCLC often have to deal poor-quality data. Such data can negatively impact the matching algorithm responsible for linking incoming records with the main database, thus leading to duplicates or unintended merges. To avoid this, data can go through a time-consuming cleaning process. Recent research in record linkage suggests that deep learning models are better suited for handling data dirtiness, as evidenced by the high F1 scores obtained by the authors. However, this research is based on data which is not as complex as the bibliographic records typically found in libraries. This thesis attempts to bridge this gap by testing the alleged performance of a state-of-the-art deep learning library, DeepMatcher, on bibliographic records from the WorldCat union catalog. The results are compared with an ad hoc rule-based algorithm, RuleMatcher. The link between data quality issues and record linkage performance is also studied. The experiment reveals that, while deep learning models indeed perform better overall, they do not show stronger resilience against dirty data, and they are more sensitive to certain types of data errors. To get these results, some methodological concessions must be made along the way, chief among which is the generation of artificial ground truth to compensate for the lack of existing ground truth for WorldCat bibliographic data. Suggestions are given to improve these pitfalls and pave the way for future research.

## Repository Structure

* `0_data_prep`: The scripts in this folder are used to go from the raw CDF XML records to the labeled ground truth needed for the project (in both its clean and dirty version).
* `1_matching`: This folder contains the scripts to predict whether record pairs are a match or not, using the two competing approaches: RuleMatcher and [DeepMatcher](https://github.com/anhaidgroup/deepmatcher).
* `2_analysis`: Various short scripts to analyze the predictions made by each approach.
