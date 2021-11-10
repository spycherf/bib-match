import math

import matplotlib.pyplot as plt
import pandas as pd
import pylab as py
import scipy.stats as stats

scores = pd.read_csv("../../results/dm_clean_vs_dirty/50k_balanced/scores_sample_50k_balanced.csv")
f1_val = scores["f1_val"]
f1_test = scores["f1_test"]
f1_test_dirty = scores["f1_test_dirty"]

size_val = len(f1_val)
size_test = len(f1_test)
size_test_dirty = len(f1_test_dirty)
min_val = f1_val.min(axis=0)
min_test = f1_test.min(axis=0)
min_test_dirty = f1_test_dirty.min(axis=0)
max_val = f1_val.max(axis=0)
max_test = f1_test.max(axis=0)
max_test_dirty = f1_test_dirty.max(axis=0)
mean_val = f1_val.mean(axis=0)
mean_test = f1_test.mean(axis=0)
mean_test_dirty = f1_test_dirty.mean(axis=0)
std_val = f1_val.std(axis=0)
std_test = f1_test.std(axis=0)
std_test_dirty = f1_test_dirty.std(axis=0)

print("Sample size:", size_val, size_test, size_test_dirty)
print("Min:", min_val, min_test, min_test_dirty)
print("Max:", max_val, max_test, max_test_dirty)
print("Mean:", mean_val, mean_test, mean_test_dirty)
print("Standard deviation:", std_val, std_test, std_test_dirty)

for z in [1.645, 1.96]:  # 1.96 = 95%, 1.645 = 90%
    interval_val = z * std_val / math.sqrt(size_val)
    lower_interval_val = mean_val - interval_val
    upper_interval_val = mean_val + interval_val

    interval_test = z * std_test / math.sqrt(size_test)
    lower_interval_test = mean_test - interval_test
    upper_interval_test = mean_test + interval_test

    interval_test_dirty = z * std_test_dirty / math.sqrt(size_test_dirty)
    lower_interval_test_dirty = mean_test_dirty - interval_test_dirty
    upper_interval_test_dirty = mean_test_dirty + interval_test_dirty

    print("Interval @ {}".format(z), interval_val, interval_test, interval_test_dirty)
    print("Lower bound", lower_interval_val, lower_interval_test, lower_interval_test_dirty)
    print("Upper bound", upper_interval_val, upper_interval_test, upper_interval_test_dirty)


# stats.probplot(f1_test, dist="norm", plot=py)
# py.show()
