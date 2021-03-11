import math

import pandas as pd
import pylab as py
import scipy.stats as stats

z = 1.96  # 1.96 = 95%, 1.645 = 90%
scores = pd.read_csv("../../results/sample_size/scores_sample_50k_balanced_batch_16.csv")
f1_val = scores["f1_val"]
f1_test = scores["f1_test"]

size_val = len(f1_val)
size_test = len(f1_test)
min_val = f1_val.min(axis=0)
min_test = f1_test.min(axis=0)
max_val = f1_val.max(axis=0)
max_test = f1_test.max(axis=0)
mean_val = f1_val.mean(axis=0)
mean_test = f1_test.mean(axis=0)
std_val = f1_val.std(axis=0)
std_test = f1_test.std(axis=0)

print("Sample size:", size_val, size_test)
print("Min:", min_val, min_test)
print("Max:", max_val, max_test)
print("Mean:", mean_val, mean_test)
print("Standard deviation:", std_val, std_test)

interval_val = z * std_val / math.sqrt(size_val)
lower_interval_val = mean_val - interval_val
upper_interval_val = mean_val + interval_val

interval_test = z * std_test / math.sqrt(size_test)
lower_interval_test = mean_test - interval_test
upper_interval_test = mean_test + interval_test

print("Interval", interval_val, interval_test)
print("Lower bound", lower_interval_val, lower_interval_test)
print("Upper bound", upper_interval_val, upper_interval_test)

stats.probplot(f1_val, dist="norm", plot=py)
py.show()

stats.probplot(f1_test, dist="norm", plot=py)
py.show()
