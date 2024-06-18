import numpy as np
import scipy.stats as stats

# Generate two large samples with substantial differences
np.random.seed(0)  # For reproducibility
sample1 = np.random.normal(loc=10, scale=5, size=10000)  # Sample 1 from normal distribution
sample2 = np.random.normal(loc=100, scale=5, size=10000)  # Sample 2 from normal distribution, 10 times larger mean

# Perform Mann-Whitney U test
stat, p_value = stats.mannwhitneyu(sample1, sample2)

print(f"U statistic: {stat}")
print(f"P-value: {p_value}")
