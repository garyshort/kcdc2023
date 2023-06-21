import numpy as np

# 5-point description of the distribution
min_val = 1
q25 = 2
median = 3
q75 = 4
max_val = 5

# Generate a sample that approximately matches the given 5-point summary
sample_size = 10000
sample = np.concatenate([
    np.random.uniform(min_val, q25, size=int(sample_size * 0.25)),
    np.random.uniform(q25, median, size=int(sample_size * 0.25)),
    np.random.uniform(median, q75, size=int(sample_size * 0.25)),
    np.random.uniform(q75, max_val, size=int(sample_size * 0.25)),
])

# Estimate the mean of the distribution
estimated_mean = np.mean(sample)
print(f"Estimated mean: {estimated_mean}")
