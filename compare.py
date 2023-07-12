import numpy as np
from scipy.stats import wasserstein_distance

# Assuming pred and sim are NumPy arrays or lists
pred, sim
# Calculate the mean difference over standard deviation
mean_diff_std = np.mean(pred - sim) / np.std(pred)

# Calculate the median difference over M_sim
median_diff_M_sim = np.median(pred - sim) / np.median(sim)

# Calculate the ratio of standard deviations (std_pred / std_sim)
std_ratio = np.std(pred) / np.std(sim)

# Calculate the Wasserstein distance
wasserstein_dist = wasserstein_distance(pred, sim)

# Print the results
print("Mean difference over standard deviation:", mean_diff_std)
print("Median difference over M_sim:", median_diff_M_sim)
print("Ratio of standard deviations (std_pred / std_sim):", std_ratio)
print("Wasserstein distance:", wasserstein_dist)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, ttest_ind, mannwhitneyu

# Function to generate prediction distribution
def generate_prediction_distribution(a, b):
    # Your prediction algorithm here
    # Return a probability distribution (e.g., numpy array) for the given parameters

# Function to generate simulation distribution
def generate_simulation_distribution(a, b):
    # Your simulation algorithm here
    # Return a probability distribution (e.g., numpy array) for the given parameters

# Define the sets of parameters [a, b]
parameter_sets = [[1, 2], [3, 4], [5, 6], [7, 8]]  # Update with your parameter sets

# Initialize lists to store the results
ks_scores = []
wasserstein_distances = []
ttest_p_values = []
mannwhitneyu_p_values = []

# Iterate over each parameter set
for params in parameter_sets:
    a, b = params

    # Generate the prediction and simulation distributions
    prediction_dist = generate_prediction_distribution(a, b)
    simulation_dist = generate_simulation_distribution(a, b)

    # Calculate the scores using different methods
    ks_score, _ = ks_2samp(prediction_dist, simulation_dist)
    wasserstein_dist = wasserstein_distance(prediction_dist, simulation_dist)
    ttest_p_value = ttest_ind(prediction_dist, simulation_dist).pvalue
    mannwhitneyu_p_value = mannwhitneyu(prediction_dist, simulation_dist).pvalue

    # Append the scores to the respective lists
    ks_scores.append(ks_score)
    wasserstein_distances.append(wasserstein_dist)
    ttest_p_values.append(ttest_p_value)
    mannwhitneyu_p_values.append(mannwhitneyu_p_value)

# Print the results
print("KS Scores:", ks_scores)
print("Wasserstein Distances:", wasserstein_distances)
print("T-test p-values:", ttest_p_values)
print("Mann-Whitney U p-values:", mannwhitneyu_p_values)

# Plot the distributions for the first parameter set as an example
plt.plot(prediction_dist, label='Prediction Distribution')
plt.plot(simulation_dist, label='Simulation Distribution')
plt.legend()
plt.show()

