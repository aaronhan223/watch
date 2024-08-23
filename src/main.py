import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate a 2D dataset (1000 samples)
mean = [2, 2]
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix
X_train = np.random.multivariate_normal(mean, cov, 1000)
y_train = (X_train[:, 0] + X_train[:, 1] > 4).astype(int)  # Simple linear decision boundary

# Plot the original training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6)
plt.title("Original Training Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Simulate covariate shift by changing the mean of the test data
mean_shifted = [5, 5]  # Shifted mean
X_test_covariate_shift = np.random.multivariate_normal(mean_shifted, cov, 1000)
y_test_covariate_shift = (X_test_covariate_shift[:, 0] + X_test_covariate_shift[:, 1] > 4).astype(int)

# Plot the covariate shifted test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test_covariate_shift[:, 0], X_test_covariate_shift[:, 1], c=y_test_covariate_shift, cmap='coolwarm', alpha=0.6)
plt.title("Test Data with Covariate Shift")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Simulate label shift by changing the label distribution
# Keep the input distribution the same as the original
X_test_label_shift = np.random.multivariate_normal(mean, cov, 1000)

# Change the proportion of the labels (e.g., increase the number of class 1)
y_test_label_shift = np.random.choice([0, 1], size=1000, p=[0.2, 0.8])

# Plot the label shifted test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test_label_shift[:, 0], X_test_label_shift[:, 1], c=y_test_label_shift, cmap='coolwarm', alpha=0.6)
plt.title("Test Data with Label Shift")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()