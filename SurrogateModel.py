import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def complex_2d_function(x):
    """
    A more complex 2D function to approximate.
    Inputs are a 2-column array, x = [x1, x2].
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 20 * np.exp(-((x1 - 2)**2 + (x2 - 3)**2) / 2) + \
           15 * np.exp(-((x1 + 4)**2 + (x2 + 1)**2) / 3) + \
           10 * np.exp(-((x1 - 5)**2 + (x2 + 4)**2) / 4)
		   
# Generate a grid of points for plotting the true function
xx, yy = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
true_points = np.c_[xx.ravel(), yy.ravel()]
true_values = complex_2d_function(true_points)

# Generate a small number of random sample points (expensive runs)
np.random.seed(0)
num_samples = 100
sample_points = np.random.uniform(-8, 8, size=(num_samples, 2))
sample_values = complex_2d_function(sample_points)		   

# Define the kernel for the Gaussian Process
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# Initialize and train the GPR model
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_model.fit(sample_points, sample_values)

# Predict on the full grid using the surrogate model
surrogate_values, surrogate_std = gp_model.predict(true_points, return_std=True)

# Reshape the results for plotting
true_values_plot = true_values.reshape(xx.shape)
surrogate_values_plot = surrogate_values.reshape(xx.shape)
surrogate_std_plot = surrogate_std.reshape(xx.shape)

# Create subplots for visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: The Original "Complex" Function
c1 = axes[0].contourf(xx, yy, true_values_plot, levels=50)
axes[0].set_title('Original "Complex" Function')
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
fig.colorbar(c1, ax=axes[0])

# Plot 2: The Surrogate Model Prediction
c2 = axes[1].contourf(xx, yy, surrogate_values_plot, levels=50)
axes[1].scatter(sample_points[:, 0], sample_points[:, 1], color='red', marker='o', s=10, label='Sampled Points')
axes[1].set_title('Surrogate Model Prediction')
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].legend()
fig.colorbar(c2, ax=axes[1])

# Plot 3: The Uncertainty of the Prediction
c3 = axes[2].contourf(xx, yy, surrogate_std_plot, levels=50)
axes[2].set_title('Prediction Uncertainty (Std Dev)')
axes[2].set_xlabel('$x_1$')
axes[2].set_ylabel('$x_2$')
fig.colorbar(c3, ax=axes[2])

plt.tight_layout()
plt.show()