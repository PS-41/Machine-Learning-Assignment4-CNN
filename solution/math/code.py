import numpy as np

# Define the input matrix X and the filter F
X = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 4, 1, 0, 0],
    [0, 3, 1, 1, 0, 1, 0],
    [0, 2, 4, 1, 0, 1, 0],
    [0, 2, 0, 5, 2, 2, 0],
    [0, 0, 1, 3, 2, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

F = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Get the dimensions of X and F
X_dim = X.shape[0]
F_dim = F.shape[0]
output_dim = X_dim - F_dim + 1  # Since stride is 1 and no padding

# Initialize the output matrix
output = np.zeros((output_dim, output_dim))

# Perform the convolution
for i in range(output_dim):
    for j in range(output_dim):
        # Extract the region of X that the filter will convolve with
        region = X[i:i+F_dim, j:j+F_dim]
        # Perform element-wise multiplication and sum the result
        print(region * F)
        output[i, j] = np.sum(region * F)

# Print the result
print("Output matrix after convolution:")
print(output)
