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
stride = 2

# Calculate the output dimensions
output_dim = (X_dim - F_dim) // stride + 1

# Initialize the output matrix
output = np.zeros((output_dim, output_dim))

# Perform the convolution with a stride of 2
for i in range(0, output_dim):
    for j in range(0, output_dim):
        # Calculate the top-left corner of the current region in X
        x_start, y_start = i * stride, j * stride
        # Extract the region of X that the filter will convolve with
        region = X[x_start:x_start+F_dim, y_start:y_start+F_dim]
        # Perform element-wise multiplication and sum the result
        print(region * F)
        output[i, j] = np.sum(region * F)

# Print the result
print("Output matrix after convolution with stride 2:")
print(output)
