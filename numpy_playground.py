# Lets explore this 21 chapters together:)
import numpy as np

# Section 1: Array Creation and Basic Operations
# -----------------------------------------------

# Create a simple 1D array
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1)

# Create a 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array (Matrix):\n", arr2)

# Create arrays of zeros, ones, and a constant value
zeros_array = np.zeros((3, 3))
ones_array = np.ones((2, 4))
constant_array = np.full((3, 3), 7)
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)
print("Constant Array:\n", constant_array)

# Create an array with a range of values
range_array = np.arange(10, 20)
print("Range Array:", range_array)

# Create an array with equally spaced values
linspace_array = np.linspace(0, 1, 5)
print("Linspace Array:", linspace_array)

# Create an identity matrix
identity_matrix = np.eye(3)
print("Identity Matrix:\n", identity_matrix)

# Section 2: Array Indexing and Slicing
# -------------------------------------

# Accessing elements in a 1D array
element1 = arr1[0]  # First element
element2 = arr1[-1]  # Last element
print("First element:", element1)
print("Last element:", element2)

# Accessing elements in a 2D array
element3 = arr2[0, 1]  # Element in first row, second column
element4 = arr2[1, -1]  # Element in last row, last column
print("Element at (0, 1):", element3)
print("Element at (1, -1):", element4)

# Slicing 1D arrays
slice1 = arr1[1:4]  # Elements from index 1 to 3
print("Sliced 1D Array:", slice1)

# Slicing 2D arrays
slice2 = arr2[:, 1:3]  # All rows, columns 1 to 2
print("Sliced 2D Array:\n", slice2)

# Section 3: Array Reshaping and Flattening
# -----------------------------------------

# Reshape a 1D array into a 2D array
reshaped_array = np.reshape(arr1, (5, 1))
print("Reshaped Array (1D to 2D):\n", reshaped_array)

# Flatten a 2D array into a 1D array
flattened_array = arr2.flatten()
print("Flattened Array:", flattened_array)

# Transpose of a matrix
transposed_array = np.transpose(arr2)
print("Transposed Array:\n", transposed_array)

# Section 4: Array Concatenation and Splitting
# --------------------------------------------

# Concatenate two 1D arrays
arr3 = np.array([6, 7, 8])
concatenated_array = np.concatenate((arr1, arr3))
print("Concatenated 1D Array:", concatenated_array)

# Concatenate two 2D arrays along rows
arr4 = np.array([[7, 8, 9], [10, 11, 12]])
concatenated_array2d = np.concatenate((arr2, arr4), axis=0)
print("Concatenated 2D Array along rows:\n", concatenated_array2d)

# Concatenate two 2D arrays along columns
concatenated_array2d_col = np.concatenate((arr2, arr4), axis=1)
print("Concatenated 2D Array along columns:\n", concatenated_array2d_col)

# Split an array into multiple sub-arrays
split_array = np.split(concatenated_array, 3)
print("Split Array:", split_array)

# Section 5: Mathematical Operations
# ----------------------------------

# Element-wise addition, subtraction, multiplication, and division
arr5 = np.array([10, 20, 30, 40, 50])
sum_array = arr1 + arr5
difference_array = arr1 - arr5
product_array = arr1 * arr5
quotient_array = arr5 / arr1
print("Element-wise Sum:", sum_array)
print("Element-wise Difference:", difference_array)
print("Element-wise Product:", product_array)
print("Element-wise Quotient:", quotient_array)

# Element-wise square root and exponential
sqrt_array = np.sqrt(arr1)
exp_array = np.exp(arr1)
print("Square Root Array:", sqrt_array)
print("Exponential Array:", exp_array)

# Trigonometric functions
sin_array = np.sin(np.pi * arr1 / 4)
cos_array = np.cos(np.pi * arr1 / 4)
print("Sine Array:", sin_array)
print("Cosine Array:", cos_array)

# Sum, mean, median, standard deviation, and variance
sum_value = np.sum(arr1)
mean_value = np.mean(arr1)
median_value = np.median(arr1)
std_dev = np.std(arr1)
variance = np.var(arr1)
print("Sum:", sum_value)
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# Section 6: Broadcasting
# -----------------------

# Broadcasting: Adding a scalar to an array
broadcasted_sum = arr1 + 5
print("Broadcasted Sum:", broadcasted_sum)

# Broadcasting: Adding a 1D array to a 2D array
arr6 = np.array([[1], [2], [3]])
broadcasted_2d_sum = arr2 + arr6
print("Broadcasted 2D Sum:\n", broadcasted_2d_sum)

# Section 7: Logical Operations and Filtering
# -------------------------------------------

# Element-wise comparison
comparison_result = arr1 > 3
print("Comparison Result:", comparison_result)

# Filtering elements based on a condition
filtered_array = arr1[arr1 > 3]
print("Filtered Array:", filtered_array)

# Logical operations: AND, OR, NOT
and_result = np.logical_and(arr1 > 2, arr1 < 5)
or_result = np.logical_or(arr1 < 2, arr1 > 4)
not_result = np.logical_not(arr1 > 3)
print("Logical AND Result:", and_result)
print("Logical OR Result:", or_result)
print("Logical NOT Result:", not_result)

# Section 8: Linear Algebra Operations
# ------------------------------------

# Matrix multiplication
matrix_product = np.dot(arr2, arr4.T)  # T denotes transpose
print("Matrix Product:\n", matrix_product)

# Determinant of a matrix
matrix_determinant = np.linalg.det(identity_matrix)
print("Matrix Determinant:", matrix_determinant)

# Inverse of a matrix
matrix_inverse = np.linalg.inv(identity_matrix)
print("Matrix Inverse:\n", matrix_inverse)

# Eigenvalues and Eigenvectors
eig_values, eig_vectors = np.linalg.eig(identity_matrix)
print("Eigenvalues:", eig_values)
print("Eigenvectors:\n", eig_vectors)

# Section 9: Random Number Generation
# -----------------------------------

# Generate random numbers from a uniform distribution
random_array = np.random.rand(3, 3)
print("Random Array (Uniform Distribution):\n", random_array)

# Generate random integers within a range
random_int_array = np.random.randint(0, 10, size=(3, 3))
print("Random Integer Array:\n", random_int_array)

# Generate random numbers from a normal distribution
normal_array = np.random.randn(3, 3)
print("Random Array (Normal Distribution):\n", normal_array)

# Section 10: Saving and Loading Arrays
# -------------------------------------

# Save an array to a file
np.save('saved_array.npy', arr1)
print("Array saved to 'saved_array.npy'")

# Load an array from a file
loaded_array = np.load('saved_array.npy')
print("Loaded Array:", loaded_array)

# Save multiple arrays to a single file
np.savez('saved_arrays.npz', arr1=arr1, arr2=arr2)
print("Arrays saved to 'saved_arrays.npz'")

# Load multiple arrays from a single file
loaded_arrays = np.load('saved_arrays.npz')
print("Loaded Arrays:", dict(loaded_arrays))

# Section 11: Advanced Indexing
# -----------------------------

# Boolean indexing
boolean_index = arr1 > 2
print("Boolean Indexing Result:", arr1[boolean_index])

# Fancy indexing: Using an array of indices to access elements
fancy_index = [0, 2, 4]
print("Fancy Indexing Result:", arr1[fancy_index])

# Section 12: Array Broadcasting and Vectorization
# ------------------------------------------------

# Broadcasting a scalar to an array
scalar_broadcast = 10 + arr1
print("Scalar Broadcast Result:", scalar_broadcast)

# Vectorized operations: Operations on the entire array without loops
vectorized_sum = np.sum(arr1 ** 2)
print("Vectorized Sum:", vectorized_sum)

# Section 13: Handling Missing Data
# ---------------------------------

# Introduce NaN values into an array
arr_with_nan = np.array([1, 2, np.nan, 4, 5])
print("Array with NaN:", arr_with_nan)

# Check for NaN values
nan_check = np.isnan(arr_with_nan)
print("NaN Check:", nan_check)

# Fill NaN values with a specific value
filled_array = np.nan_to_num(arr_with_nan, nan=0)
print("Filled Array:", filled_array)

# Section 14: Advanced Linear Algebra
# -----------------------------------

# Singular Value Decomposition (SVD)
U, S, V = np.linalg.svd(arr2)
print("SVD - U Matrix:\n", U)
print("SVD - S Values:\n", S)
print("SVD - V Matrix:\n", V)

# Solving a system of linear equations
coefficients = np.array([[3, 1], [1, 2]])
constants = np.array([9, 8])
solutions = np.linalg.solve(coefficients, constants)
print("Solutions to Linear Equations:", solutions)

# Section 15: Array Manipulations
# -------------------------------

# Sorting an array
sorted_array = np.sort(arr1)
print("Sorted Array:", sorted_array)

# Sorting along a specific axis
sorted_2d_array = np.sort(arr2, axis=1)
print("Sorted 2D Array along rows:\n", sorted_2d_array)

# Section 16: Performance Optimization with NumPy
# -----------------------------------------------

# Time comparison: Vectorized operation vs. loop-based operation
import time

# Create a large array
large_array = np.random.rand(1000000)

# Vectorized operation
start_time = time.time()
vectorized_result = np.sum(large_array ** 2)
vectorized_time = time.time() - start_time

# Loop-based operation
start_time = time.time()
loop_result = sum([x ** 2 for x in large_array])
loop_time = time.time() - start_time

print("Vectorized Time:", vectorized_time)
print("Loop Time:", loop_time)
print("Speedup:", loop_time / vectorized_time)

# Section 17: Miscellaneous Functions
# -----------------------------------

# Find unique elements in an array
unique_elements = np.unique(arr1)
print("Unique Elements:", unique_elements)

# Count occurrences of elements
element_counts = np.bincount(arr1)
print("Element Counts:", element_counts)

# Cumulative sum and product
cumsum_array = np.cumsum(arr1)
cumprod_array = np.cumprod(arr1)
print("Cumulative Sum Array:", cumsum_array)
print("Cumulative Product Array:", cumprod_array)

# Section 18: Array Statistics
# ----------------------------

# Minimum, maximum, and range of values in an array
min_value = np.min(arr1)
max_value = np.max(arr1)
range_value = np.ptp(arr1)  # Range: max - min
print("Min Value:", min_value)
print("Max Value:", max_value)
print("Range Value:", range_value)

# Percentiles and quantiles
percentile_25 = np.percentile(arr1, 25)
median = np.median(arr1)
percentile_75 = np.percentile(arr1, 75)
quantiles = np.quantile(arr1, [0.25, 0.5, 0.75])
print("25th Percentile:", percentile_25)
print("Median:", median)
print("75th Percentile:", percentile_75)
print("Quantiles:", quantiles)

# Section 19: Creating Arrays from Functions
# ------------------------------------------

# Create an array using a function
def my_function(x, y):
    return x + y

function_array = np.fromfunction(my_function, (3, 3))
print("Function-based Array:\n", function_array)

# Create an array using a lambda function
lambda_array = np.fromfunction(lambda i, j: i + j, (3, 3))
print("Lambda-based Array:\n", lambda_array)

# Section 20: Array Broadcasting in Practice
# ------------------------------------------

# Broadcasting a 1D array to a 2D array
broadcasted_matrix = np.add(arr2, arr3.reshape(1, 3))
print("Broadcasted 1D Array to 2D Array:\n", broadcasted_matrix)

# Broadcasting a scalar to a 3D array
arr3d = np.random.rand(2, 3, 4)
broadcasted_3d_array = arr3d + 5
print("Broadcasted Scalar to 3D Array:\n", broadcasted_3d_array)

# Section 21: Final Remarks and Cleanup :)
# -------------------------------------

# Demonstrating the use of assertions for validation
assert np.array_equal(vectorized_sum, loop_result), "Results do not match!"

# Clear variables from memory (optional)
del arr1, arr2, arr3, arr4, arr5, arr6
del concatenated_array, identity_matrix, zeros_array

print("Wow complete!")
