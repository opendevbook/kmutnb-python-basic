
# NumPy for Data Science: Intermediate Tutorial

NumPy is an essential library for scientific computing in Python. It provides support for large multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays.

In this tutorial, we'll explore intermediate features of NumPy for data science, including:

1. **Advanced Indexing**
2. **Linear Algebra Operations**
3. **Broadcasting**
4. **Statistical Functions**

---

### **1. Advanced Indexing in NumPy**

Advanced indexing allows you to select and manipulate specific elements or groups of elements from arrays in powerful ways.

#### **Boolean Indexing**

You can use boolean arrays to select elements that satisfy a specific condition.

```python
import numpy as np

# Create an array of random integers
data = np.random.randint(1, 10, size=(5, 5))
print("Data Array:
", data)

# Select values greater than 5
filtered_data = data[data > 5]
print("
Filtered Data (values > 5):
", filtered_data)
```

#### **Fancy Indexing**

You can index arrays with lists or arrays of integers, allowing you to select specific rows or columns.

```python
# Select specific rows and columns using fancy indexing
rows = np.array([0, 3])
cols = np.array([1, 4])

subset = data[rows, cols]
print("
Subset of Data (rows 0,3 and columns 1,4):
", subset)
```

---

### **2. Linear Algebra with NumPy**

NumPy provides efficient functions for linear algebra operations like matrix multiplication, eigenvalues, etc.

#### **Matrix Multiplication**

Use `np.dot()` or `@` operator for matrix multiplication.

```python
# Define two random matrices
A = np.random.rand(3, 2)
B = np.random.rand(2, 3)

# Matrix multiplication
result = np.dot(A, B)
print("
Matrix Multiplication (A * B):
", result)
```

Alternatively, you can use the `@` operator:

```python
result = A @ B
```

#### **Eigenvalues and Eigenvectors**

You can calculate eigenvalues and eigenvectors using `np.linalg.eig()`.

```python
# Create a square matrix
matrix = np.random.rand(3, 3)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("
Eigenvalues:", eigenvalues)
print("Eigenvectors:
", eigenvectors)
```

---

### **3. Broadcasting in NumPy**

Broadcasting allows NumPy to perform operations on arrays of different shapes, making it very efficient.

#### **Example of Broadcasting**

When performing element-wise operations on arrays of different shapes, NumPy automatically adjusts the smaller array to match the larger one.

```python
# Create a 3x3 matrix and a 1x3 array
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])

# Broadcasting the vector across the matrix
result = matrix + vector
print("
Broadcasted Matrix + Vector:
", result)
```

Here, the vector `[10, 20, 30]` is added to each row of the matrix.

---

### **4. Statistical Functions in NumPy**

NumPy provides a variety of functions to perform statistical analysis.

#### **Mean, Median, and Standard Deviation**

You can calculate these statistical measures easily with NumPy.

```python
# Create an array of random numbers
data = np.random.randn(1000)

# Calculate the mean, median, and standard deviation
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)

print("
Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
```

#### **Correlation and Covariance**

Use `np.corrcoef()` and `np.cov()` to calculate correlation and covariance matrices.

```python
# Create two random data arrays
x = np.random.rand(100)
y = np.random.rand(100)

# Calculate correlation coefficient
corr = np.corrcoef(x, y)
print("
Correlation Coefficient:
", corr)

# Calculate covariance matrix
cov = np.cov(x, y)
print("
Covariance Matrix:
", cov)
```

---

### **Summary**

In this intermediate-level tutorial, we covered the following essential NumPy features for data scientists:

- **Advanced Indexing**: Fancy and boolean indexing for selecting data.
- **Linear Algebra**: Matrix multiplication, eigenvalues, and eigenvectors.
- **Broadcasting**: Efficient operations on arrays of different shapes.
- **Statistical Functions**: Mean, median, standard deviation, correlation, and covariance.

These features will help you efficiently process and analyze large datasets in your data science projects.

---

### **Next Steps**

- Experiment with real-world datasets to apply these concepts.
- Explore NumPy's `random` module to generate datasets for analysis.
- Dive into more advanced linear algebra operations, such as solving systems of linear equations with `np.linalg.solve()`.

If you have any questions or would like to see more examples, feel free to ask!
