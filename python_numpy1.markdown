
# Python NumPy Tutorial

## Basic Concepts

1. **Creating Arrays**:
   - 1D Array: 
   ```python
   arr = np.array([1, 2, 3, 4, 5])
   ```
   Output: `1D Array: [1 2 3 4 5]`

   - 2D Array:
   ```python
   arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
   ```
   Output: `2D Array (Matrix):
[[1 2 3]
 [4 5 6]]`

2. **Array Properties**:
   - Shape of arr_2d:
   ```python
   arr_2d.shape
   ```
   Output: `Shape of 2D Array: (2, 3)`

   - Dimensions of arr_2d:
   ```python
   arr_2d.ndim
   ```
   Output: `Dimensions of 2D Array: 2`

3. **Creating Arrays with Specific Values**:
   - Zeros Array:
   ```python
   zeros_array = np.zeros((2, 3))
   ```
   Output: `Zeros Array:
[[0. 0. 0.]
 [0. 0. 0.]]`

   - Ones Array:
   ```python
   ones_array = np.ones((3, 2))
   ```
   Output: `Ones Array:
[[1. 1.]
 [1. 1.]
 [1. 1.]]`

   - Identity Matrix:
   ```python
   identity_matrix = np.eye(2)
   ```
   Output: `Identity Matrix:
[[1. 0.]
 [0. 1.]]`

## Intermediate Concepts

1. **Array Operations**:
   - Element-wise addition:
   ```python
   add_result = arr1 + arr2
   ```
   Output: `Element-wise Addition: [5 7 9]`

   - Element-wise multiplication:
   ```python
   mul_result = arr1 * arr2
   ```
   Output: `Element-wise Multiplication: [ 4 10 18]`

2. **Reshaping Arrays**:
   ```python
   reshaped_arr = arr_2d.reshape(3, 2)
   ```
   Output: `Reshaped Array (3x2):
[[1 2]
 [3 4]
 [5 6]]`

3. **Flattening Arrays**:
   ```python
   flattened_arr = arr_2d.flatten()
   ```
   Output: `Flattened Array: [1 2 3 4 5 6]`

## Advanced Concepts

1. **Stacking Arrays**:
   - Vertical stack:
   ```python
   v_stack = np.vstack((arr1, arr2))
   ```
   Output: `Vertical Stack:
[[1 2 3]
 [4 5 6]]`

   - Horizontal stack:
   ```python
   h_stack = np.hstack((arr1, arr2))
   ```
   Output: `Horizontal Stack: [1 2 3 4 5 6]`

2. **Mathematical Functions**:
   - Mean, Median, and Standard Deviation:
   ```python
   mean_value = np.mean(arr_2d)
   median_value = np.median(arr_2d)
   std_dev = np.std(arr_2d)
   ```
   Output:
   ```
   Mean of arr_2d: 3.5
   Median of arr_2d: 3.5
   Standard Deviation of arr_2d: 1.707825127659933
   ```

