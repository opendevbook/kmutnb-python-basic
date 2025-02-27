
# Python Pandas Tutorial for Data Scientists

Pandas is one of the most popular Python libraries for data manipulation and analysis. It provides data structures like `DataFrame` and `Series` to handle and process structured data efficiently.

## Installation

You can install Pandas using pip:

```bash
pip install pandas
```

---

## 1. Introduction to Pandas Data Structures

### 1.1 Series
A **Series** is a one-dimensional array-like object that can hold any data type.

```python
import pandas as pd

# Create a Series
data = pd.Series([10, 20, 30, 40])
print(data)
```

Output:

```
0    10
1    20
2    30
3    40
dtype: int64
```

### 1.2 DataFrame
A **DataFrame** is a two-dimensional labeled data structure with columns of potentially different types.

```python
# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)

print(df)
```

Output:

```
      Name  Age         City
0    Alice   25     New York
1      Bob   30  Los Angeles
2  Charlie   35      Chicago
```

---

## 2. Importing and Exporting Data

### 2.1 Reading CSV Files
You can load a CSV file into a DataFrame using `pd.read_csv()`.

```python
# Load a CSV file
df = pd.read_csv('data.csv')
print(df.head())
```

### 2.2 Writing Data to CSV
You can save a DataFrame to a CSV file using `to_csv()`.

```python
# Save DataFrame to CSV
df.to_csv('output.csv', index=False)
```

---

## 3. Basic DataFrame Operations

### 3.1 Viewing Data

- **head()**: Returns the first n rows of the DataFrame.
- **tail()**: Returns the last n rows of the DataFrame.
- **shape**: Returns the number of rows and columns.

```python
print(df.head())   # First 5 rows
print(df.tail())   # Last 5 rows
print(df.shape)    # (rows, columns)
```

### 3.2 Selecting Columns

You can select columns by using the column name.

```python
# Select a single column
print(df['Name'])

# Select multiple columns
print(df[['Name', 'City']])
```

### 3.3 Filtering Data

You can filter rows using conditions.

```python
# Filter rows where Age > 30
filtered_df = df[df['Age'] > 30]
print(filtered_df)
```

---

## 4. Data Cleaning and Transformation

### 4.1 Handling Missing Data

Pandas provides functions like `isna()`, `fillna()`, and `dropna()` to handle missing data.

```python
# Check for missing data
print(df.isna().sum())

# Fill missing values
df.fillna({'Age': 0, 'City': 'Unknown'}, inplace=True)

# Drop rows with missing data
df.dropna(inplace=True)
```

### 4.2 Renaming Columns

You can rename columns using the `rename()` function.

```python
# Rename columns
df.rename(columns={'Name': 'Full Name', 'Age': 'Age in Years'}, inplace=True)
print(df)
```

### 4.3 Changing Data Types

You can change the data type of columns using `astype()`.

```python
# Convert Age to string
df['Age in Years'] = df['Age in Years'].astype(str)
print(df.dtypes)
```

---

## 5. Data Aggregation and Grouping

### 5.1 Group By
You can use the `groupby()` function to group data based on columns and perform aggregation.

```python
# Group by 'City' and calculate the average age
grouped = df.groupby('City')['Age in Years'].mean()
print(grouped)
```

### 5.2 Aggregation Functions

Pandas allows you to apply multiple aggregation functions.

```python
# Apply multiple aggregation functions
agg_df = df.groupby('City').agg({'Age in Years': ['mean', 'sum']})
print(agg_df)
```

---

## 6. Merging and Joining DataFrames

### 6.1 Merging DataFrames
You can merge two DataFrames using the `merge()` function.

```python
# Merge two DataFrames on a common column
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'City': ['New York', 'Los Angeles', 'Chicago']})

merged_df = pd.merge(df1, df2, on='ID', how='inner')
print(merged_df)
```

Output:

```
   ID     Name         City
0   1    Alice     New York
1   2      Bob  Los Angeles
```

### 6.2 Joining DataFrames
You can also join DataFrames using the `join()` function.

```python
# Join DataFrames based on index
df1 = pd.DataFrame({'A': [1, 2, 3]})
df2 = pd.DataFrame({'B': [4, 5, 6]})
joined_df = df1.join(df2)
print(joined_df)
```

---

## 7. Data Visualization

Pandas integrates well with libraries like `Matplotlib` and `Seaborn` for data visualization.

```python
import matplotlib.pyplot as plt

# Create a simple bar plot
df['Age'].plot(kind='bar')
plt.show()
```

---

## 8. Working with Time Series Data

Pandas provides powerful functionality for time series data manipulation.

### 8.1 Converting to Datetime

You can convert strings to datetime objects using `pd.to_datetime()`.

```python
# Convert a column to datetime
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
```

### 8.2 Resampling Time Series Data

You can resample the data to different frequencies.

```python
# Resample data to monthly frequency and calculate the mean
df_resampled = df.resample('M').mean()
print(df_resampled)
```

---

## Conclusion

Pandas is a powerful tool for data manipulation and analysis. By mastering its functions like `DataFrame`, `Series`, and common operations, you'll be able to efficiently handle and analyze large datasets.

For more advanced topics, explore [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/).
