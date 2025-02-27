# Python Pandas Tutorial

## Introduction to Pandas
Pandas is a powerful open-source library for data analysis and manipulation. It provides data structures like `DataFrame` and `Series`, which are essential for handling structured data.

### Installing Pandas
```bash
pip install pandas
```

## Creating a DataFrame
A DataFrame is a two-dimensional, tabular data structure with labeled axes (rows and columns).

```python
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)
print(df)
```

### Output:
```
    Name  Age         City
0  Alice   25     New York
1    Bob   30  Los Angeles
2 Charlie   35      Chicago
```

## Reading and Writing Data

### Reading from CSV
data.csv  
```
Name,Age,City,Salary
Alice,25,New York,50000
Bob,30,Los Angeles,60000
Charlie,35,Chicago,70000
David,40,Houston,80000
Emma,28,Miami,55000

```

```python
df = pd.read_csv("data.csv")
print(df.head())  # Display the first 5 rows
```

### Writing to CSV
```python
df.to_csv("output.csv", index=False)
```

## Selecting and Filtering Data

### Selecting a Column
```python
print(df["Name"])
```

### Selecting Multiple Columns
```python
print(df[["Name", "Age"]])
```

### Filtering Rows
```python
print(df[df["Age"] > 25])
```

## Data Manipulation

### Adding a New Column
```python
df["Salary"] = [50000, 60000, 70000]
print(df)
```

### Updating Column Values
```python
df["Age"] = df["Age"] + 1
print(df)
```

### Dropping a Column
```python
df = df.drop(columns=["City"])
print(df)
```

## Grouping and Aggregation
```python
print(df.groupby("City").mean())
```

## Merging DataFrames
```python
df1 = pd.DataFrame({"ID": [1, 2, 3], "Name": ["Alice", "Bob", "Charlie"]})
df2 = pd.DataFrame({"ID": [1, 2, 3], "Salary": [50000, 60000, 70000]})
merged_df = pd.merge(df1, df2, on="ID")
print(merged_df)
```

## Conclusion
Pandas is a versatile and powerful library for handling and analyzing structured data. By mastering its features, you can efficiently manage data for real-world applications.

# Statistical Analysis with Pandas and Matplotlib

This example demonstrates how to use **Pandas** and **Matplotlib** to perform statistical analysis on a dataset.  
We will generate **random sales data**, calculate **basic statistics**, and visualize the data using **histograms** and **box plots**.

## 📌 Steps:
1. Generate a dataset with random sales values.
2. Compute **mean, median, and standard deviation**.
3. Plot **histogram and boxplot** to visualize the distribution.

---

## 📜 Python Code:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (e.g., monthly sales)
np.random.seed(42)
data = {
    "Month": pd.date_range(start="2024-01-01", periods=12, freq='M'),
    "Sales": np.random.randint(5000, 20000, size=12)  # Random sales values
}

# Create DataFrame
df = pd.DataFrame(data)

# Compute basic statistics
mean_sales = df["Sales"].mean()
median_sales = df["Sales"].median()
std_sales = df["Sales"].std()

# Print statistics
print("Mean Sales:", mean_sales)
print("Median Sales:", median_sales)
print("Standard Deviation:", std_sales)

# Plot histogram and boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(df["Sales"], bins=6, color="skyblue", edgecolor="black")
axes[0].axvline(mean_sales, color="red", linestyle="dashed", label=f"Mean: {mean_sales:.2f}")
axes[0].set_title("Sales Distribution (Histogram)")
axes[0].set_xlabel("Sales Amount")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Boxplot
axes[1].boxplot(df["Sales"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
axes[1].set_title("Sales Distribution (Boxplot)")
axes[1].set_xlabel("Sales Amount")

plt.tight_layout()
plt.show()
```