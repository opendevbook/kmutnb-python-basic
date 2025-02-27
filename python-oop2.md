# Python 3.12 Object-Oriented Programming (OOP) in Data Science

## Introduction to OOP in Python 3.12 for Data Science
Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects. Python 3.12 continues to improve its OOP capabilities, providing a robust environment for designing modular, reusable, and maintainable code, which is particularly useful in data science.

### Key OOP Concepts in Data Science:
- **Class**: A blueprint for creating objects, useful for structuring data processing tasks.
- **Object**: An instance of a class representing datasets, models, or transformations.
- **Encapsulation**: Hiding internal details of an object and only exposing necessary functionalities for data preprocessing and model building.
- **Inheritance**: Creating new classes from existing ones to build reusable data science tools.
- **Polymorphism**: Using a unified interface for different data processing tasks.
- **Abstraction**: Hiding complex implementation details and exposing only essential features, useful for ML pipelines.

## Defining a Data Science Class
```python
import pandas as pd

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def clean_data(self):
        self.data.dropna(inplace=True)
        return self.data

# Usage
sample_data = pd.DataFrame({"A": [1, 2, None], "B": [3, None, 5]})
processor = DataProcessor(sample_data)
print(processor.clean_data())
```

## Encapsulation in Data Science
Encapsulation ensures that critical data transformations are protected from unintended modifications.
```python
class Model:
    def __init__(self, model_type: str):
        self.__model_type = model_type  # Private attribute
    
    def get_model_type(self):
        return self.__model_type

# Usage
ml_model = Model("RandomForest")
print(ml_model.get_model_type())
```

## Inheritance for Machine Learning Models
Inheritance enables extending base classes for different models.
```python
from sklearn.linear_model import LinearRegression

class CustomLinearModel(LinearRegression):
    def __init__(self):
        super().__init__()
    
    def train(self, X, y):
        self.fit(X, y)
        return self

# Usage
import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = CustomLinearModel()
model.train(X, y)
```

## Polymorphism in Data Processing
Polymorphism allows different data handlers to work through a common interface.
```python
class CSVLoader:
    def load(self, filepath):
        return pd.read_csv(filepath)

class ExcelLoader:
    def load(self, filepath):
        return pd.read_excel(filepath)

# Polymorphic function
def load_data(loader, filepath):
    return loader.load(filepath)
```

```Python
import pandas as pd
from typing import Protocol

# Define a protocol (interface) for loaders
class DataLoader(Protocol):
    def load(self, filepath: str) -> pd.DataFrame:
        ...

class CSVLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

class ExcelLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        return pd.read_excel(filepath)

# Polymorphic function with type hints
def load_data(loader: DataLoader, filepath: str) -> pd.DataFrame:
    try:
        return loader.load(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Usage
csv_loader = CSVLoader()
df = load_data(csv_loader, "data.csv")
print(df.head())

```

## Meaning of `...` in This Context

### 1. Placeholder for Method Implementation
- The ellipsis `...` means that the method is not implemented in the `DataLoader` protocol itself.
- Since `Protocol` is used to define an interface, the actual implementation must be provided by concrete classes (e.g., `CSVLoader`, `ExcelLoader`).

### 2. Equivalent to `pass`
- The `...` serves the same purpose as `pass`, meaning "this function should exist, but I’m not defining it here."
- Instead of `...`, you could also write:

```python
class DataLoader(Protocol):
    def load(self, filepath: str) -> pd.DataFrame:
        pass

```

### 3. Common Usage in Abstract Methods
It is often used in abstract base classes (ABC) or protocols to indicate that a method should be implemented by subclasses.

Example of implementation
```python
from typing import Protocol
import pandas as pd

class DataLoader(Protocol):
    def load(self, filepath: str) -> pd.DataFrame:
        ...

class CSVLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

# Using the loader
csv_loader = CSVLoader()
df = csv_loader.load("data.csv")
print(df.head())
```


## Abstraction in Data Science Pipelines
Abstraction helps simplify complex pipelines by exposing only essential methods.
```python
from abc import ABC, abstractmethod

class DataPipeline(ABC):
    @abstractmethod
    def process(self, data):
        pass

class NormalizationPipeline(DataPipeline):
    def process(self, data):
        return (data - data.mean()) / data.std()
```

## New Features in Python 3.12 Relevant to OOP in Data Science
Python 3.12 introduces enhancements useful for data science:
- **Performance improvements** for method calls in large datasets.
- **Better error messages** to debug data-related issues.
- **Enhanced `typing` module** for improved type hinting in data pipelines.

Example with type hints:
```python
from typing import Union

def scale_values(data: pd.Series) -> pd.Series:
    return (data - data.min()) / (data.max() - data.min())
```

## Conclusion
OOP in Python 3.12 enhances data science workflows by promoting modular, reusable, and maintainable code. Using OOP principles, you can build robust data processing pipelines, encapsulate transformations, and develop scalable machine learning models.

