# Python in Data Science: A Comprehensive Tutorial

## 1. Introduction to Python for Data Science

Python has become the de facto language for data science due to its simplicity, readability, and powerful ecosystem of libraries. This tutorial will guide you through essential Python tools and techniques for data science.

## 2. Setting Up Your Environment

### The Scientific Python Stack

For data science work, you'll need these core libraries:
- **NumPy**: For numerical computing and array operations
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For data visualization
- **Scikit-learn**: For machine learning
- **SciPy**: For scientific computing
- **Jupyter**: For interactive computing environments

### Installation with Anaconda

The easiest way to get started is with Anaconda, which comes with all these packages pre-installed:

```bash
# Download Anaconda from https://www.anaconda.com/products/individual

# Create a new environment
conda create -n datasci python=3.8

# Activate the environment
conda activate datasci

# Install additional packages as needed
conda install package_name
```

## 3. Data Handling with NumPy

NumPy provides the foundation for numerical computing in Python.

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))  # 3x4 array of zeros
arr3 = np.random.rand(2, 3)  # 2x3 array of random values

# Array operations
print(arr1 * 2)  # Element-wise multiplication
print(arr1.mean())  # Calculate mean
print(np.sqrt(arr1))  # Element-wise square root

# Array indexing and slicing
print(arr3[0, 1])  # Element at row 0, column 1
print(arr3[:, 0])  # First column
```

## 4. Data Analysis with Pandas

Pandas provides high-level data structures and functions for data manipulation.

### Working with DataFrames

```python
import pandas as pd

# Creating a DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 34, 29, 42],
    'City': ['New York', 'Paris', 'Berlin', 'London'],
    'Salary': [65000, 70000, 62000, 85000]
}
df = pd.DataFrame(data)

# Basic operations
print(df.head())  # First 5 rows
print(df.info())  # Information about the DataFrame
print(df.describe())  # Statistical summary

# Accessing data
print(df['Name'])  # Access a column
print(df[['Name', 'Age']])  # Multiple columns
print(df.loc[0])  # First row
print(df.iloc[1:3])  # Rows 1 to 2
```

### Data Loading and Preprocessing

```python
# Reading data from various sources
df_csv = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx')
df_sql = pd.read_sql('SELECT * FROM table', connection)

# Data cleaning
df.dropna()  # Remove rows with missing values
df.fillna(0)  # Fill missing values with 0
df.drop_duplicates()  # Remove duplicate rows

# Feature engineering
df['new_column'] = df['Age'] * 2
df['category'] = pd.cut(df['Salary'], bins=[0, 60000, 75000, 100000], 
                        labels=['Low', 'Medium', 'High'])
```

### Data Aggregation and Grouping

```python
# Grouping data
grouped = df.groupby('City')
print(grouped['Salary'].mean())  # Average salary by city

# Aggregation
result = df.groupby('City').agg({
    'Salary': ['mean', 'min', 'max', 'count'],
    'Age': ['mean', 'min', 'max']
})

# Pivot tables
pivot = df.pivot_table(values='Salary', index='City', 
                       columns='category', aggfunc='mean')
```

## 5. Data Visualization

### Matplotlib for Basic Plotting

```python
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(range(10), np.random.randn(10).cumsum(), label='Line 1')
plt.plot(range(10), np.random.randn(10).cumsum(), label='Line 2')
plt.title('Random Walk')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()

# Bar plot
cities = df['City'].value_counts()
plt.figure(figsize=(8, 5))
cities.plot(kind='bar')
plt.title('Count of Employees by City')
plt.ylabel('Count')
plt.show()
```

### Seaborn for Statistical Visualization

```python
import seaborn as sns

# Set the theme
sns.set_theme(style="whitegrid")

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()

# Pair plot
sns.pairplot(df[['Age', 'Salary']])
plt.show()

# Heatmap
correlation = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Interactive Visualization with Plotly

```python
import plotly.express as px

# Interactive scatter plot
fig = px.scatter(df, x='Age', y='Salary', color='City', 
                 size='Salary', hover_name='Name',
                 title='Salary vs. Age by City')
fig.show()

# Interactive bar chart
fig = px.bar(df, x='City', y='Salary', color='City',
             title='Total Salary by City')
fig.show()
```

## 6. Machine Learning with Scikit-learn

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Prepare dataset
X = df[['Age', 'City']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['Age']
categorical_features = ['City']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
```

### Training Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Random Forest
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R²: {r2_score(y_test, y_pred_rf)}")
```

### Model Evaluation and Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30]
}

# Grid search
grid_search = GridSearchCV(
    rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred_best)}")
print(f"R²: {r2_score(y_test, y_pred_best)}")
```

## 7. Working with Real-world Datasets

### The Kaggle API

```python
# Install kaggle API
# pip install kaggle

# Download a dataset (requires configuration of API key)
# kaggle datasets download -d username/dataset-name

# Load and explore the dataset
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.info())
```

### Working with Missing Values

```python
# Check for missing values
print(data.isnull().sum())

# Visualize missing values
import missingno as msno
msno.matrix(data)
plt.show()

# Imputation strategies
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data[['feature1', 'feature2']]),
    columns=['feature1', 'feature2'])

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
data_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(data[['feature1', 'feature2']]),
    columns=['feature1', 'feature2'])
```

### Feature Engineering

```python
# Date features
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek

# Text features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of words
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(data['text'])

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data['text'])
```

## 8. Advanced Data Science Topics

### Time Series Analysis with Prophet

```python
from prophet import Prophet

# Prepare data
df_ts = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365),
    'y': np.random.randn(365).cumsum() + 100
})

# Fit model
model = Prophet()
model.fit(df_ts)

# Make future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
plt.show()

# Components
fig = model.plot_components(forecast)
plt.show()
```

### Natural Language Processing with NLTK and spaCy

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Tokenization
text = "Python is a great language for data science and machine learning."
tokens = word_tokenize(text)
print(tokens)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print(stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(lemmatized)

# spaCy for more advanced NLP
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.pos_, token.dep_)

# Named entity recognition
for ent in doc.ents:
    print(ent.text, ent.label_)
```

### Deep Learning with TensorFlow and Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, batch_size=128, epochs=5, 
                    validation_split=0.2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## 9. Building Data Science Pipelines

### Creating End-to-End Workflows

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Evaluate with cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### MLflow for Experiment Tracking

```python
import mlflow
import mlflow.sklearn

# Start an MLflow run
with mlflow.start_run():
    # Set parameters
    n_estimators = 100
    max_depth = 10
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                  max_depth=max_depth,
                                  random_state=42)
    model.fit(X_train, y_train)
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
```

## 10. Best Practices for Data Science Projects

### Project Structure

A well-organized data science project might look like this:

```
project/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code for modules
│   ├── data/           # Data processing code
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training code
│   └── visualization/  # Visualization code
├── tests/              # Unit tests
├── models/             # Saved models
├── reports/            # Reports and presentations
├── README.md           # Project documentation
├── requirements.txt    # Dependencies
└── setup.py            # Make the project pip-installable
```

### Version Control for Data Science

```bash
# Initialize a git repository
git init

# Create a .gitignore file for data science
# Include entries for large data files, credentials, etc.

# Add and commit files
git add .
git commit -m "Initial commit"

# Use Git LFS for large files
git lfs install
git lfs track "*.csv" "*.pkl" "*.h5"
```

### Documentation

Good documentation is crucial for data science projects:

```python
def preprocess_features(df):
    """
    Preprocess features for model training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with raw features
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with engineered features
    
    Notes:
    ------
    This function performs the following steps:
    1. Handles missing values
    2. Encodes categorical variables
    3. Scales numerical features
    """
    # Implementation
    return processed_df
```

## 11. Resources for Further Learning

1. **Books**:
   - "Python for Data Analysis" by Wes McKinney
   - "Hands-On Machine Learning with Scikit-Learn & TensorFlow" by Aurélien Géron
   - "Deep Learning with Python" by François Chollet

2. **Online Courses**:
   - DataCamp's Python for Data Science track
   - Coursera's Data Science Specialization
   - Fast.ai's Practical Deep Learning for Coders

3. **Websites**:
   - Kaggle (datasets and competitions)
   - Towards Data Science (articles and tutorials)
   - scikit-learn, TensorFlow, and PyTorch documentation

4. **Communities**:
   - Stack Overflow
   - Reddit's r/datascience and r/MachineLearning
   - PyData conferences and meetups
