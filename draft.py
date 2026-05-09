import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report
)

# Fixing the seed makes random results reproducible across runs.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
n_train = 500

# Create synthetic training features with realistic ranges.
train_df = pd.DataFrame({
    "study_hours": np.clip(np.random.normal(5, 2, n_train), 0, 12),
    "sleep_hours": np.clip(np.random.normal(7, 1.2, n_train), 3, 11),
    "attendance": np.clip(np.random.normal(75, 15, n_train), 0, 100),
    "previous_score": np.clip(np.random.normal(65, 12, n_train), 0, 100)
})

# Build a hidden scoring formula, then convert it to pass/fail labels.
train_score = (
    0.9 * train_df["study_hours"]
    + 0.06 * train_df["attendance"]
    + 0.08 * train_df["previous_score"]
    + 0.5 * train_df["sleep_hours"]
    + np.random.normal(0, 3, n_train)
)

# Thresholding turns a continuous score into binary classes (0/1).
train_df["passed"] = (train_score > 16).astype(int)

# Quick preview of generated data.
train_df.head()

# Check the shape of the dataset
# rows = number of observations/examples
# columns = number of features/variables
print("Rows and columns:", train_df.shape)

# Check the column names and data types
# This helps identify numeric, categorical, and text-based variables.
train_df.info()

# Count missing values in each column
missing_values = train_df.isnull().sum()

# Show only columns that actually have missing values
missing_values = missing_values[missing_values > 0]

print("Columns with missing values:")
print(missing_values)

# Get summary statistics for numeric columns
# This helps us understand the scale, spread, and possible outliers in the data.
train_df.describe()

# Replace 'target' with the actual name of the column we are trying to predict
target_column = "passed"

# Count how many examples belong to each class
train_df[target_column].value_counts()