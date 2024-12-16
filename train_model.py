# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB

# File Path
file_path = 'gbsg.csv'

# Load the Dataset
df = pd.read_csv(file_path)

# Check for Missing Values
print("Missing values:\n", df.isnull().sum())
print("Nan:\n", df.isna().sum())

# Check for Duplicates
duplicates = df.duplicated()
duplicate_rows = df[duplicates]
num_duplicates = duplicates.sum()
print("Duplicate Rows:\n", duplicate_rows)
print(f"Number of duplicate rows: {num_duplicates}")

# Splitting the Data
X = df.drop(columns=['status', 'pid', 'Unnamed: 0'])  # Drop target and identifier columns
y = df['status']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing High-Variance Columns
high_variance_columns = ['age', 'size', 'nodes', 'pgr', 'er', 'rfstime']
scaler = StandardScaler()
X_train[high_variance_columns] = scaler.fit_transform(X_train[high_variance_columns])
X_test[high_variance_columns] = scaler.transform(X_test[high_variance_columns])

# Ensure 'meno' and 'hormon' are Categorical
X_train['meno'] = X_train['meno'].astype('category')
X_train['hormon'] = X_train['hormon'].astype('category')
X_test['meno'] = X_test['meno'].astype('category')
X_test['hormon'] = X_test['hormon'].astype('category')

# Creating a New Feature
X_train['aggressive'] = np.where(
    (X_train['size'] > X_train['size'].median()) & (X_train['grade'] > X_train['grade'].median()), 1, 0
)
X_test['aggressive'] = np.where(
    (X_test['size'] > X_test['size'].median()) & (X_test['grade'] > X_test['grade'].median()), 1, 0
)

# Feature Importance
scores = mutual_info_classif(X_train, y_train)
feature_importance = pd.Series(scores, index=X_train.columns)
print("Feature Importance:\n", feature_importance.sort_values(ascending=False))

# Train a Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Save medians for prediction
median_size = X_train['size'].median()
median_grade = X_train['grade'].median()

# Save to a file
import joblib
joblib.dump({'median_size': median_size, 'median_grade': median_grade}, 'model/medians.pkl')
print("Medians saved successfully!")

# Save the Model
import joblib
joblib.dump(model, 'model/breast_cancer_model.pkl')
print("Model saved successfully!")
