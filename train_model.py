import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(
    n_samples=1000, n_features=8, random_state=42, n_classes=2
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to 'model/breast_cancer_model.pkl'
joblib.dump(model, 'model/breast_cancer_model.pkl')
print("Model saved successfully!")

