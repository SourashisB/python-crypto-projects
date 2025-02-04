import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset (replace 'your_dataset.csv' with the actual file name)
file_path = 'diabetes_dataset.csv'  # Update with the actual file path
data = pd.read_csv(file_path)

# Inspect the data
print("Dataset Overview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Define the target variable and features
target_column = 'Diabetes_Diagnosis'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Detect categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# Handle missing values
# For numerical columns, fill missing values with the mean
X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

# For categorical columns, fill missing values with the mode
X[categorical_columns] = X[categorical_columns].fillna(X[categorical_columns].mode().iloc[0])

# Define the preprocessing pipeline
# - StandardScaler for numerical columns
# - OneHotEncoder for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Refactor Random Forest Classifier to reduce overfitting
# Use fewer estimators, limit max depth, and increase min_samples_split and min_samples_leaf
classifier = RandomForestClassifier(
    random_state=42,
    n_estimators=50,  # Fewer trees
    max_depth=10,  # Limit depth of trees
    min_samples_split=10,  # Minimum samples required to split a node
    min_samples_leaf=5  # Minimum samples required per leaf node
)

# Define the machine learning pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
pipeline.fit(X_train, y_train)

# Cross-validate the model to evaluate performance on multiple splits
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

# Make predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nModel Evaluation:")
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix (Testing Data):")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report (Testing Data):")
print(classification_report(y_test, y_test_pred))

# Plot ROC Curve
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("ROC Curve for Diabetes Diagnosis")
plt.show()

# Plot Training vs Testing Accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'green'])
plt.ylim(0.0, 1.0)
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy')
plt.show()

# Plot Cross-Validation Scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, 6), cv_scores, marker='o', label='Cross-Validation Score')
plt.axhline(np.mean(cv_scores), color='red', linestyle='--', label='Mean CV Score')
plt.ylim(0.0, 1.0)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.legend()
plt.show()