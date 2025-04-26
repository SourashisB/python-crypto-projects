import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'diabetes_dataset.csv'
data = pd.read_csv(file_path)

# Quick overview of the data
print("Dataset Overview:")
print(data.shape)
print(data.head())

# Check class distribution
print("\nTarget Distribution:")
print(data['Diabetes_Diagnosis'].value_counts(normalize=True))

# Define the target variable and features
target_column = 'Diabetes_Diagnosis'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Check class balance and apply SMOTE if needed
class_counts = np.bincount(y_train)
print(f"\nClass distribution before SMOTE: {class_counts}")

if min(class_counts) / max(class_counts) < 0.75:  # If imbalanced
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_preprocessed, y_train = smote.fit_resample(X_train_preprocessed, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train)}")

# Focus on efficient and effective models for diabetes prediction
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), random_state=42)
}

best_auc = 0
best_model_name = None
best_model = None
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train with early stopping for XGBoost
    if name == 'XGBoost':
        X_train_valid, X_valid, y_train_valid, y_valid = train_test_split(
            X_train_preprocessed, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        model.fit(
            X_train_valid, y_train_valid,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=10,
            verbose=False
        )
    else:
        model.fit(X_train_preprocessed, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_preprocessed)
    y_test_pred = model.predict(X_test_preprocessed)
    
    # For ROC AUC
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test_preprocessed)[:, 1]
        auc = roc_auc_score(y_test, y_test_proba)
    else:
        y_test_proba = model.predict(X_test_preprocessed)  # Fallback
        auc = roc_auc_score(y_test, y_test_proba)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"{name} Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")
    
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'auc': auc
    }
    
    # Track the best model based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = model

print(f"\nBest Model: {best_model_name} with AUC = {best_auc:.4f}")

# Define focused hyperparameter tuning for the best model - using RandomizedSearchCV for efficiency
if best_model_name == 'Random Forest':
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
elif best_model_name == 'Gradient Boosting':
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
elif best_model_name == 'XGBoost':
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
else:  # AdaBoost
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'base_estimator__max_depth': [1, 2, 3]
    }

# Use RandomizedSearchCV instead of GridSearchCV for efficiency
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    best_model, param_distributions, 
    n_iter=20,  # Try 20 parameter combinations
    cv=cv, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1,
    random_state=42
)

print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
random_search.fit(X_train_preprocessed, y_train)

print("\nBest Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

# Evaluate the tuned model
tuned_model = random_search.best_estimator_
y_train_pred = tuned_model.predict(X_train_preprocessed)
y_test_pred = tuned_model.predict(X_test_preprocessed)

if hasattr(tuned_model, "predict_proba"):
    y_test_proba = tuned_model.predict_proba(X_test_preprocessed)[:, 1]
else:
    y_test_proba = tuned_model.predict(X_test_preprocessed)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_proba)

print("\nTuned Model Evaluation:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"ROC AUC Score: {auc:.4f}")

print("\nConfusion Matrix (Testing Data):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Calculate and display classification metrics
test_report = classification_report(y_test, y_test_pred)
print("\nClassification Report (Testing Data):")
print(test_report)

# Create confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Analyze feature importance for tree-based models
if hasattr(tuned_model, 'feature_importances_'):
    # Get feature names
    feature_names = []
    
    # Add numerical feature names
    for col in numerical_columns:
        feature_names.append(col)
    
    # Add one-hot encoded feature names
    if categorical_columns:
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_features = encoder.get_feature_names_out(categorical_columns)
        feature_names.extend(encoded_features)
    
    # Ensure feature names match the number of importances
    importances = tuned_model.feature_importances_
    if len(feature_names) > len(importances):
        feature_names = feature_names[:len(importances)]
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

# Create the final pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', tuned_model)
])

# Fit the pipeline on the original training data
final_pipeline.fit(X_train, y_train)

# Final evaluation
pipeline_pred = final_pipeline.predict(X_test)
pipeline_accuracy = accuracy_score(y_test, pipeline_pred)
print(f"\nFinal Pipeline Accuracy: {pipeline_accuracy:.4f}")

# Save the model
import joblib
joblib.dump(final_pipeline, 'diabetes_prediction_model.pkl')
print("\nFinal model saved as 'diabetes_prediction_model.pkl'")

# Prediction function
def predict_diabetes(patient_data):
    """
    Make diabetes predictions on new patient data.
    
    Parameters:
    patient_data (DataFrame): Patient data in the same format as training data
    
    Returns:
    prediction, probability
    """
    model = joblib.load('diabetes_prediction_model.pkl')
    
    # For single samples, reshape if needed
    if len(patient_data.shape) == 1 or (isinstance(patient_data, pd.DataFrame) and len(patient_data) == 1):
        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]
    else:
        prediction = model.predict(patient_data)
        probability = model.predict_proba(patient_data)[:, 1]
        
    return prediction, probability

print("\nModel is ready for deployment. Use the predict_diabetes() function to make predictions on new data.")