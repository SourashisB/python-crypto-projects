import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'diabetes_dataset.csv'
data = pd.read_csv(file_path)

# Display initial information
print("Dataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)

# Check class distribution
print("\nTarget Distribution:")
print(data['Diabetes_Diagnosis'].value_counts(normalize=True))

# Data quality check
print("\nMissing values per column:")
print(data.isnull().sum())

# Define the target variable and features
target_column = 'Diabetes_Diagnosis'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Convert target to numeric if it's not already
if y.dtype == 'object':
    print("Converting target to numeric...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(f"Target classes: {label_encoder.classes_}")

# Identify categorical and numerical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumerical features:", numerical_columns)
print("Categorical features:", categorical_columns)

# Exploratory Data Analysis
print("\nPerforming Exploratory Data Analysis...")

# Check for class imbalance
plt.figure(figsize=(8, 6))
sns.countplot(x=target_column, data=data)
plt.title('Class Distribution')
plt.show()

# Distribution of numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns[:min(9, len(numerical_columns))]):
    plt.subplot(3, 3, i+1)
    sns.histplot(data=data, x=col, hue=target_column, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 10))
numeric_data = data[numerical_columns + [target_column]].copy()
# Ensure target is numeric for correlation
if numeric_data[target_column].dtype == 'object':
    numeric_data[target_column] = label_encoder.transform(numeric_data[target_column])
correlation = numeric_data.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Feature Correlation Matrix')
plt.show()

# Create custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols):
        self.numerical_cols = numerical_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Only perform operations on numerical columns
        for col in self.numerical_cols:
            if col in X_new.columns:
                # Create log features for skewed numerical features
                skewness = X_new[col].skew()
                if abs(skewness) > 1:
                    # Ensure no negative values for log transform
                    if X_new[col].min() <= 0:
                        X_new[f'{col}_log'] = np.log1p(X_new[col] - X_new[col].min() + 1)
                    else:
                        X_new[f'{col}_log'] = np.log1p(X_new[col])
        
        # Create interaction terms only for numerical features
        numerical_present = [col for col in self.numerical_cols if col in X_new.columns]
        
        if 'BMI' in numerical_present and 'Age' in numerical_present:
            X_new['BMI_x_Age'] = X_new['BMI'] * X_new['Age']
        
        if 'Glucose' in numerical_present and 'BMI' in numerical_present:
            X_new['Glucose_x_BMI'] = X_new['Glucose'] * X_new['BMI']
        
        if 'Glucose' in numerical_present and 'Age' in numerical_present:
            X_new['Glucose_x_Age'] = X_new['Glucose'] * X_new['Age']
        
        # Create ratio features for numerical values
        if 'BMI' in numerical_present and 'Age' in numerical_present:
            X_new['BMI_to_Age'] = X_new['BMI'] / (X_new['Age'] + 1)  # +1 to avoid division by zero
        
        # Polynomial features for key numerical variables
        if 'Glucose' in numerical_present:
            X_new['Glucose_squared'] = X_new['Glucose'] ** 2
        
        if 'BMI' in numerical_present:
            X_new['BMI_squared'] = X_new['BMI'] ** 2
        
        return X_new

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Apply feature engineering for EDA
feature_engineer = FeatureEngineer(numerical_cols=numerical_columns)
X_train_engineered = feature_engineer.transform(X_train)

# Get updated numerical columns list after feature engineering
engineered_numerical_cols = X_train_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
engineered_categorical_cols = X_train_engineered.select_dtypes(include=['object', 'category']).columns.tolist()

# Define the preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create the preprocessing pipeline that will be used after feature engineering
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, engineered_numerical_cols),
        ('cat', categorical_transformer, engineered_categorical_cols)
    ]
)

# Define a complete pipeline function that includes feature engineering
def create_pipeline(estimator, feature_selection='none', smote=True):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    
    # Steps for the pipeline
    steps = []
    
    # Feature engineering
    steps.append(('feature_engineering', FeatureEngineer(numerical_cols=numerical_columns)))
    
    # Preprocessing
    steps.append(('preprocessor', preprocessor))
    
    # Feature selection
    if feature_selection == 'kbest':
        steps.append(('feature_selection', SelectKBest(f_classif, k=min(20, len(engineered_numerical_cols)))))
    elif feature_selection == 'rfe':
        steps.append(('feature_selection', RFE(estimator=LogisticRegression(max_iter=1000), 
                                               n_features_to_select=min(15, len(engineered_numerical_cols)))))
    
    # Add SMOTE if requested
    if smote:
        steps.append(('smote', SMOTE(random_state=42)))
    
    # Add the estimator
    steps.append(('estimator', estimator))
    
    # Return as imbalanced-learn Pipeline if using SMOTE, otherwise as sklearn Pipeline
    if smote:
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)

# Define estimators with regularization to reduce overfitting
estimators = {
    'LogisticRegression': LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000, 
                                             class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, 
                                           class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                                   learning_rate=0.1, subsample=0.8, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, 
                                       class_weight='balanced', random_state=42),
    'LinearSVC': LinearSVC(C=0.1, penalty='l2', dual=False, class_weight='balanced', 
                           max_iter=2000, random_state=42)
}

# Define feature selection methods and SMOTE options
feature_selections = ['none', 'kbest', 'rfe']
smote_options = [True, False]

# Use cross-validation to select the best approach
best_score = 0
best_pipeline = None
best_config = None

print("\nTesting model configurations with cross-validation...")

# Prepare cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

# Test different configurations
results = []

# Count total iterations
total_configs = len(estimators) * len(feature_selections) * len(smote_options)
print(f"Testing {total_configs} different configurations...")

# Test different configurations
for estimator_name, estimator in estimators.items():
    for feature_selection in feature_selections:
        for smote in smote_options:
            config = f"{estimator_name} | Selection: {feature_selection} | SMOTE: {smote}"
            print(f"Testing: {config}")
            
            # Create pipeline
            pipeline = create_pipeline(
                estimator=estimator,
                feature_selection=feature_selection,
                smote=smote
            )
            
            # Evaluate with cross-validation
            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                print(f"  ROC AUC: {mean_score:.4f} (±{std_score:.4f})")
                
                # Store result
                results.append({
                    'estimator': estimator_name,
                    'feature_selection': feature_selection,
                    'smote': smote,
                    'mean_score': mean_score,
                    'std_score': std_score
                })
                
                # Update best if better
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = config
                    best_pipeline = pipeline
            except Exception as e:
                print(f"  Error: {str(e)}")

# Display best config
print(f"\nBest configuration: {best_config}")
print(f"Best cross-validation ROC AUC: {best_score:.4f}")

# Display all results in a DataFrame
results_df = pd.DataFrame(results)
print("\nAll configurations sorted by performance:")
print(results_df.sort_values('mean_score', ascending=False).head(10))

# Fit best pipeline to training data
print("\nFitting best model to training data...")
best_pipeline.fit(X_train, y_train)

# Evaluate on test data
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, 'predict_proba') else None

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_report = classification_report(y_test, y_pred)
test_confusion = confusion_matrix(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

print("\nTest Results:")
print(f"Accuracy: {test_accuracy:.4f}")
if test_auc is not None:
    print(f"ROC AUC: {test_auc:.4f}")
print("\nClassification Report:")
print(test_report)
print("\nConfusion Matrix:")
print(test_confusion)

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create ensemble from top models
print("\nCreating an ensemble model from top performing configurations...")

# Get top 3 configurations
top_configs = results_df.sort_values('mean_score', ascending=False).head(3)

# Create and fit an ensemble
ensemble_members = []
for idx, config in top_configs.iterrows():
    estimator_name = config['estimator']
    feature_selection = config['feature_selection']
    smote = config['smote']
    
    # Create pipeline
    pipeline = create_pipeline(
        estimator=estimators[estimator_name],
        feature_selection=feature_selection,
        smote=smote
    )
    
    # Add to ensemble
    ensemble_members.append((f"model_{idx}", pipeline))

# Create voting classifier
ensemble_estimators = [(f"model_{i}", member[1]) for i, member in enumerate(ensemble_members)]
voting_classifier = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft'  # Use probability-based voting
)

# Fit voting classifier
print("Fitting ensemble model...")
voting_classifier.fit(X_train, y_train)

# Evaluate ensemble
y_ensemble_pred = voting_classifier.predict(X_test)
y_ensemble_proba = voting_classifier.predict_proba(X_test)[:, 1]

# Calculate metrics
ensemble_accuracy = accuracy_score(y_test, y_ensemble_pred)
ensemble_report = classification_report(y_test, y_ensemble_pred)
ensemble_confusion = confusion_matrix(y_test, y_ensemble_pred)
ensemble_auc = roc_auc_score(y_test, y_ensemble_proba)

print("\nEnsemble Test Results:")
print(f"Accuracy: {ensemble_accuracy:.4f}")
print(f"ROC AUC: {ensemble_auc:.4f}")
print("\nClassification Report:")
print(ensemble_report)
print("\nConfusion Matrix:")
print(ensemble_confusion)

# Final model: use the best between the best individual model and the ensemble
final_model = best_pipeline
final_accuracy = test_accuracy

if ensemble_accuracy > test_accuracy:
    final_model = voting_classifier
    final_accuracy = ensemble_accuracy
    print(f"\nFinal model selected: Ensemble")
else:
    print(f"\nFinal model selected: Best Individual Model ({best_config})")

print(f"Final test accuracy: {final_accuracy:.4f}")

# Save the model
import joblib
joblib.dump(final_model, 'improved_diabetes_prediction_model.pkl')
print("\nFinal model saved as 'improved_diabetes_prediction_model.pkl'")

# Function to make predictions on new data
def predict_diabetes(patient_data):
    """
    Make diabetes predictions on new patient data.
    
    Parameters:
    patient_data (DataFrame): Patient data in the same format as training data
    
    Returns:
    prediction, probability
    """
    model = joblib.load('improved_diabetes_prediction_model.pkl')
    
    # For single samples, reshape if needed
    if len(patient_data.shape) == 1 or (isinstance(patient_data, pd.DataFrame) and len(patient_data) == 1):
        prediction = model.predict(patient_data)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(patient_data)[0][1]
        else:
            probability = None
    else:
        prediction = model.predict(patient_data)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(patient_data)[:, 1]
        else:
            probability = None
        
    return prediction, probability

print("\nModel is ready for deployment. Use the predict_diabetes() function to make predictions on new data.")