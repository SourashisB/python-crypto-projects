import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style='whitegrid')

# Load the CSV file
file_path = 'diabetes_dataset.csv' 
df = pd.read_csv(file_path)

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show basic info
print("\nData Info:")
df.info()

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Check missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Visualize missing values as a heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Distribution plots for numerical features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove('Diabetes_Diagnosis')  # Target variable

for col in num_features:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Count plots for categorical features
cat_features = df.select_dtypes(include=['object']).columns.tolist()
cat_features = [c for c in cat_features if c != 'Diabetes_Diagnosis']

for col in cat_features:
    plt.figure(figsize=(6, 3))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot: {col}')
    plt.xticks(rotation=45)
    plt.show()

# Distribution of target variable
plt.figure(figsize=(6, 3))
sns.countplot(data=df, x='Diabetes_Diagnosis')
plt.title('Target Variable Distribution')
plt.show()

# Pairplot for selected features (sample to avoid overload)
sample_cols = ['Age', 'BMI', 'Blood_Pressure', 'Glucose_Level', 'HbA1c', 'Diabetes_Diagnosis']
sns.pairplot(df[sample_cols].dropna(), hue='Diabetes_Diagnosis', diag_kind='kde')
plt.show()

# Correlation matrix and heatmap (numerical features)
plt.figure(figsize=(12, 8))
corr = df[num_features + ['Diabetes_Diagnosis']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Boxplots of numerical features by target
for col in num_features:
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, x='Diabetes_Diagnosis', y=col)
    plt.title(f'{col} by Diabetes Diagnosis')
    plt.show()

# Cross-tabulations for categorical features vs target
for col in cat_features:
    ct = pd.crosstab(df[col], df['Diabetes_Diagnosis'])
    print(f"\nCross-tabulation: {col} vs Diabetes_Diagnosis")
    print(ct)
    ct.plot(kind='bar', stacked=True, figsize=(7, 4))
    plt.title(f'{col} vs Diabetes Diagnosis')
    plt.ylabel('Count')
    plt.show()