import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load data
df = pd.read_csv('diabetes_dataset.csv')

# 2. Encode categorical variables

# Map ordinal/string columns to numbers
ordinal_maps = {
    'Smoking_Status': {'Never': 0, 'Former': 1, 'Current': 2},
    'Physical_Activity_Level': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Stress_Level': {'Low': 0, 'Moderate': 1, 'High': 2}
}
for col, mapping in ordinal_maps.items():
    df[col] = df[col].map(mapping)

# One-hot encode Gender and Ethnicity
df = pd.get_dummies(df, columns=['Gender', 'Ethnicity'], drop_first=True)

# 3. Drop leakage columns
leakage_cols = ['Glucose_Level', 'HbA1c', 'Insulin_Resistance']
X = df.drop(['Diabetes_Diagnosis'] + leakage_cols, axis=1)
y = df['Diabetes_Diagnosis']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Fit Random Forest (with class_weight for balance)
rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 6. Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))