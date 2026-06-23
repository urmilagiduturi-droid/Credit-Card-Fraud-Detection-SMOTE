# Task 5: Credit Card Fraud Detection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Step 1: Loading data...")
df = pd.read_csv('creditcard.csv')
print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df['Class'].value_counts())
print("Fraud %:", round(df['Class'].value_counts()[1]/len(df)*100, 4), "%")

print("\nStep 2: Scaling Amount & Time...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

print("\nStep 3: Applying SMOTE for balancing...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE - Class counts:\n", y_train_res.value_counts())

print("\nStep 4: Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_res, y_train_res)

print("\nStep 5: Predicting...")
y_pred = model.predict(X_test)

print("\n--- FINAL MODEL RESULTS ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-Score:", round(f1_score(y_test, y_pred), 4))

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Fraud Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()