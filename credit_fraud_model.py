import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset
df = pd.read_csv("C:\\Users\\user\\Downloads\\creditcard_2023.csv\\creditcard_2023.csv")  # Make sure this CSV file is in the same directory

# 2. Data overview
print("Dataset Shape:", df.shape)
print(df["Class"].value_counts())  # 0: Non-fraud, 1: Fraud

# 3. Feature Scaling for 'Amount'
scaler = StandardScaler()
df["normalizedAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
# 4. Train-test split
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 5. Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions and Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Feature Importance Plot
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 10 Important Features")
plt.show()
