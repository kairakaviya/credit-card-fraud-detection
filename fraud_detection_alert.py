import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Optional: For desktop notifications
try:
    from plyer import notification
    desktop_notification = True
except ImportError:
    desktop_notification = False
    print("plyer not installed. Desktop notifications will be skipped.")

# 1. Load dataset
df = pd.read_csv("C:\\Users\\user\\Downloads\\creditcard_2023.csv\\creditcard_2023.csv")

# 2. Data overview
print("Dataset Shape:", df.shape)
print(df["Class"].value_counts())  # 0: Non-fraud, 1: Fraud

# 3. Visualizing Fraud vs Non-Fraud Transactions
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.ylabel('Number of Transactions')
plt.xlabel('Transaction Class')
plt.show()

# 4. Feature Scaling for 'Amount'
scaler = StandardScaler()
df["normalizedAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

# 5. Drop 'Amount' and 'Time' columns
df = df.drop(["Time", "Amount"], axis=1)

# 6. Train-test split
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 7. Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predictions and Evaluation
y_pred = model.predict(X_test)

print("\n--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Fraud Detection Alerts
fraud_indices = np.where(y_pred == 1)[0]

if len(fraud_indices) > 0:
    print("\n🚨 ALERT: Fraudulent Transactions Detected!")
    for idx in fraud_indices:
        print(f"Fraud detected at Test Sample #{idx}")
        
    if desktop_notification:
        notification.notify(
            title="Credit Card Fraud Alert 🚨",
            message=f"{len(fraud_indices)} fraudulent transaction(s) detected!",
            timeout=10
        )
else:
    print("\n✅ No Fraudulent Transactions Detected.")

# 10. Feature Importance Plot
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.show()


