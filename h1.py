import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Create folders if they don't exist
folders = ['Folder1', 'Folder2', 'Folder3', 'Folder4', 'Folder5']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Generate Transaction Data
transaction_records = pd.DataFrame({
    'TransactionID': range(1, 1001),
    'Amount': np.random.uniform(10, 100, 1000),
    'CustomerID': np.random.randint(1001, 2001, 1000)
})

# ... (generate other dataframes similarly)

# Save the generated data into CSV files
transaction_records.to_csv('Folder1/transaction_records.csv', index=False)
# ... (save other dataframes similarly)

# Load new dataset
df_transaction_records = pd.read_csv('Folder1/transaction_records.csv')
df_transaction_metadata = pd.read_csv('Folder1/transaction_metadata.csv')
df_customer_data = pd.read_csv('Folder2/customer_data.csv')
df_account_activity = pd.read_csv('Folder2/account_activity.csv')
df_fraud_indicators = pd.read_csv('Folder3/fraud_indicators.csv')
df_suspicious_activity = pd.read_csv('Folder3/suspicious_activity.csv')
df_amount_data = pd.read_csv('Folder4/amount_data.csv')
df_anomaly_scores = pd.read_csv('Folder4/anomaly_scores.csv')
df_merchant_data = pd.read_csv('Folder5/merchant_data.csv')
df_transaction_category_labels = pd.read_csv('Folder5/transaction_category_labels.csv')

# Merge DataFrames based on common columns
df_merged = pd.merge(df_transaction_records, df_transaction_metadata, on='TransactionID', how='inner')
df_merged = pd.merge(df_merged, df_customer_data, on='CustomerID', how='inner')
df_merged = pd.merge(df_merged, df_account_activity, on='CustomerID', how='inner')
df_merged = pd.merge(df_merged, df_fraud_indicators, on='TransactionID', how='inner')
df_merged = pd.merge(df_merged, df_suspicious_activity, on='CustomerID', how='inner')
df_merged = pd.merge(df_merged, df_amount_data, on='TransactionID', how='inner')
df_merged = pd.merge(df_merged, df_anomaly_scores, on='TransactionID', how='inner')
df_merged = pd.merge(df_merged, df_merchant_data, on='MerchantID', how='inner')
df_merged = pd.merge(df_merged, df_transaction_category_labels, on='TransactionID', how='inner')

# Feature engineering and handle datetime conversion
df_merged['Timestamp'] = pd.to_datetime(df_merged['Timestamp'], errors='coerce')
df_merged['LastLogin'] = pd.to_datetime(df_merged['LastLogin'], errors='coerce')
df_merged['TransactionHour'] = df_merged['Timestamp'].dt.hour

# One-hot encode categorical columns
df_merged = pd.get_dummies(df_merged, columns=['Category', 'Name', 'Address', 'MerchantName', 'Location'], drop_first=True)

# Handle missing values if necessary
df_merged.fillna(0, inplace=True)

# Exclude datetime columns from features
X = df_merged.drop(['FraudIndicator', 'SuspiciousFlag', 'Timestamp', 'LastLogin'], axis=1)
y = df_merged['SuspiciousFlag']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation - Logistic Regression
logmodel = LogisticRegression(class_weight='balanced', random_state=42)
logmodel.fit(X_train_scaled, y_train)
predictions = logmodel.predict(X_test_scaled)
print("Logistic Regression Results with Balanced Class Weights:")
print(classification_report(y_test, predictions))
conf_matrix_logreg = confusion_matrix(y_test, predictions)
print(conf_matrix_logreg)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Model Training and Evaluation - Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)
predictions = rfc.predict(X_test_scaled)
print("\nRandom Forest Classifier Results:")
print(classification_report(y_test, predictions, zero_division=1))
conf_matrix_rfc = confusion_matrix(y_test, predictions)
print(conf_matrix_rfc)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rfc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Random Forest Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Model Training and Evaluation - Support Vector Machine Classifier
svc = SVC()
svc.fit(X_train_scaled, y_train)
predictions = svc.predict(X_test_scaled)
print("\nSupport Vector Machine Results:")
print(classification_report(y_test, predictions, zero_division=1))
conf_matrix_svc = confusion_matrix(y_test, predictions)
print(conf_matrix_svc)


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Support Vector Machine Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# Model Training and Evaluation - Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train_scaled, y_train)
predictions = gbc.predict(X_test_scaled)
print("\nGradient Boosting Classifier Results:")
print(classification_report(y_test, predictions, zero_division=1))
conf_matrix_gbc = confusion_matrix(y_test, predictions)
print(conf_matrix_gbc)

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_gbc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Gradient Boosting Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Obtain a tuple of tuples for malicious transactions
malicious_transactions = df_merged[df_merged['SuspiciousFlag'] == 1]
result_tuple = tuple(tuple(row[['TransactionID', 'Amount', 'CustomerID']]) for _, row in malicious_transactions.iterrows())
print(result_tuple)



