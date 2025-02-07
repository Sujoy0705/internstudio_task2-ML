import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the dataset
file_path = "loan.csv"
data = pd.read_csv(file_path)

# Drop the Loan_ID column (not useful for prediction)
data.drop(columns=["Loan_ID"], inplace=True)

# Convert Loan_Status ('Y' → 1, 'N' → 0)
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

# Handle missing values
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
cat_cols = data.select_dtypes(include=["object"]).columns

# Fill numerical columns with median
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Fill categorical columns with the most frequent value
data[cat_cols] = data[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Split features and target
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
