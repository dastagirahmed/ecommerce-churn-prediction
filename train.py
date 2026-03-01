
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
data = pd.read_csv("telco_churn.csv")

# Clean target column
data['Churn'] = data['Churn'].astype(str).str.strip()
data = data[data['Churn'].isin(['Yes','No'])]
data['Churn'] = data['Churn'].map({'Yes':1,'No':0})

# Clean numeric features
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing numeric features
data = data.dropna(subset=features)

# Reassign X and y
X = data[features]
y = data['Churn']

# Check missing values
print("Missing values in target:", y.isnull().sum())
print("Missing values in features:\n", X.isnull().sum())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict & evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)


