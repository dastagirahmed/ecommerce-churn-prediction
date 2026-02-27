import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1️⃣ Load dataset
data = pd.read_csv("telco_churn.csv")  # Make sure file is in the same folder

# 2️⃣ Clean the target column (Churn)
# Strip spaces and convert to string
data['Churn'] = data['Churn'].astype(str).str.strip()

# Keep only rows with Yes/No in Churn
data = data[data['Churn'].isin(['Yes', 'No'])]

# Convert to numeric
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# 3️⃣ Clean numeric features
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = data[features]

# Convert TotalCharges to numeric, coerce errors
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# Drop rows where features have NaN
data = data.dropna(subset=features)
X = data[features]
y = data['Churn']

# 4️⃣ Check everything
print("Missing values in target:", y.isnull().sum())
print("Missing values in features:\n", X.isnull().sum())

# 5️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to avoid convergence warning
model.fit(X_train, y_train)

# 7️⃣ Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# 8️⃣ Add intentional error for university form
# Uncomment this line to see an error
# model.fit(X_train, y_train + 1)  # This will fail because target is modified