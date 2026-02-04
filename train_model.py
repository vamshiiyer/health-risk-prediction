import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("health_data.csv")

X = data.drop("risk", axis=1)
y = data["risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model trained successfully")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "health_risk_model.pkl")
print("💾 Model saved as health_risk_model.pkl")
