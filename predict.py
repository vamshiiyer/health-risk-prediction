import joblib

# Load trained model
model = joblib.load("health_risk_model.pkl")

print("\n🩺 AI Health Risk Prediction System\n")

# User input
age = int(input("Enter age: "))
bmi = float(input("Enter BMI: "))
bp = int(input("Enter blood pressure: "))
sugar = int(input("Enter blood sugar level: "))
smoker = int(input("Smoker? (1 = Yes, 0 = No): "))
activity = int(input("Physically active? (1 = Yes, 0 = No): "))

user_data = [[age, bmi, bp, sugar, smoker, activity]]

risk = model.predict(user_data)[0]

print("\n--- RESULT ---")

if risk == 0:
    print("🟢 Health Risk: LOW")
    print("Advice: Maintain a healthy lifestyle.")
elif risk == 1:
    print("🟡 Health Risk: MEDIUM")
    print("Advice: Improve diet, exercise, and monitor health regularly.")
else:
    print("🔴 Health Risk: HIGH")
    print("Advice: Consult a healthcare professional soon.")
