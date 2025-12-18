import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
data = pd.read_csv("vehicle_data.csv")

X = data[["age", "mileage", "engine_temp", "error_count"]]
y = data["risk"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("ML model trained and saved")
