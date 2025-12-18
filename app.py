import pickle
import pandas as pd
from flask import Flask, render_template
import json

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# -------- ML Prediction Function --------
def predict_risk(vehicle):
    data = pd.DataFrame([[
        vehicle.get("age", 0),
        vehicle.get("mileage", 0),
        vehicle.get("engine_temp", 0),
        vehicle.get("error_count", 0)
    ]], columns=["age", "mileage", "engine_temp", "error_count"])

    prediction = model.predict(data)[0]
    return "High" if prediction == 1 else "Low"


# -------- Load Vehicles & Apply ML --------
def load_vehicles():
    with open('vehicles.json') as f:
        vehicles = json.load(f)

    for v in vehicles:
        v["risk"] = predict_risk(v)
        v["alert"] = (
            "Immediate service recommended"
            if v["risk"] == "High"
            else "Vehicle operating normally"
        )

    return vehicles


# -------- Routes --------
@app.route('/')
def dashboard():
    vehicles = load_vehicles()
    return render_template('dashboard.html', vehicles=vehicles)


@app.route('/schedule/<vin>')
def schedule(vin):
    return render_template('schedule.html', vin=vin)


@app.route('/chat/<vin>')
def chat(vin):
    return render_template('chat.html', vin=vin)


@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')


if __name__ == '__main__':
    app.run(debug=True)
