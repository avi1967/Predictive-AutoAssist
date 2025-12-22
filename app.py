import pickle
import pandas as pd
import mysql.connector
from io import BytesIO

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, Response
)
from werkzeug.security import check_password_hash
from flask_mail import Mail, Message
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import os

load_dotenv()

# Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4"
    )

# Load ML Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Email Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)

# Audit Log Helper
def log_action(action, vin=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_logs (user_role, action, vin)
        VALUES (%s, %s, %s)
        """,
        (session.get('role'), action, vin)
    )
    conn.commit()
    cursor.close()
    conn.close()

def compute_vehicle_risk(vehicle):
    data = pd.DataFrame([[
        vehicle.get("age", 0),
        vehicle.get("mileage", 0),
        vehicle.get("engine_temp", 0),
        vehicle.get("error_count", 0)
    ]], columns=[
        "age", "mileage", "engine_temp", "error_count"
    ])

    prob = model.predict_proba(data)[0][1]
    risk = "High" if prob > 0.6 else "Low"

    return {
        "risk": risk,
        "risk_score": round(prob * 100, 2)
    }

def agentic_ai_response(user_message, vehicle):
    msg = user_message.lower()

    risk = vehicle['risk']
    score = vehicle['risk_score']

    # Intent detection
    if "risk" in msg:
        return (
            f"Your vehicle has a {score}% failure probability, "
            f"which is classified as {risk} risk based on sensor data "
            f"and historical failure patterns."
        )

    if "why" in msg:
        return (
            "The risk is driven by high mileage, engine temperature trends, "
            "and recent error codes indicating component wear."
        )

    if "service" in msg or "book" in msg:
        return (
            "I recommend scheduling a service within the next 2–3 weeks. "
            "You can book an appointment directly from the Service Scheduling section."
        )

    if "cost" in msg:
        return (
            "The estimated service cost is calculated from historical maintenance data "
            "and will be confirmed by the service center."
        )

    # Default fallback
    return (
        "I’m here to help with vehicle risk, service recommendations, "
        "maintenance history, and cost insights. What would you like to know?"
    )


# Load Vehicles + ML
def load_vehicles():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 🔒 Customer sees ONLY their own vehicle
    if session.get('role') == 'customer':
        cursor.execute(
            "SELECT * FROM vehicles WHERE vin = %s",
            (session['vin'],)
        )
    else:
        cursor.execute("SELECT * FROM vehicles")

    vehicles = cursor.fetchall()

    for v in vehicles:
        # Prepare ML input
        data = pd.DataFrame([[
            v.get("age", 0),
            v.get("mileage", 0),
            v.get("engine_temp", 0),
            v.get("error_count", 0)
        ]], columns=[
            "age",
            "mileage",
            "engine_temp",
            "error_count"
        ])

        # ML Prediction
        prob = model.predict_proba(data)[0][1]

        v["risk"] = "High" if prob > 0.6 else "Low"
        v["risk_score"] = round(prob * 100, 2)

        v["alert"] = (
            "Immediate service recommended"
            if v["risk"] == "High"
            else "Vehicle operating normally"
        )

        # Ensure notified flag exists (safety fallback)
        if "notified" not in v:
            v["notified"] = 0

    cursor.close()
    conn.close()

    return vehicles

def ai_auto_suggestions(vehicle):
    suggestions = []

    if vehicle['risk'] == 'High':
        suggestions.append("Book service appointment")
        suggestions.append("Explain risk calculation")

    suggestions.append("View maintenance report")
    suggestions.append("How is my vehicle health?")

    return suggestions

# Load Appointments
def load_appointments():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if session['role'] == 'customer':
        cursor.execute(
            "SELECT * FROM service_appointments WHERE vin=%s",
            (session['vin'],)
        )
    else:
        cursor.execute("SELECT * FROM service_appointments")

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Routes
@app.route('/')
def dashboard():
    if 'role' not in session:
        return redirect(url_for('login'))

    if session['role'] == 'customer':
        return redirect(url_for('chat', vin=session['vin']))

    vehicles = load_vehicles()
    return render_template('dashboard.html', vehicles=vehicles)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM users WHERE username=%s",
            (request.form['username'],)
        )
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], request.form['password']):
            session['user_id'] = user['id']
            session['role'] = user['role']
            session['vin'] = user['vin']
            log_action("LOGIN", user['vin'])

            if user['role'] == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('chat', vin=user['vin']))

        return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


@app.route('/vehicle-health')
def vehicle_health():
    if 'role' not in session:
        return redirect(url_for('login'))

    vehicles = load_vehicles()
    return render_template('vehicle_health.html', vehicles=vehicles)


@app.route('/predictions')
def predictions():
    if 'role' not in session:
        return redirect(url_for('login'))

    vehicles = load_vehicles()
    return render_template('predictions.html', vehicles=vehicles)


@app.route('/service-scheduling')
def service_scheduling():
    if 'role' not in session:
        return redirect(url_for('login'))

    appointments = load_appointments()
    return render_template(
        'service_scheduling.html',
        appointments=appointments,
        role=session['role'],
        vin=session.get('vin')
    )


@app.route('/reports')
def reports():
    if 'role' not in session:
        return redirect(url_for('login'))

    vehicles = load_vehicles()
    return render_template('reports.html', vehicles=vehicles)


@app.route('/schedule/<vin>', methods=['GET', 'POST'])
def schedule(vin):
    if 'role' not in session:
        return redirect(url_for('login'))

    if session['role'] == 'customer' and session['vin'] != vin:
        return "Unauthorized", 403

    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO service_appointments
            (vin, service_center, service_date, service_time, status, cost)
            VALUES (%s,%s,%s,%s,'Scheduled',%s)
            """,
            (
                vin,
                request.form['service_center'],
                request.form['service_date'],
                request.form['service_time'],
                request.form.get('cost')
            )
        )

        conn.commit()
        cursor.close()
        conn.close()

        log_action("SERVICE_BOOKED", vin)

        msg = Message(
            "Service Appointment Confirmed",
            sender=app.config['MAIL_USERNAME'],
            recipients=["customer@email.com"]
        )
        msg.body = f"Service booked for vehicle {vin}"
        mail.send(msg)

        return redirect(url_for('confirmation'))

    return render_template(
        'schedule.html',
        vin=vin,
        avg_cost=4500,
        role=session['role']
    )


@app.route('/chat/<vin>', methods=['GET', 'POST'])
def chat(vin):
    if 'role' not in session:
        return redirect(url_for('login'))

    if session['role'] == 'customer' and session['vin'] != vin:
        return "Unauthorized", 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Load raw vehicle data
    cursor.execute("SELECT * FROM vehicles WHERE vin=%s", (vin,))
    vehicle = cursor.fetchone()

    if not vehicle:
        return "Vehicle not found", 404

    # Compute ML risk dynamically
    ml = compute_vehicle_risk(vehicle)
    vehicle['risk'] = ml['risk']
    vehicle['risk_score'] = ml['risk_score']

    # POST: Customer message
    if request.method == 'POST':

        if session['role'] == 'admin':
            return redirect(url_for('chat', vin=vin))

        user_message = request.form['message']

        # Save customer message
        cursor.execute(
            """
            INSERT INTO chat_messages (vin, sender_role, message)
            VALUES (%s, %s, %s)
            """,
            (vin, "customer", user_message)
        )

        # Generate AI reply (NOW SAFE)
        ai_reply = agentic_ai_response(user_message, vehicle)

        cursor.execute(
            """
            INSERT INTO chat_messages (vin, sender_role, message)
            VALUES (%s, %s, %s)
            """,
            (vin, "ai", ai_reply)
        )

        log_action("AI_CHAT_RESPONSE", vin)
        conn.commit()

    # Load chat history
    cursor.execute(
        """
        SELECT sender_role, message, timestamp
        FROM chat_messages
        WHERE vin=%s
        ORDER BY timestamp
        """,
        (vin,)
    )
    messages = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        'chat.html',
        vin=vin,
        messages=messages,
        role=session['role']
    )


@app.route('/notify/<vin>')
def notify_customer(vin):
    if 'role' not in session or session['role'] != 'admin':
        return "Unauthorized", 403

    conn = get_db_connection()
    cursor = conn.cursor()

    # Prevent duplicate notifications
    cursor.execute(
        "SELECT notified FROM vehicles WHERE vin=%s",
        (vin,)
    )
    row = cursor.fetchone()

    if row and row[0] == 1:
        return redirect(url_for('dashboard'))

    # AGENTIC AI MESSAGE 
    ai_message = (
        "⚠️ AutoCare AI Alert:\n\n"
        "Our predictive model has detected a high probability of component failure "
        "within the next 2–3 weeks. We strongly recommend scheduling a service "
        "to avoid unexpected breakdowns.\n\n"
        "You can book a service directly from the app."
    )

    # Insert AI message
    cursor.execute(
        """
        INSERT INTO chat_messages (vin, sender_role, message)
        VALUES (%s, %s, %s)
        """,
        (vin, "ai", ai_message)
    )

    # Mark vehicle as notified
    cursor.execute(
        "UPDATE vehicles SET notified=1 WHERE vin=%s",
        (vin,)
    )

    # Audit log
    cursor.execute(
        """
        INSERT INTO audit_logs (user_role, action, vin)
        VALUES (%s, %s, %s)
        """,
        ("admin", "AI_NOTIFICATION_SENT", vin)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return redirect(url_for('dashboard'))


@app.route('/audit-logs')
def audit_logs():
    if 'role' not in session or session['role'] != 'admin':
        return "Unauthorized", 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('audit_logs.html', logs=logs)


@app.route('/logout')
def logout():
    log_action("LOGOUT", session.get('vin'))
    session.clear()
    return redirect(url_for('login'))


@app.route('/confirmation')
def confirmation():
    return render_template('confirmation.html')


if __name__ == '__main__':
    app.run(debug=True)


