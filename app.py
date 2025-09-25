import os
import shutil
import pandas as pd
import sqlite3
import threading
import re
import io
import base64
import json
import matplotlib

matplotlib.use("Agg")  # Fix for threading issues
import matplotlib.pyplot as plt
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from chatbot_model import get_chat_response  # Make sure chatbot_model.py exists
from bs4 import BeautifulSoup
import traceback
from Appointment.generate_time import generate_time_slots
from datetime import date
import time
import requests
from Appointment.view_data import fetch_appointments_from_db
from Appointment.Intent_appoint import parse_booking_command

# === Paths ===
stop_execution_flag = False
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"csv", "db"}
STATIC_CSV = os.path.join(BASE_DIR, "patient_details2.csv")  # Default CSV
DB_FILE = os.path.join(BASE_DIR, "chatbot_data.db")
# === Flask App ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "AIzaSyDhSrwZaIdEM2WVIELNAu7qIa-WRfbsqn4"


# === DB Initialization ===
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_history 
                        (id INTEGER PRIMARY KEY, message TEXT, response TEXT)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS current_file 
                        (id INTEGER PRIMARY KEY, filename TEXT)"""
        )
        conn.execute(
            """ CREATE TABLE IF NOT EXISTS patients ( 
                     id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     mrn_number TEXT, patient_name TEXT, registration_date TEXT,
                     city TEXT, invoice_number TEXT, 
                     invoice_date TEXT, description TEXT, 
                     price REAL, doctor_name TEXT ) """
        )

        # New Appointments table
        conn.execute(
            """CREATE TABLE IF NOT EXISTS appointments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            patient_name TEXT NOT NULL,
                            doctor TEXT NOT NULL,
                            date TEXT NOT NULL,
                            time TEXT NOT NULL,
                            status TEXT DEFAULT 'booked'
                        )"""
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS content_links 
                        (id TEXT PRIMARY KEY, content_type TEXT, content_data TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()


init_db()
# === Cache & Lock ===
data_cache = None
data_lock = threading.Lock()


# === File Utils ===
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_current_file():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM current_file ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
    return result[0] if result else None


def set_current_file(filename):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM current_file")
        cursor.execute("INSERT INTO current_file (filename) VALUES (?)", (filename,))
        conn.commit()


def load_data():
    global data_cache
    current_file = get_current_file()
    if current_file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], current_file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                with data_lock:
                    data_cache = df
                print(f"[DATA] Loaded {current_file} into cache")
            except Exception as e:
                print(f"[DATA] Failed to read CSV {file_path}: {e}")
                with data_lock:
                    data_cache = None
        else:
            with data_lock:
                data_cache = None
    else:
        with data_lock:
            data_cache = None


# Change STATIC_CSV path to match where you actually store it in repo
STATIC_CSV = os.path.join(BASE_DIR, "uploads", "patient_details2.csv")


def bootstrap_dataset():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    current = get_current_file()
    current_path = os.path.join(UPLOAD_FOLDER, current) if current else None
    needs_seed = (not current) or (current and not os.path.exists(current_path))
    if needs_seed:
        if os.path.exists(STATIC_CSV):
            dest = os.path.join(UPLOAD_FOLDER, os.path.basename(STATIC_CSV))
            shutil.copy(STATIC_CSV, dest)  # Always overwrite to be safe
            set_current_file(os.path.basename(STATIC_CSV))
            print(f"[INIT] Seed dataset loaded: {dest}")
        else:
            print(f"[INIT] No static CSV found at {STATIC_CSV}")


try:
    bootstrap_dataset()
    load_data()
except Exception as e:
    print(f"[INIT] Bootstrap error: {e}")


# === Helper function to parse HTML table to JSON ===
def parse_table_to_json(html_content):
    """Parse HTML table to structured JSON data"""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")

        if not table:
            return None

        # Extract headers
        headers = []
        header_row = table.find("tr")
        if header_row:
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))

        # Extract rows
        rows = []
        for row in table.find_all("tr")[1:]:  # Skip header row
            cells = []
            for cell in row.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(cells)

        return {"headers": headers, "rows": rows}
    except Exception as e:
        print(f"[ERROR] Error parsing table: {str(e)}")
        return None


# === Helper function to generate table HTML ===
def generate_table_html(headers, rows):
    """Generate HTML table from headers and rows"""
    html = "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"

    # Add headers
    html += "<tr style='background-color: #f2f2f2;'>\n"
    for header in headers:
        html += f"<th style='padding: 8px; text-align: left;'>{header}</th>\n"
    html += "</tr>\n"

    # Add rows
    for i, row in enumerate(rows):
        bg_color = "#f9f9f9" if i % 2 == 0 else "white"
        html += f"<tr style='background-color: {bg_color};'>\n"
        for cell in row:
            html += f"<td style='padding: 8px;'>{cell}</td>\n"
        html += "</tr>\n"

    html += "</table>"
    return html


@app.route("/get_slots", methods=["GET"])
def get_slots():
    doctor_id = request.args.get("doctor_id")
    date = request.args.get("date")

    # For now, ignore doctor_id & date
    slots = generate_time_slots()

    return jsonify(
        {"status": "success", "doctor_id": doctor_id, "date": date, "slots": slots}
    )


# === Helper function to generate chart HTML ===
def generate_chart_html(chart_data):
    """Generate HTML for chart from chart data"""
    try:
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        title = chart_data.get("title", "Chart")
        chart_type = chart_data.get(
            "chart_type", "bar"
        )  # Get chart type, default to bar

        # Generate chart based on type
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type.lower() == "pie":
            # For pie charts, we need to ensure values are positive
            pie_values = [abs(float(v)) for v in values]
            # Create pie chart
            ax.pie(pie_values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        elif chart_type.lower() == "line":
            # For line charts, we need to convert labels to numeric if they're not already
            try:
                # Try to convert labels to numeric values for x-axis
                x_values = [
                    float(i) for i in range(len(labels))
                ]  # Use index as x-values
                y_values = [float(v) for v in values]

                # Create line chart
                ax.plot(x_values, y_values, marker="o", linestyle="-")

                # Set x-axis ticks to show actual labels
                ax.set_xticks(x_values)
                ax.set_xticklabels(labels, rotation=45, ha="right")
            except (ValueError, TypeError) as e:
                print(f"[ERROR] Error creating line chart: {str(e)}")
                # Fall back to bar chart if line chart fails
                ax.bar(labels, values)
                chart_type = "bar"
        else:
            # Default to bar chart
            ax.bar(labels, values)
            ax.set_xlabel("Category")
            ax.set_ylabel("Value")
            plt.xticks(rotation=45, ha="right")

        ax.set_title(title)

        # Add appropriate labels based on chart type
        if chart_type.lower() == "line":
            ax.set_xlabel("Category")
            ax.set_ylabel("Value")
        elif chart_type.lower() != "pie":  # For bar and other charts
            ax.set_xlabel("Category")
            ax.set_ylabel("Value")

        plt.tight_layout()

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        # Create HTML
        html = f"""
        <div style="text-align: center; margin: 20px;">
            <h2>{title}</h2>
            <img src="data:image/png;base64,{plot_url}" alt="{title}" style="max-width: 100%; height: auto;">
        </div>
        """
        return html
    except Exception as e:
        print(f"[ERROR] Error generating chart HTML: {str(e)}")
        return f"<p>Error generating chart: {str(e)}</p>"


# === Helper function to convert table to text list ===
def table_to_text_list(html_content):
    """Convert HTML table to text list format"""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table")

        if not table:
            return html_content

        # Extract headers
        headers = []
        header_row = table.find("tr")
        if header_row:
            for th in header_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True))

        # Extract rows
        rows = []
        for row in table.find_all("tr")[1:]:  # Skip header row
            cells = []
            for cell in row.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(cells)

        # Convert to text list format
        text_list = ""
        for i, row in enumerate(rows):
            text_list += f"{i+1}. "
            for j, cell in enumerate(row):
                if j < len(headers):
                    text_list += f"{headers[j]}: {cell}"
                else:
                    text_list += f"{cell}"
                if j < len(row) - 1:
                    text_list += ", "
            text_list += "\n"

        return text_list
    except Exception as e:
        print(f"[ERROR] Error converting table to text list: {str(e)}")
        return html_content


# === Helper function to check if user is asking for a list ===
def is_asking_for_list(user_input):
    """Check if the user is asking for a list format"""
    list_keywords = [
        "list",
        "listing",
        "show me list",
        "give me list",
        "in list format",
        "as a list",
    ]
    user_input_lower = user_input.lower()

    for keyword in list_keywords:
        if keyword in user_input_lower:
            return True

    return False


# === Helper function to check if user is asking for a chart ===
def is_asking_for_chart(user_input):
    """Check if the user is asking for a chart"""
    chart_keywords = [
        "chart",
        "graph",
        "plot",
        "visualize",
        "visualization",
        "bar chart",
        "pie chart",
        "line chart",
    ]
    user_input_lower = user_input.lower()

    for keyword in chart_keywords:
        if keyword in user_input_lower:
            return True

    return False


# === Helper function to extract invoice numbers from user input ===
def extract_invoices_from_input(user_input):
    """Extract invoice numbers from user input"""
    # Pattern to match invoice numbers (INV followed by digits)
    invoice_pattern = re.compile(r"INV\d+")
    return invoice_pattern.findall(user_input)


# === Helper function to extract MRN numbers from user input ===
def extract_mrn_numbers_from_input(user_input):
    """Extract MRN numbers from user input (starting with 24 or 25)"""
    # Pattern to match MRN numbers (starting with 24 or 25, followed by more digits)
    mrn_pattern = re.compile(r"\b(24\d+|25\d+)\b")
    return mrn_pattern.findall(user_input)


# === Routes ===
@app.route("/")
def index():
    current_file = get_current_file()
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT message, response FROM chat_history")
        history = cursor.fetchall()
    return render_template("index.html", history=history, filename=current_file)


@app.route("/view/<content_id>")
def view_content(content_id):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_type, content_data FROM content_links WHERE id = ?",
                (content_id,),
            )
            result = cursor.fetchone()

        if not result:
            return "Content not found", 404

        content_type, content_data = result
        content_data = json.loads(content_data)

        if content_type == "table":
            headers = content_data.get("headers", [])
            rows = content_data.get("rows", [])
            table_html = generate_table_html(headers, rows)

            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Table</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>Data Table</h1>
                {table_html}
            </body>
            </html>
            """

        elif content_type == "chart":
            chart_html = generate_chart_html(content_data)

            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chart</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>Chart</h1>
                {chart_html}
            </body>
            </html>
            """
        else:
            return "Invalid content type", 400

    except Exception as e:
        print(f"[ERROR] Error viewing content: {str(e)}")
        return f"Error: {str(e)}", 500


API_URL = "https://dental.jantrah.com/api/doctor/doctor-list"
TOKEN = "GT4hXeEmyj8HUui9W6BMUBXsF4Waflyw6sklL9k6ScKHK38GXFrb035YFzlUhVqDgSfAbAZgcdHKGjQprWGB9pkN1r"


@app.route("/api/doctors", methods=["GET"])
def get_doctors():
    try:
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        }
        response = requests.get(API_URL, headers=headers)
        data = response.json()

        if data.get("status") and "doctors" in data.get("data", {}):
            # ✅ Only return what frontend needs
            doctors = [
                {
                    "id": doc["doctor_id"],
                    "name": doc["name"],
                    "specialist": doc["specialist"],
                    "designation": doc["designation"],
                }
                for doc in data["data"]["doctors"]
            ]
            return jsonify({"success": True, "doctors": doctors})

        return (
            jsonify({"success": False, "error": data.get("message", "Unknown error")}),
            400,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


MRN_NUMBER = "25081024"


def get_patient_name_from_mrn(mrn_number):
    """Fetch patient_name from DB using MRN number"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT patient_name 
                FROM Patients 
                WHERE mrn_number = ?
            """,
                (mrn_number,),
            )

            row = cursor.fetchone()
            if row:
                return row[0]  # patient_name
            else:
                print(f"⚠️ No patient found for MRN {mrn_number}")
                return None
    except Exception as e:
        print("❌ Error fetching patient from MRN:", e)
        return None


@app.route("/ask", methods=["POST"])
def ask():
    global stop_execution_flag
    stop_execution_flag = False  # reset at the start of request
    start_time = time.time()

    try:
        # ---------------- Parse Input ----------------
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "response": "⚠ No message provided.",
                        "template_url": "",
                    }
                ),
                400,
            )

        user_input = data["message"]
        print(f"[DEBUG] Received request: {user_input}")

        with data_lock:
            df = data_cache

        if df is None:
            print("[DEBUG] No data loaded")
            return jsonify(
                {
                    "status": "error",
                    "response": "⚠ No file uploaded or data loaded. Please upload a CSV first.",
                    "template_url": "",
                }
            )

        # ---------------- Initialize response early ----------------
        response = None
        # ---------------- Session History ----------------
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT message, response FROM chat_history ORDER BY id ASC")
            session_history = cursor.fetchall()

        # ---------------- Intent Detection ----------------
        result = get_chat_response(user_input, df, session_history=session_history)
        intent = result.get("intent") if isinstance(result, dict) else None
        raw_response = result.get("response") if isinstance(result, dict) else result

        # 🔹 Handle Appointment Intents
        if intent == "booking":
            parsed = parse_booking_command(user_input)
            patient_name = get_patient_name_from_mrn(MRN_NUMBER)
            parsed["patient"] = patient_name

            if parsed and all(
                [
                    parsed.get("patient"),
                    parsed.get("doctor"),
                    parsed.get("date"),
                    parsed.get("time"),
                ]
            ):
                booking_result = book_appointment_from_parsed(parsed, MRN_NUMBER)
                result_json = booking_result.get_json()
                if result_json.get("success"):
                    response = result_json.get("message")
                else:
                    response = (
                        f"❌ Could not book appointment: {result_json.get('error')}"
                    )
            else:
                response = None
                return jsonify(
                    {"action": "show_form"}
                )  # form response should not be saved

        elif intent == "cancellation":
            response = (
                "Okay, I'll help you cancel. Can you tell me the appointment details?"
            )

        elif intent == "view_appointment":
            try:
                appointments = fetch_appointments_from_db()
                if not appointments:
                    response = "You have no booked appointments."
                else:
                    response = "<strong>Your booked appointments:</strong><br><ul>"
                    for appt in appointments:
                        response += f"<li>{appt['patient_name']} → {appt['doctor']} on {appt['date']} at {appt['time']} ({appt['status']})</li>"
                    response += "</ul>"
            except Exception as e:
                response = f"Error fetching appointments: {str(e)}"

        elif intent == "greeting":
            response = "Hello! How can I assist you today?"

        # ---------------- Original Chart/Table Handling ----------------
        if stop_execution_flag:
            print("[DEBUG] Execution stopped by flag")
            return jsonify({"status": "stopped", "response": None, "template_url": ""})

        if is_asking_for_chart(user_input):
            invoices = extract_invoices_from_input(user_input)
            if invoices and len(invoices) < 15:
                return jsonify(
                    {
                        "response": "Need at least 15 invoices for a meaningful chart.",
                        "template_url": "",
                    }
                )

            mrn_numbers = extract_mrn_numbers_from_input(user_input)
            if mrn_numbers and len(mrn_numbers) < 15:
                return jsonify(
                    {
                        "response": "Need at least 15 MRN numbers for a meaningful chart.",
                        "template_url": "",
                    }
                )

        print(
            f"[DEBUG] Getting chat response... (Time: {time.time() - start_time:.2f}s)"
        )
        # Only override response if it hasn’t already been set
        if not response:
            response = (
                str(raw_response).strip()
                if raw_response is not None
                else "I'm sorry, I couldn’t generate a response."
            )

        # ---------------- Handle Table/List/Chart ----------------
        # ---------------- Response Finalization ----------------
        final_response = response
        template_url = ""

        if is_asking_for_list(user_input) and "<table" in response:
            final_response = table_to_text_list(response)

        elif ("**" in response or "|" in response) and "<table" not in response:

            headers, rows = [], []

            if "|" in response and "---" in response:
                lines = [line.strip() for line in response.splitlines() if "|" in line]
                headers = [h.strip() for h in lines[0].split("|") if h.strip()]
                for line in lines[2:]:
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    if cells:
                        rows.append(cells)
            else:
                headers = ["Field", "Value"]
                for line in response.splitlines():
                    line = line.strip("* ").strip()
                    if ":" in line:
                        field, value = line.split(":", 1)
                        rows.append([field.strip(), value.strip()])

            fake_html = generate_table_html(headers, rows)
            structured_data = parse_table_to_json(fake_html)

            if structured_data:
                content_id = str(uuid.uuid4())
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO content_links (id, content_type, content_data) VALUES (?, ?, ?)",
                        (content_id, "table", json.dumps(structured_data)),
                    )
                conn.commit()
                template_url = url_for(
                    "view_content", content_id=content_id, _external=True
                )
                final_response = ""

        elif "<table" in response:
            # ... html table case ...
            structured_data = parse_table_to_json(response)
            if structured_data:
                content_id = str(uuid.uuid4())
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO content_links (id, content_type, content_data) VALUES (?, ?, ?)",
                        (content_id, "table", json.dumps(structured_data)),
                    )
                    conn.commit()
                template_url = url_for(
                    "view_content", content_id=content_id, _external=True
                )
                final_response = ""

        elif "CHART_DATA:" in response:
            # ... chart case ...
            chart_str = response.split("CHART_DATA:")[1].strip()
            chart_json = json.loads(chart_str)
            content_id = str(uuid.uuid4())
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO content_links (id, content_type, content_data) VALUES (?, ?, ?)",
                    (content_id, "chart", json.dumps(chart_json)),
                )
                conn.commit()
            template_url = url_for(
                "view_content", content_id=content_id, _external=True
            )
            final_response = ""

        # ---------------- Save History ----------------
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_history (message, response) VALUES (?, ?)",
                    (user_input, final_response if final_response else template_url),
                )
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Error saving to chat history: {str(e)}")

        # ---------------- Return ----------------
        return jsonify({"response": final_response, "template_url": template_url})

    except Exception as e:
        print(f"[ERROR] Unhandled exception in /ask: {str(e)}")
        return jsonify(
            {"status": "error", "response": f"Error: {str(e)}", "template_url": ""}
        )

    finally:
        print(f"[DEBUG] Request completed in {time.time() - start_time:.2f}s")


# ✅ Booking logic with patient name fetched from DB
def book_appointment_from_parsed(parsed, mrn_number):
    try:
        doctor_name = parsed.get("doctor")
        date = parsed.get("date")
        time = parsed.get("time")

        # 🔹 Fetch patient_name from DB using MRN number
        patient = get_patient_name_from_mrn(mrn_number)

        if not patient:
            return (
                jsonify({"success": False, "error": "Patient not found for given MRN"}),
                404,
            )

        if not all([doctor_name, date, time]):
            return jsonify({"success": False, "error": "Missing fields"}), 400

        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()

            # 🔹 Check if slot is already booked
            cursor.execute(
                """
                SELECT 1 FROM appointments
                WHERE doctor = ? AND date = ? AND time = ?
            """,
                (doctor_name, date, time),
            )
            if cursor.fetchone():
                return jsonify({"success": False, "error": "Slot not available"}), 400

            # 🔹 Insert new appointment
            cursor.execute(
                """
                INSERT INTO appointments (patient_name, doctor, date, time, status)
                VALUES (?, ?, ?, ?, ?)
            """,
                (patient, doctor_name, date, time, "booked"),
            )
            conn.commit()

        # ✅ Always jsonify the response
        return jsonify(
            {
                "success": True,
                "message": f"✅ Appointment booked for {patient} with {doctor_name} on {date} at {time}.",
            }
        )

    except Exception as e:
        print("❌ Error booking appointment:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ✅ Keep structured JSON booking for form use
@app.route("/book_appointment/<mrn_number>", methods=["POST"])
def book_appointment(mrn_number):
    try:
        data = request.get_json()
        print("📥 Incoming JSON (form):", data)

        # ✅ Always fetch patient name from DB using MRN
        patient = get_patient_name_from_mrn(mrn_number)
        if not patient:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"No patient found for MRN {mrn_number}",
                    }
                ),
                404,
            )

        # If user sent a raw message, parse it
        if "message" in data:
            parsed = parse_booking_command(data["message"])
            parsed["patient"] = patient
        else:
            parsed = {
                "patient": patient,
                "doctor": data.get("doctor_name"),
                "date": data.get("date"),
                "time": data.get("time"),
            }

        return book_appointment_from_parsed(parsed, mrn_number)

    except Exception as e:
        print("❌ Error booking appointment:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/stop_execution", methods=["POST"])
def stop_execution():
    global stop_execution_flag
    stop_execution_flag = True
    return jsonify({"status": "stopped"})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(save_path)
        set_current_file(filename)
        load_data()
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()
    return redirect(url_for("index"))


@app.route("/delete_file", methods=["POST"])
def delete_file():
    current_file = get_current_file()
    if current_file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], current_file)
        if os.path.exists(file_path):
            os.remove(file_path)
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM current_file")
            cursor.execute("DELETE FROM chat_history")
            conn.commit()
        global data_cache
        with data_lock:
            data_cache = None
    return redirect(url_for("index"))


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM chat_history")
        conn.commit()
    return jsonify({"status": "cleared"})


# === Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
