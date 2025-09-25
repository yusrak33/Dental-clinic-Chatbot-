import pandas as pd
import google.generativeai as genai
import re
from langdetect import detect, DetectorFactory
import logging
import io
import numpy as np
import json
from analytics import get_statistics, find_trends, get_correlation, detect_anomalies, predict
  # ✅ Added import for analytics.py
from analytics import polish_with_gemini
from analytics import resolve_column
import importlib
import analytics
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from viz_utils import detect_visualization_request, generate_chart_data
from business_strategies import handle_business_strategy_query, is_business_strategy_query


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # to make language detection consistent

# 🔑 Gemini API key
GEMINI_API_KEY = "AIzaSyDhSrwZaIdEM2WVIELNAu7qIa-WRfbsqn4"
genai.configure(api_key=GEMINI_API_KEY)

# 📦 Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")


# Load your dataset and precompute statistics
try:
    df = pd.read_csv('patient_details2.csv')
    logger.info(f"Dataset loaded with {len(df)} rows")
    
    # Pre-compute dataset statistics
    dataset_stats = {
        'total_rows': len(df),
        'unique_doctors': df['doctor_name'].nunique() if 'doctor_name' in df.columns else 0,
        'doctor_names': df['doctor_name'].unique().tolist() if 'doctor_name' in df.columns else [],
        'unique_invoices': df['invoice_number'].nunique() if 'invoice_number' in df.columns else 0,
        'unique_patients': df['patient_name'].nunique() if 'patient_name' in df.columns else 0,
        'unique_mrn': df['mrn_number'].nunique() if 'mrn_number' in df.columns else 0,
        'total_price': df['price'].sum() if 'price' in df.columns else 0
    }
    
    logger.info(f"Dataset statistics: {dataset_stats}")
        
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    df = pd.DataFrame()
    dataset_stats = {}

# Function to extract entities from query
def extract_entities(query):
    entities = {
        'invoice_numbers': [],
        'mrn_numbers': [],
        'doctor_names': [],
        'patient_names': []
    }
    
    # Extract invoice numbers (INV followed by digits)
    invoice_pattern = r'INV\d+'
    entities['invoice_numbers'] = re.findall(invoice_pattern, query, re.IGNORECASE)
    
    # Extract MRN numbers (6 or 7 digits)
    mrn_pattern = r'\b\d{6,7}\b'
    entities['mrn_numbers'] = re.findall(mrn_pattern, query)
    
    # Extract doctor names (Dr. followed by name)
    doctor_pattern = r'(?:Dr\.?|Doctor)\s+([A-Za-z\s]+)'
    doctor_matches = re.findall(doctor_pattern, query, re.IGNORECASE)
    entities['doctor_names'] = [name.strip() for name in doctor_matches]
    
    # Extract patient names (capitalized words that might be names)
    if not entities['invoice_numbers'] and not entities['mrn_numbers'] and not entities['doctor_names']:
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, query)
        # Filter out common words that aren't names
        common_words = ['Table', 'The', 'And', 'But', 'For', 'How', 'What', 'When', 'Where', 'Why', 'Who', 'Which', 'Show', 'Me', 'In', 'Form']
        filtered_names = [name for name in potential_names if name not in common_words]
        if filtered_names:
            entities['patient_names'] = filtered_names
    
    return entities

# Function to get relevant rows based on query
def get_relevant_rows(query, df):
    entities = extract_entities(query)
    logger.info(f"Extracted entities: {entities}")
    
    # Start with an empty dataframe
    relevant_rows = pd.DataFrame()
    
    # If invoice numbers found, filter for those invoices
    if entities['invoice_numbers'] and 'invoice_number' in df.columns:
        invoice_filter = df['invoice_number'].isin(entities['invoice_numbers'])
        invoice_rows = df[invoice_filter]
        relevant_rows = pd.concat([relevant_rows, invoice_rows])
        logger.info(f"Found {len(invoice_rows)} rows for invoices: {entities['invoice_numbers']}")
    
    # If MRN numbers found, filter for those MRNs
    if entities['mrn_numbers'] and 'mrn_number' in df.columns:
        mrn_filter = df['mrn_number'].astype(str).isin(entities['mrn_numbers'])
        mrn_rows = df[mrn_filter]
        relevant_rows = pd.concat([relevant_rows, mrn_rows])
        logger.info(f"Found {len(mrn_rows)} rows for MRNs: {entities['mrn_numbers']}")
    
    # If doctor names found, filter for those doctors
    if entities['doctor_names'] and 'doctor_name' in df.columns:
        doctor_pattern = '|'.join(entities['doctor_names'])
        doctor_filter = df['doctor_name'].str.contains(doctor_pattern, case=False, na=False)
        doctor_rows = df[doctor_filter]
        relevant_rows = pd.concat([relevant_rows, doctor_rows])
        logger.info(f"Found {len(doctor_rows)} rows for doctors: {entities['doctor_names']}")
    
    # If patient names found, filter for those patients
    if entities['patient_names'] and 'patient_name' in df.columns:
        patient_pattern = '|'.join(entities['patient_names'])
        patient_filter = df['patient_name'].str.contains(patient_pattern, case=False, na=False)
        patient_rows = df[patient_filter]
        relevant_rows = pd.concat([relevant_rows, patient_rows])
        logger.info(f"Found {len(patient_rows)} rows for patients: {entities['patient_names']}")
    
    # If we found specific rows, return them
    if len(relevant_rows) > 0:
        relevant_rows = relevant_rows.drop_duplicates()
        logger.info(f"Returning {len(relevant_rows)} specific rows")
        return relevant_rows
    
    # For general queries, return a larger random sample
    sample_size = min(300, len(df))  # Keep within token limits
    random_sample = df.sample(sample_size)
    logger.info(f"Returning random sample of {sample_size} rows")
    return random_sample

# Function to handle general queries using pre-computed statistics
def handle_general_query(query):
    query_lower = query.lower()
    
    # Check for greetings
    if query_lower in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']:
        return "Hello! I'm your dental clinic data assistant. How can I help you today?"
    
    # Check for acknowledgments
    if query_lower in ['ok', 'okay', 'thanks', 'thank you', 'alright']:
        return "You're welcome! Is there anything else you'd like to know about the dental clinic data?"
    
    # Check for doctor count queries
    if any(phrase in query_lower for phrase in ['how many doctors', 'doctor count', 'number of doctors', 'doctors available']):
        if dataset_stats.get('unique_doctors', 0) > 0:
            return f"There are {dataset_stats['unique_doctors']} doctors available: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for doctor verification queries
    if 'doctor' in query_lower and any(phrase in query_lower for phrase in ['check the', 'verify the', 'is the', 'are the']):
        # Extract any number in the query
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            count = int(numbers[0])
            actual_count = dataset_stats.get('unique_doctors', 0)
            if count == actual_count:
                return f"Yes, that's correct! We have {actual_count} doctors: {', '.join(dataset_stats['doctor_names'])}."
            else:
                return f"No, that's not correct. We actually have {actual_count} doctors: {', '.join(dataset_stats['doctor_names'])}."
        else:
            if dataset_stats.get('unique_doctors', 0) > 0:
                return f"We have {dataset_stats['unique_doctors']} doctors: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for total rows query
    if any(phrase in query_lower for phrase in ['total rows', 'how many rows', 'number of rows']):
        if dataset_stats.get('total_rows', 0) > 0:
            return f"The dataset contains {dataset_stats['total_rows']} rows in total."
    
    # Check for total price queries
    if any(phrase in query_lower for phrase in ['total count price', 'total price', 'sum of prices']):
        if dataset_stats.get('total_price', 0) > 0:
            return f"The total price of all treatments in the dataset is {dataset_stats['total_price']}."
    
    # Check for doctor list query
    if any(phrase in query_lower for phrase in ['list of doctors', 'doctor names', 'show doctors']):
        if dataset_stats.get('doctor_names', []):
            return f"The available doctors are: {', '.join(dataset_stats['doctor_names'])}."
    
    # Check for help queries
    if any(phrase in query_lower for phrase in ['help', 'what can you do', 'how to use']):
        return "I can help you with information about patients, doctors, invoices, and treatments in the dental clinic dataset. You can ask about specific patients, doctors, invoice numbers, or general statistics about the clinic."
    
    # Check for "how you check" type queries
    if any(phrase in query_lower for phrase in ['how you check', 'how do you know', 'how did you find']):
        return "I look through all the patient records in our system and count the unique doctor names to get the accurate count."
    
    # If no general query pattern matches, return None
    return None

# ✅ Urdu/Roman Urdu detection
def is_urdu(text):
    try:
        lang = detect(text)
    except:
        lang = ""
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    has_urdu_script = len(urdu_chars) > 5
    is_probably_roman_urdu = lang in ["ur", "hi", "fa"]
    return has_urdu_script or is_probably_roman_urdu

# ---------- Formatting Functions ----------
def format_response_table(response_text: str):
    """
    Converts markdown or tab-separated table to styled HTML table.
    Removes markdown separator lines with only dashes.
    """
    table_lines = []
    in_table = False
    for line in response_text.splitlines():
        if "|" in line or "\t" in line:
            if re.match(r"^\s*[-\s|]+\s*$", line):  # Skip separator row
                continue
            table_lines.append(line.strip())
            in_table = True
        elif in_table and line.strip() == "":
            break  # End of table
    if table_lines:
        # Detect separator type
        sep = "|" if "|" in table_lines[0] else "\t"
        # Normalize rows
        normalized_lines = []
        for line in table_lines:
            if sep == "|":
                cells = [cell.strip() for cell in line.strip('|').split('|')]
            else:
                cells = [cell.strip() for cell in line.split('\t')]
            normalized_lines.append('\t'.join(cells))  # Normalize to tab
        fixed_table = "\n".join(normalized_lines)
        try:
            df = pd.read_csv(io.StringIO(fixed_table), sep="\t")
            df = df.dropna(how='all')  # Drop empty rows
            df.columns = [col.strip() for col in df.columns]
            # Generate HTML table with improved styling
            html_table = '''
<style>
.table-container {
    max-width: 100%;
    overflow-x: auto;
    margin: 20px 0;
    font-family: Arial, sans-serif;
}
.solid-table {
    border-collapse: collapse;
    width: 100%;
    background-color: #fff;
    box-shadow: 0 2px 3px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
}
.solid-table th, .solid-table td {
    border: 1px solid #e0e0e0;
    padding: 12px 15px;
    text-align: left;
}
.solid-table th {
    background-color: #f8f9fa;
    color: #333;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 14px;
}
.solid-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.solid-table tr:hover {
    background-color: #f1f1f1;
}
.solid-table tr.dash-row {
    display: none; /* hide dashed rows if any */
}
</style>
<div class="table-container">
<table class="solid-table">
<thead><tr>
'''
            # Headers
            for col in df.columns:
                html_table += f"<th>{col}</th>"
            html_table += "</tr></thead><tbody>"
            # Rows
            for _, row in df.iterrows():
                row_values = list(row)
                if all(re.match(r"^-+$", str(cell).strip()) for cell in row_values):
                    html_table += '<tr class="dash-row">'
                else:
                    html_table += '<tr>'
                for cell in row_values:
                    html_table += f"<td>{cell}</td>"
                html_table += "</tr>"
            html_table += "</tbody></table></div>"
            return html_table
        except Exception as e:
            logger.error(f"Error parsing table: {str(e)}")
            return "<pre>" + fixed_table + "</pre>"
    return response_text.strip() if response_text else "I couldn't format the table."


def format_response_list(response_text: str) -> str:
    logger.info(f"Formatting response as list: {response_text[:100]}...")
    if not response_text or response_text.strip() == "":
        return "I'm sorry, I couldn't generate a response. Please try again."
    # Remove markdown code blocks
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    # REMOVE BOLD MARKDOWN (**text**) globally!
    response_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    # Proceed as before
    records = re.split(r'(?=Patient:|MRN:)', response_text)
    formatted_records = []
    for record in records:
        if not record.strip():
            continue
        record = record.strip()
        record = re.sub(r'^-+\s*', '', record)
        # This regex may be unnecessary for your city list, so just add as bullet
        formatted_records.append(f"- {record}")
    result = '\n'.join(formatted_records).strip()
    return result if result else response_text

def format_response_paragraph(response_text: str) -> str:
    logger.info(f"Formatting response as paragraph: {response_text[:100]}...")
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\*\*(.*?)\*\*', r'\1', response_text)
    return response_text.replace("\n", " ").strip()



# ---------- AI-Powered Visualization Detection Functions ----------
def detect_visualization_request(user_message):
    """
    Uses AI to intelligently detect and interpret visualization requests.
    Returns a dict with visualization type and parameters, or None if no visualization requested.
    """
    user_message_lower = user_message.lower()
    
    # Keywords that indicate visualization requests
    chart_keywords = ['chart', 'graph', 'plot', 'visualization', 'visualize', 'show me a', 'display', 'draw', 'histogram', 'distribution']
    
    # Check if it's a visualization request
    is_viz_request = any(keyword in user_message_lower for keyword in chart_keywords)
    
    if not is_viz_request:
        return None
    
    # Use AI to interpret the visualization request
    return interpret_chart_request_with_ai(user_message)

def interpret_chart_request_with_ai(user_message):
    """
    Uses Gemini AI to intelligently interpret chart requests and determine
    the best visualization type, data grouping, and chart configuration.
    """
    try:
        # Available columns in the dataset
        available_columns = ['mrn_number', 'patient_name', 'Registration date', 'city', 'invoice_number', 'Invoice date', 'description', 'price', 'doctor_name']
        
        prompt = f"""
You are a data visualization expert. Analyze this user request for a chart/graph and determine the best way to visualize the data.

USER REQUEST: "{user_message}"

AVAILABLE DATA COLUMNS:
- mrn_number: Patient medical record numbers
- patient_name: Patient names  
- Registration date: When patients registered (YYYY-MM-DD format)
- city: Patient locations/cities
- invoice_number: Invoice IDs (format: INV followed by numbers)
- Invoice date: When services were provided (YYYY-MM-DD format)
- description: Treatment/service descriptions (e.g., "Root canal treatment", "Consultation Fee", "Scaling and polishing")
- price: Cost of services (numeric)
- doctor_name: Doctor names (e.g., "Dr Saqib", "Dr Israr")

RESPOND WITH A JSON OBJECT ONLY (no other text):
{{
    "chart_type": "bar|pie|line|doughnut",
    "group_by_column": "column_name_to_group_data_by",
    "aggregate_column": "column_to_count_or_sum",
    "aggregate_function": "count|sum|average",
    "filter_conditions": {{"column": "filter_value"}} or null,
    "time_period_days": number or null,
    "title": "Descriptive chart title",
    "limit_results": number or null,
    "sort_order": "desc|asc"
}}

EXAMPLES:
- "Show me a chart of patients by city" → group_by_column: "city", aggregate_function: "count"
- "Revenue by doctor" → group_by_column: "doctor_name", aggregate_column: "price", aggregate_function: "sum"  
- "Most common treatments" → group_by_column: "description", aggregate_function: "count"
- "Patients registered last month" → group_by_column: "Registration date", time_period_days: 30
- "Dr Saqib's treatments pie chart" → group_by_column: "description", filter_conditions: {{"doctor_name": "Dr Saqib"}}, chart_type: "pie"
"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response to extract JSON
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].strip()
        
        # Parse the AI response
        try:
            ai_config = json.loads(response_text)
            logger.info(f"AI interpreted chart request: {ai_config}")
            return ai_config
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {response_text}")
            # Fallback to basic interpretation
            return get_fallback_chart_config(user_message)
            
    except Exception as e:
        logger.error(f"Error in AI chart interpretation: {str(e)}")
        return get_fallback_chart_config(user_message)

def get_fallback_chart_config(user_message):
    """
    Fallback chart configuration when AI interpretation fails.
    """
    user_message_lower = user_message.lower()
    
    # Default configuration
    config = {
        "chart_type": "bar",
        "group_by_column": "description", 
        "aggregate_column": None,
        "aggregate_function": "count",
        "filter_conditions": None,
        "time_period_days": None,
        "title": "Data Overview",
        "limit_results": 10,
        "sort_order": "desc"
    }
    
    # Simple keyword-based fallbacks
    if any(word in user_message_lower for word in ['city', 'location', 'where']):
        config["group_by_column"] = "city"
        config["title"] = "Patients by City"
    elif any(word in user_message_lower for word in ['doctor', 'dr']):
        config["group_by_column"] = "doctor_name" 
        config["title"] = "Patients by Doctor"
    elif any(word in user_message_lower for word in ['treatment', 'service', 'procedure']):
        config["group_by_column"] = "description"
        config["title"] = "Most Common Treatments"
    elif any(word in user_message_lower for word in ['revenue', 'price', 'money', 'cost']):
        config["group_by_column"] = "doctor_name"
        config["aggregate_column"] = "price"
        config["aggregate_function"] = "sum"
        config["title"] = "Revenue by Doctor"
    elif any(word in user_message_lower for word in ['pie']):
        config["chart_type"] = "pie"
    elif any(word in user_message_lower for word in ['line']):
        config["chart_type"] = "line"
        
    return config

def generate_chart_data(df, viz_params):
    """
    AI-powered chart data generation that can handle any type of visualization request.
    Returns dict with labels, values, and metadata for frontend chart rendering.
    """
    try:
        logger.info(f"Generating chart with params: {viz_params}")
        
        # Extract parameters from AI configuration
        chart_type = viz_params.get('chart_type', 'bar')
        group_by_column = viz_params.get('group_by_column')
        aggregate_column = viz_params.get('aggregate_column')
        aggregate_function = viz_params.get('aggregate_function', 'count')
        filter_conditions = viz_params.get('filter_conditions')
        time_period_days = viz_params.get('time_period_days')
        title = viz_params.get('title', 'Data Visualization')
        limit_results = viz_params.get('limit_results', 10)
        sort_order = viz_params.get('sort_order', 'desc')
        
        # Validate required parameters
        if not group_by_column or group_by_column not in df.columns:
            return {'error': f'Column "{group_by_column}" not found in dataset. Available columns: {list(df.columns)}'}
        
        # Start with full dataset
        df_filtered = df.copy()
        
        # Apply time period filter if specified
        if time_period_days and group_by_column in ['Invoice date', 'Registration date']:
            try:
                df_filtered[group_by_column] = pd.to_datetime(df_filtered[group_by_column], errors='coerce')
                df_filtered = df_filtered.dropna(subset=[group_by_column])
                
                start_date = datetime.now() - timedelta(days=time_period_days)
                df_filtered = df_filtered[df_filtered[group_by_column] >= start_date]
                
                if df_filtered.empty:
                    return {'error': f'No records found in the last {time_period_days} days'}
                    
                title += f' (Last {time_period_days} Days)'
            except Exception as e:
                logger.warning(f"Could not apply time filter: {str(e)}")
        
        # Apply additional filter conditions if specified
        if filter_conditions:
            try:
                for col, value in filter_conditions.items():
                    if col in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(str(value), case=False, na=False)]
                        title += f' - {value}'
            except Exception as e:
                logger.warning(f"Could not apply filter conditions: {str(e)}")
        
        # Handle different aggregation types
        if aggregate_function == 'count':
            # Count occurrences by group
            if group_by_column in ['Invoice date', 'Registration date'] and time_period_days:
                # For time-based charts, group by date
                grouped_data = df_filtered.groupby(df_filtered[group_by_column].dt.date).size().reset_index(name='count')
                grouped_data.columns = [group_by_column, 'count']
                grouped_data[group_by_column] = grouped_data[group_by_column].astype(str)
            else:
                # Regular grouping
                grouped_data = df_filtered[group_by_column].value_counts().reset_index()
                grouped_data.columns = [group_by_column, 'count']
            
            values_column = 'count'
            
        elif aggregate_function == 'sum' and aggregate_column:
            # Sum values by group
            if aggregate_column not in df_filtered.columns:
                return {'error': f'Aggregate column "{aggregate_column}" not found in dataset'}
            
            # Convert to numeric, replacing non-numeric values with 0
            df_filtered[aggregate_column] = pd.to_numeric(df_filtered[aggregate_column], errors='coerce').fillna(0)
            
            grouped_data = df_filtered.groupby(group_by_column)[aggregate_column].sum().reset_index()
            values_column = aggregate_column
            
        elif aggregate_function == 'average' and aggregate_column:
            # Average values by group  
            if aggregate_column not in df_filtered.columns:
                return {'error': f'Aggregate column "{aggregate_column}" not found in dataset'}
                
            df_filtered[aggregate_column] = pd.to_numeric(df_filtered[aggregate_column], errors='coerce').fillna(0)
            
            grouped_data = df_filtered.groupby(group_by_column)[aggregate_column].mean().reset_index()
            values_column = aggregate_column
            
        else:
            return {'error': f'Unsupported aggregate function: {aggregate_function}'}
        
        # Sort results
        ascending = sort_order == 'asc'
        grouped_data = grouped_data.sort_values(values_column, ascending=ascending)
        
        # Limit results if specified
        if limit_results and len(grouped_data) > limit_results:
            grouped_data = grouped_data.head(limit_results)
        
        # Check if we have data
        if grouped_data.empty:
            return {'error': 'No data found matching the criteria'}
        
        # Prepare chart data
        labels = grouped_data[group_by_column].astype(str).tolist()
        values = grouped_data[values_column].tolist()
        
        # Generate metadata
        total_records = len(df_filtered)
        metadata = {}
        
        if aggregate_function == 'count':
            metadata['total_records'] = total_records
        elif aggregate_function == 'sum':
            metadata['total_sum'] = sum(values)
        elif aggregate_function == 'average':
            metadata['overall_average'] = np.mean(values) if values else 0
        
        return {
            'labels': labels,
            'values': values,
            'title': title,
            'chart_type': chart_type,
            'metadata': metadata,
            'total_records': total_records
        }
        
    except Exception as e:
        logger.error(f"Error generating chart data: {str(e)}")
        return {'error': f'Error generating chart: {str(e)}'}

from Appointment.Intent_appoint import detect_intent
    # ---------- Main Chat Function ----------
def get_chat_response(user_message, df, session_history=None, answer_format='auto'):
    """
    Handles user queries:
    - Booking, cancellation, view appointments
    - Returns first/last patient names
    - Routes to analytics functions (stats, trends, anomalies, correlations, forecasts)
    - Supports humanized outputs
    - Handles visualization requests
    - Falls back to Gemini for general conversation
    """
    try:
        intent = detect_intent(user_message)
        msg_lower = user_message.lower()
        logger.debug(f"[INTENT] User: {user_message} | Detected: {intent}")
        print(f"[DEBUG] User: {user_message}, Intent: {intent}")

        # ---------- Appointment Intents ----------
        if intent == "booking":
            logger.debug("[INTENT] Booking detected, returning dict...")
            return {"intent": "booking", "action": "show_form",
                    "response": "Okay, let's get your appointment details."}

        elif intent == "cancellation":
            logger.debug("[INTENT] Cancellation detected...")
            return {"intent": "cancellation",
                    "response": "Okay, I'll help you cancel. Can you tell me the appointment details?"}

        elif intent == "view_appointment":
            logger.debug("[INTENT] View appointment detected...")
            return {"intent": "view_appointment", "response": "Fetching your appointments..."}

        # ---------- First/Last Patient Rules ----------
        if df is not None and not df.empty:
            if "first 3 patients" in msg_lower:
                logger.debug("[RULE] First 3 patients requested")
                if "patient_name" in df.columns:
                    patients = df["patient_name"].dropna().unique()[:3]
                    resp = "- Here are the first 3 patients:\n" + "\n".join([f"* {p}" for p in patients])
                    logger.debug(f"[RETURN] {resp}")
                    return resp
                return "No patient records found."

            if "first patient" in msg_lower:
                logger.debug("[RULE] First patient requested")
                if "patient_name" in df.columns:
                    patient = df["patient_name"].dropna().unique()[0]
                    return f"- The first patient's name is {patient}."
                return "No patient records found."

            if "last patient" in msg_lower:
                logger.debug("[RULE] Last patient requested")
                if "patient_name" in df.columns:
                    patient = df["patient_name"].dropna().unique()[-1]
                    return f"- The last patient's name is {patient}."
                return "No patient records found."

        # ---------- Analytics Routing ----------
        analytics_keywords = [
            "statistics", "summary", "describe", "trend",
            "correlation", "anomalies", "forecast", "prediction",
            "daily", "weekly", "monthly", "yearly", "predict"
        ]
        if any(word in msg_lower for word in analytics_keywords):
            logger.debug(f"[ANALYTICS] Keyword matched in: {user_message}")
            if df is None or df.empty:
                return "I couldn’t find any dataset loaded for analysis."
            try:
                from analytics import (
                    get_statistics,
                    find_trends,
                    get_correlation,
                    detect_anomalies,
                    predict
                )
                # each branch below should log before return
                if "summary" in msg_lower or "statistics" in msg_lower or "describe" in msg_lower:
                    logger.debug("[ANALYTICS] summary/statistics/describe triggered")
                    result = get_statistics(df, user_message, model=model, humanize=True)
                    out = result["explanation"] if isinstance(result, dict) else result
                    logger.debug(f"[RETURN] {out[:100]}...")
                    return out

                elif "trend" in msg_lower:
                    logger.debug("[ANALYTICS] trend triggered")
                    freq = "D"
                    if "weekly" in msg_lower: freq = "W"
                    elif "monthly" in msg_lower: freq = "M"
                    elif "yearly" in msg_lower: freq = "Y"

                    matched_col = resolve_column(user_message, df)
                    if matched_col:
                        return find_trends(df, column=matched_col, freq=freq, humanize=True)
                    else:
                        value_col = next((c for c in df.columns if c.lower() in ["price", "amount", "quantity"]), None)
                        date_col = next((c for c in df.columns if "date" in c.lower()), None)
                        if not date_col or not value_col:
                            return "Cannot determine column for trend analysis."
                        return find_trends(df, user_message, freq=freq, model=model, humanize=True)

                elif "correlation" in msg_lower:
                    logger.debug("[ANALYTICS] correlation triggered")
                    cols = resolve_column(user_message, df)
                    if not cols:
                        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                        if len(numeric_cols) >= 2:
                            col1, col2 = numeric_cols[:2]
                        else:
                            return "Need at least 2 numeric columns to compute correlation."
                    else:
                        col1, col2 = cols
                    return get_correlation(df, col1, col2, humanize=True)

                elif "anomalies" in msg_lower or "detect anomalies" in msg_lower:
                    logger.debug("[ANALYTICS] anomalies triggered")
                    matched_col = resolve_column(user_message, df)
                    if not matched_col:
                        matched_col = "price" if "price" in df.columns else next(
                            iter(df.select_dtypes(include=np.number).columns), None
                        )
                    if not matched_col:
                        return "No numeric column found for anomaly detection."
                    return detect_anomalies(df, matched_col, model=model, humanize=True)

                elif "forecast" in msg_lower or "predict" in msg_lower:
                    logger.debug("[ANALYTICS] forecast/predict triggered")
                    return predict(df, user_message, resolve_column)

                else:
                    logger.warning("[ANALYTICS] keyword detected but no matching function")
                    return "Analytics keyword detected but no matching function found."

            except Exception as e:
                logger.error(f"[ANALYTICS ERROR] {e}")
                return f"Sorry, analytics failed: {e}"

        # ---------- Visualization ----------
        viz_params = detect_visualization_request(user_message)
        if viz_params:
            logger.debug(f"[VIZ] Detected visualization request: {viz_params}")
            chart_data = generate_chart_data(df, viz_params)
            if 'error' in chart_data:
                return f"I couldn't generate the chart: {chart_data['error']}"
            chart_json = json.dumps(chart_data)
            return f"CHART_DATA:{chart_json}"

        # ---------- General Queries ----------
        general_response = handle_general_query(user_message)
        if general_response:
            logger.debug(f"[GENERAL] Response: {general_response[:100]}...")
            return general_response

        # ---------- Fallback to Gemini ----------
        logger.debug("[FALLBACK] Building Gemini prompt...")
        df_sample = get_relevant_rows(user_message, df)

        if df_sample is None or not isinstance(df_sample, pd.DataFrame) or df_sample.empty:
            df_sample = df.head(5) if df is not None and not df.empty else pd.DataFrame()
        else:
            df_sample = df_sample.dropna(how='all', axis=0).dropna(how='all', axis=1)

        columns = df_sample.columns.tolist()
        data_preview = df_sample.to_dict(orient='records')

        urdu_requested = is_urdu(user_message)
        language_instruction = "جواب صرف اردو میں دیں۔ انگریزی استعمال نہ کریں۔\n\n" if urdu_requested else ""

        history_text = ""
        if session_history:
            history_text = "\n\nRECENT CHAT HISTORY:\n"
                
            for user_msg, bot_resp in session_history[-5:]:
                user_msg = (user_msg[:200] + "...") if user_msg and len(user_msg) > 200 else (user_msg or "")
                bot_resp = (bot_resp[:200] + "...") if bot_resp and len(bot_resp) > 200 else (bot_resp or "")
                history_text += f"User: {user_msg}\nBot: {bot_resp}\n\n"


        # Dataset statistics for Gemini prompt
        stats_text = f"""
Here are some key statistics about the dental clinic:
- Total patients: {dataset_stats['unique_patients']}
- Total doctors: {dataset_stats['unique_doctors']}
- Doctor names: {', '.join(dataset_stats['doctor_names'])}
- Total invoices: {dataset_stats['unique_invoices']}
- Total price: {dataset_stats['total_price']}
"""

        prompt = f"""
You are a friendly dental clinic assistant. You help with patient records, appointments, invoices, and treatments. Keep your answers short, friendly, and conversational.

{stats_text}

Columns in dataset: {columns}

Relevant records:
{data_preview}

{history_text}

Please answer this question: "{user_message}"

{language_instruction}

Remember to:
- Keep it short and friendly
- Use tables when showing data
- Be accurate with numbers (use the statistics provided above for totals)
- Don't say "based on the dataset" or mention the dataset
- If you don't know something, just say so politely
"""

        logger.info(f"[GEMINI PROMPT] {prompt[:300]}...")
        response = model.generate_content(prompt)
        logger.info(f"[GEMINI RESPONSE RAW] {response}")
        logger.info(f"[GEMINI RESPONSE TEXT] {getattr(response, 'text', None)}")

        if not response or not hasattr(response, "text"):
            logger.error("[GEMINI ERROR] No text in response object!")
            return "Sorry, I didn’t get any response from Gemini."

        resp_text = response.text.strip()
        logger.debug(f"[RETURN] Final Gemini response: {resp_text[:200]}...")
        return resp_text

    except Exception as e:
        logger.error(f"[FATAL] Error in get_chat_response: {str(e)}", exc_info=True)
        print(f"[FATAL DEBUG] Error in get_chat_response: {str(e)}")
        return f"Error generating response: {str(e)}"
