# analytics.py

import io
import re
import numpy as np
import pandas as pd
from prophet import Prophet
from scipy import stats
import google.generativeai as genai



COLUMN_ALIASES = {
    "price": ["price", "revenue", "charges", "cost", "bill", "amount", "fees"],
    "doctor_name": ["doctor", "doctors", "physician", "consultant", "doctor_name"],
    "patient_name": ["patient", "patients", "name", "patient_name"],
    "city": ["city", "location", "place", "area"],
    "invoice_number": ["invoice", "invoice_number", "bill_number"],
    "invoice date": ["invoice date", "date of invoice", "billing date"],
    "registration date": ["registration date", "reg date", "enrollment date"],
    "description": ["treatments", "traetments","services"]
}

def resolve_column(user_query: str, df: pd.DataFrame):
    """Map user query keywords to actual dataset columns"""
    q = user_query.lower()
    for col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in q:
                for real_col in df.columns:
                    if real_col.lower() == col.lower():
                        return real_col
    return None


def gemini_generate(prompt):
    """
    Pass a text prompt to Gemini and return the model's response.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error: {str(e)}]"

def polish_with_gemini(raw_text, model=None):
    """
    Polishes both numeric values and surrounding context for readability,
    but keeps all numbers unchanged.
    """
    prompt = (
        "Polish this text to make it clear, readable, and natural. "
        "Keep all numeric values exactly as they are. "
        "You can improve context and wording, but do not change any numbers.\n\n"
        f"{raw_text}"
    )
    response = gemini_generate(prompt)  # your existing Gemini call
    return response


# ---------------- Gemini helper ----------------
#def polish_with_gemini(raw_text, prompt="Make this explanation more human-friendly and conversational."):
    #try:
        # import Gemini call from chatbot model
        #from chatbot_model import gemini_generate  
        #return gemini_generate(f"{prompt}\n\n{raw_text}")
    #except Exception as e:
        #return raw_text  # fallback if Gemini fails


# ------------- helpers for safe formatting -------------

def _num(x):
    """Pretty number formatting. No NaN, no scientific unless huge."""
    if pd.isna(x):
        return None
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf) >= 1e6 or abs(xf) < 1e-2:
        return f"{xf:.2e}"
    if xf.is_integer():
        return f"{int(xf):,}"
    return f"{xf:,.2f}"


def _date(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def format_table(rows, headers):
    buf = io.StringIO()
    df = pd.DataFrame(rows, columns=headers)
    buf.write(df.to_string(index=False))
    return buf.getvalue()


def format_paragraph(lines):
    return "\n".join(lines)


# ----------------- analytics functions -----------------

def get_statistics(df, user_message, model=None, humanize=False):
    """
    Generate context-aware statistics for a dataset or specific column.
    Automatically decides what to summarize based on user query.
    """
    try:
        msg = user_message.lower()
        desc = None

        # --- Decide target column ---
        if "doctor" in msg:
            target_col = "doctor_name"
        elif "patient" in msg:
            target_col = "patient_name"
        elif "treatment" in msg or "description" in msg:
            target_col = "description"
        elif "price" in msg or "revenue" in msg or "amount" in msg:
            target_col = "price"
        else:
            target_col = None  # fallback to full dataset

        # --- Column-specific statistics ---
        if target_col and target_col in df.columns:
            series = df[target_col]

            if series.dtype == "object":  # categorical
                desc = {
                    "count": int(series.count()),
                    "unique": int(series.nunique()),
                    "top": str(series.mode().iloc[0]) if not series.mode().empty else None,
                    "freq": int(series.value_counts().iloc[0]) if not series.value_counts().empty else None,
                }
                explanation = (
                    f"The dataset contains {desc['count']} entries for **{target_col}**, "
                    f"representing {desc['unique']} unique values. "
                )
                if desc["top"]:
                    explanation += (
                        f"The most frequent is **{desc['top']}** "
                        f"with {desc['freq']} occurrences."
                    )
            else:  # numeric
                desc = series.describe().to_dict()
                explanation = f"Here are the statistics for **{target_col}**:\n{desc}"

        else:
            # --- Full dataset summary ---
            desc = {
                "rows": len(df),
                "columns": list(df.columns),
                "missing_values": df.isnull().sum().sum()
            }
            explanation = (
                f"The dataset contains {desc['rows']} entries "
                f"across {len(desc['columns'])} columns. "
                f"There are {desc['missing_values']} missing values in total."
            )

        # --- Polish with Gemini ---
        if humanize and model:
            explanation = polish_with_gemini(explanation, model)

        return {"explanation": explanation, "raw": desc}

    except Exception as e:
        return f"Sorry, analytics failed in get_statistics: {e}"




def detect_anomalies(df, column, z_thresh=3, context_col=None, humanize=True):
    """
    Detect anomalies in a numeric column using z-scores.

    Parameters:
    - df: pandas DataFrame
    - column: numeric column to check for anomalies
    - z_thresh: z-score threshold for flagging anomalies (default 3)
    - context_col: optional column (like 'description') to show alongside anomalies
    - humanize: if True, return natural language summary, else raw table
    """
    try:
        if column not in df:
            return f"Column '{column}' not found in data."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Anomaly detection only works for numeric columns, but '{column}' is not numeric."

        # drop NaN values safely
        valid_data = df[[column]].dropna()
        if valid_data.empty:
            return f"No valid data in '{column}' for anomaly detection."

        # calculate z-scores
        zs = np.abs(stats.zscore(valid_data[column]))
        anomalies = valid_data.loc[zs > z_thresh].copy()

        # attach context if available
        if context_col and context_col in df:
            anomalies[context_col] = df.loc[anomalies.index, context_col]

        if anomalies.empty:
            result = "No significant anomalies found."
        else:
            if context_col and context_col in anomalies:
                lines = [
                    f"{row[context_col]} → {_num(row[column])}"
                    for _, row in anomalies.iterrows()
                ]
            else:
                lines = [f"Index {i}: {_num(v)}" for i, v in anomalies[column].items()]
            result = "Anomalies detected:\n" + "\n".join(lines)

        return polish_with_gemini(result) if humanize else result

    except Exception as e:
        return f"Anomaly detection failed: {str(e)}"
   
def get_correlation(df, col1, col2, model=None, humanize=True):
    if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
        return "Correlation requires numeric columns."
    corr = df[col1].corr(df[col2])
    result = f"Correlation between {col1} and {col2}: {corr:.2f}"
    if humanize:
        return polish_with_gemini(result)
    return result



def find_trends(df, UserMessage, model=None, column=None, freq="W", humanize=True):
    """
    Find trends in the data over a specified frequency (weekly, monthly, etc.).
    If no column is mentioned in the user message, default to 'price'.
    """

    # make sure invoice date is datetime
    if "Invoice date" not in df.columns:
        raise ValueError("No 'Invoice date' column found in data")
    df["Invoice date"] = pd.to_datetime(df["Invoice date"], errors="coerce")

    # set it as index for resampling
    df = df.set_index("Invoice date")

    # --- pick column from UserMessage ---
    if column is None:
        if "price" in UserMessage.lower():
            column = "price"
        elif "patient" in UserMessage.lower():
            column = "mrn_number"  # count unique patients
        elif "doctor" in UserMessage.lower():
            column = "doctor_name"  # count unique doctors
        elif "treatment" in UserMessage.lower():
            column = "description"
        elif "revenue" in UserMessage.lower():
            column = "price"
        else:
            column = "price"  # fallback default

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    # --- handle numeric vs categorical ---
    if pd.api.types.is_numeric_dtype(df[column]):
        trend = df[column].resample(freq).mean()
        summary = f"{column.capitalize()} had {freq}-based trends. The lowest average was {trend.min():,.2f}, the highest reached {trend.max():,.2f}, and the overall average was {trend.mean():,.2f}."
    else:
        trend = df[column].resample(freq).nunique()
        summary = f"{column.capitalize()} counts varied across {freq.lower()}s. The lowest was {trend.min()}, the highest was {trend.max()}, with an overall average of {trend.mean():.0f} unique {column}s."

    # --- polish with Gemini ---
    polished = polish_with_gemini(summary)
    return polished



def predict(df, userMessage, model=None):
    """
    Generalized forecast function that predicts based on user query.
    It auto-detects what the user wants (patients, revenue, treatments, doctors).
    """
    
    # --- Step 1: Decide target column from userMessage ---
    userMessage = userMessage.lower()

    if "revenue" in userMessage or "price" in userMessage:
        target_col = "price"
        agg_func = "sum"
    elif "patient" in userMessage:
        target_col = "mrn_number"  # assuming each row = patient visit
        agg_func = "count"
    elif "doctor" in userMessage:
        target_col = "doctor_name"
        agg_func = "count"
    elif "treatment" in userMessage or "description" in userMessage:
        target_col = "description"
        agg_func = "count"
    else:
        raise ValueError("Couldn't detect what to predict from the query")

    # --- Step 2: Prepare time series ---
    # make sure date column exists
    if "Invoice date" in df.columns:
        date_col = "Invoice date"
    elif "Registration date" in df.columns:
        date_col = "Registration date"
    else:
        raise ValueError("No valid date column found in dataset")

    # aggregate daily
    if agg_func == "sum":
        daily = df.groupby(date_col)[target_col].sum().reset_index()
    else:
        daily = df.groupby(date_col)[target_col].count().reset_index()

    daily.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)

    # --- Step 3: Forecast with Prophet ---
    model_p = Prophet()
    model_p.fit(daily)

    # decide horizon from user query
    if "month" in userMessage or "30" in userMessage:
        periods = 30
    elif "week" in userMessage or "7" in userMessage:
        periods = 7
    elif "day" in userMessage:
        periods = 1
    elif "year" in userMessage or "365" in userMessage:
        periods = 365
    else:
        periods = 30  # default = 1 month

    future = model_p.make_future_dataframe(periods=periods)
    predict = model_p.predict(future)

    # --- Step 4: Summarize result ---
    yhat_total = predict.tail(periods)["yhat"].sum()

    result= (
        f"Based on historical data, I project approximately {int(yhat_total)} "
        f"{target_col.replace('_', ' ')}s over the next {periods} days."
    )

    # --- Step 5: Polish with Gemini (optional) ---
    if model:
        result = polish_with_gemini(result, model)

    return result
