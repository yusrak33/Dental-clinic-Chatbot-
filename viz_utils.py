# viz_utils.py

import matplotlib.pyplot as plt
import pandas as pd
import os

# Create folder for saving visualizations
VIS_DIR = "static/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)


def save_chart(fig, name="chart.png"):
    """
    Saves the chart in the visualizations directory and returns the path.
    """
    path = os.path.join(VIS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def detect_visualization_request(user_query: str) -> bool:
    """
    Simple check if user query is about visualization.
    """
    keywords = ["chart", "plot", "graph", "visual", "show", "distribution", "trend"]
    return any(word in user_query.lower() for word in keywords)


def generate_chart_data(user_query: str, df: pd.DataFrame):
    """
    Decides which visualization to generate based on query.
    Returns (message, chart_path or None).
    """
    query = user_query.lower()

    # --- Revenue related ---
    if "revenue" in query and "month" in query:
        if "date" not in df.columns or "revenue" not in df.columns:
            return "Dataset missing 'date' or 'revenue' columns.", None

        df['date'] = pd.to_datetime(df['date'])
        monthly = df.groupby(df['date'].dt.to_period("M"))['revenue'].sum()

        fig, ax = plt.subplots()
        monthly.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Monthly Revenue Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")

        return "Here’s the monthly revenue trend chart:", save_chart(fig, "monthly_revenue.png")

    elif "revenue" in query and ("service" in query or "treatment" in query):
        if "service" not in df.columns or "revenue" not in df.columns:
            return "Dataset missing 'service' or 'revenue' columns.", None

        service_rev = df.groupby("service")["revenue"].sum().sort_values(ascending=False)

        fig, ax = plt.subplots()
        service_rev.plot(kind="bar", ax=ax)
        ax.set_title("Revenue by Service")
        ax.set_ylabel("Revenue")
        ax.set_xlabel("Service")
        plt.xticks(rotation=45)

        return "Here’s the revenue by service chart:", save_chart(fig, "revenue_by_service.png")

    # --- Patients related ---
    elif "patients" in query and "month" in query:
        if "date" not in df.columns or "patient_id" not in df.columns:
            return "Dataset missing 'date' or 'patient_id' columns.", None

        df['date'] = pd.to_datetime(df['date'])
        monthly_patients = df.groupby(df['date'].dt.to_period("M"))['patient_id'].nunique()

        fig, ax = plt.subplots()
        monthly_patients.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Patients per Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Unique Patients")

        return "Here’s the patients per month chart:", save_chart(fig, "patients_per_month.png")

    elif "age" in query and "patients" in query:
        if "age" not in df.columns:
            return "Dataset missing 'age' column.", None

        fig, ax = plt.subplots()
        df['age'].plot(kind="hist", bins=10, ax=ax, edgecolor="black")
        ax.set_title("Patient Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")

        return "Here’s the patient age distribution chart:", save_chart(fig, "patient_age_distribution.png")

    # --- Invoices related ---
    elif "invoice" in query and "month" in query:
        if "date" not in df.columns or "invoice_amount" not in df.columns:
            return "Dataset missing 'date' or 'invoice_amount' columns.", None

        df['date'] = pd.to_datetime(df['date'])
        monthly_invoices = df.groupby(df['date'].dt.to_period("M"))['invoice_amount'].sum()

        fig, ax = plt.subplots()
        monthly_invoices.plot(kind="bar", ax=ax)
        ax.set_title("Invoice Amounts by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Invoice Amount")

        return "Here’s the invoices by month chart:", save_chart(fig, "invoices_by_month.png")

    elif "invoice" in query and ("paid" in query or "unpaid" in query):
        if "status" not in df.columns:
            return "Dataset missing 'status' column.", None

        status_counts = df['status'].value_counts()

        fig, ax = plt.subplots()
        status_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Paid vs Unpaid Invoices")

        return "Here’s the invoice status chart:", save_chart(fig, "invoice_status.png")

    # --- Services related ---
    elif "service" in query or "treatment" in query:
        if "service" not in df.columns:
            return "Dataset missing 'service' column.", None

        service_counts = df['service'].value_counts()

        fig, ax = plt.subplots()
        service_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title("Treatment/Service Popularity")

        return "Here’s the treatment/service popularity chart:", save_chart(fig, "service_popularity.png")

    # --- Default fallback ---
    return "Sorry, I couldn’t detect what visualization you need. Try asking about revenue, patients, invoices, or services.", None
