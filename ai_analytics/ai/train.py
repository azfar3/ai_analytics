import frappe
import pandas as pd
import pickle
import os
from frappe.utils import nowdate

MODEL_PATH = "/home/erp/frappe-v15/apps/ai_analytics/ai_analytics/ai/salary_model.pkl"

@frappe.whitelist()
def rrr():
    print("sssssssssssssssss")


@frappe.whitelist()
def run_training():

    # 1. Get data
    data = frappe.db.sql("""
        SELECT 
            e.date_of_joining,
            e.designation,
            e.department,
            ss.gross_pay
        FROM `tabEmployee` e
        JOIN `tabSalary Slip` ss ON ss.employee = e.name
        WHERE ss.docstatus = 1
    """, as_dict=True)
    clean_data = []
    for row in data:
        clean_row = {
            "date_of_joining": str(row.get("date_of_joining")) if row.get("date_of_joining") else None,
            "designation": str(row.get("designation")) if row.get("designation") else None,
            "department": str(row.get("department")) if row.get("department") else None,
            "gross_pay": float(row.get("gross_pay")) if row.get("gross_pay") else None
        }
        clean_data.append(clean_row)

    df = pd.DataFrame(clean_data)

    # 2. Clean data
    df = df.dropna()

    # 3. Calculate experience
    df["date_of_joining"] = pd.to_datetime(df["date_of_joining"])
    today = pd.to_datetime(nowdate())
    df["experience"] = (today - df["date_of_joining"]).dt.days / 365

    # 4. Convert text to numbers
    df["designation"] = df["designation"].astype("category").cat.codes
    df["department"] = df["department"].astype("category").cat.codes

    # 5. Train model
    from sklearn.linear_model import LinearRegression

    X = df[["experience", "designation", "department"]]
    y = df["gross_pay"]

    model = LinearRegression()
    model.fit(X, y)

    # 6. Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return "Model trained and saved"
