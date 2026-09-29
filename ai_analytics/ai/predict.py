import frappe
import pickle
import pandas as pd
from frappe.utils import nowdate

MODEL_PATH = "/home/erp/frappe-v15/apps/ai_analytics/ai_analytics/ai/salary_model.pkl"


@frappe.whitelist()
def get_salary_prediction(employee):

    # 1. Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. Get employee
    emp = frappe.get_doc("Employee", employee)

    # 3. Calculate experience
    doj = pd.to_datetime(emp.date_of_joining)
    today = pd.to_datetime(nowdate())
    experience = (today - doj).days / 365

    # 4. Convert text → number (simple way)
    designation = hash(emp.designation) % 100
    department = hash(emp.department) % 100

    # 5. Predict
    df = pd.DataFrame([{
        "experience": experience,
        "designation": designation,
        "department": department
    }])

    prediction = model.predict(df)[0]

    return prediction
