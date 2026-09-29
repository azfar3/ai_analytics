import frappe
import cv2
import pandas as pd


# ================================
# MAIN TRIGGER FUNCTION
# ================================
def process_checklist(doc, method):

    # 1. Check if image exists
    if not doc.image:
        return

    # 2. Avoid duplicate processing
    if doc.processed:
        return

    # 3. Get full file path
    file_path = frappe.get_site_path(doc.image.strip("/"))

    # 4. Extract data from image
    data = extract_checklist_data(file_path)

    # 5. Insert into child table
    for row in data:
        doc.append("items", row)

    # 6. Mark as processed
    doc.processed = 1


# ================================
# IMAGE → DATA PIPELINE
# ================================
def extract_checklist_data(image_path):

    marks = detect_marks(image_path)
    structured_data = map_to_table(marks)

    return structured_data


# ================================
# STEP 1: DETECT MARKS (✔ / ✖)
# ================================
def detect_marks(image_path):

    img = cv2.imread(image_path, 0)  # grayscale

    # Threshold to highlight marks
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours (possible marks)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small shapes (ticks/cross)
        if 15 < w < 60 and 15 < h < 60:
            marks.append((x, y))

    return marks


# ================================
# STEP 2: MAP TO TABLE STRUCTURE
# ================================
def map_to_table(marks):

    tasks = [
        "Clean sinks",
        "Clean toilets",
        "Clean floors",
        "Refill paper",
        "Refill soap",
        "Empty trash"
    ]

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    slots = ["Morning", "Afternoon", "Evening"]

    data = []

    for (x, y) in marks:

        # ⚠️ THESE VALUES MAY NEED ADJUSTMENT
        row = int(y / 50)
        col = int(x / 80)

        if row < len(tasks) * len(slots) and col < len(days):

            slot_index = row // len(tasks)
            task_index = row % len(tasks)

            data.append({
                "time_slot": slots[slot_index],
                "task": tasks[task_index],
                "day": days[col],
                "status": "Done"
            })

    return data
