import io
import re
import fitz
import json
import base64
import requests
import subprocess
from groq import Groq
from PIL import Image
from pypdf import PdfReader
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import frappe
from datetime import datetime
from frappe.utils import add_years, getdate, now, today


class LeaveFeatureEngineer:
    VAGUE_KEYWORDS = {"other", "personal", "urgent", "not well", "n/a", "na", "none"}

    def __init__(self, leaves: List[Dict], holidays: List[Dict]):
        self.leaves = leaves or []
        self.holidays = holidays or []
        self._holiday_dates = self._extract_holiday_dates()

    def _extract_holiday_dates(self) -> set:
        dates = set()
        for h in self.holidays:
            hd = h.get("holiday_date")
            if hd:
                if isinstance(hd, str):
                    hd = self._parse_date(hd)
                if isinstance(hd, date):
                    dates.add(hd)
        return dates

    @staticmethod
    def _parse_date(date_val):
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            from frappe.utils import getdate

            try:
                return getdate(date_val)
            except Exception:
                return None
        return None

    @staticmethod
    def _is_weekend(d: date) -> bool:
        return d.weekday() >= 5

    @staticmethod
    def _next_business_day(d: date) -> date:
        next_day = d + timedelta(days=1)
        while LeaveFeatureEngineer._is_weekend(next_day):
            next_day += timedelta(days=1)
        return next_day

    @staticmethod
    def _prev_business_day(d: date) -> date:
        prev_day = d - timedelta(days=1)
        while LeaveFeatureEngineer._is_weekend(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_policy_mode(self) -> str:
        if not self.leaves:
            return "TYPE_BASED"
        reason_based_count = sum(
            1
            for l in self.leaves
            if l.get("custom_policy_table") == 1 or l.get("project") == "ComNet - HR"
        )
        total = len(self.leaves)
        if reason_based_count == total:
            return "REASON_BASED"
        elif reason_based_count == 0:
            return "TYPE_BASED"
        return "MIXED"

    def count_vague_reasons(self, mode: Optional[str] = None) -> Dict:
        if mode is None:
            mode = self.get_policy_mode()
        if mode == "TYPE_BASED":
            return {"vague_count": 0, "vague_percentage": 0.0, "flag": False}

        count = 0
        for leave in self.leaves:
            reason = str(leave.get("custom_leave_reason") or "").lower()
            desc = str(leave.get("description") or "").lower()
            combined = f"{reason} {desc}"
            if any(kw in combined for kw in self.VAGUE_KEYWORDS):
                count += 1

        total = len(self.leaves)
        return {
            "vague_count": count,
            "vague_percentage": round((count / total * 100), 2) if total else 0.0,
            "flag": count > 0,
        }

    def detect_bridge_leaves(self) -> Dict:
        if not self.leaves or not self._holiday_dates:
            return {
                "bridge_count": 0,
                "pattern_breakdown": {"pre_holiday": 0, "post_holiday": 0},
                "flag": False,
            }

        pre_count = 0
        post_count = 0

        for leave in self.leaves:
            from_date = self._parse_date(leave.get("from_date"))
            to_date = self._parse_date(leave.get("to_date"))
            if not from_date or not to_date:
                continue

            day_after = self._next_business_day(to_date)
            if day_after in self._holiday_dates:
                pre_count += 1

            day_before = self._prev_business_day(from_date)
            if day_before in self._holiday_dates:
                post_count += 1

        total = pre_count + post_count
        return {
            "bridge_count": total,
            "pattern_breakdown": {"pre_holiday": pre_count, "post_holiday": post_count},
            "flag": total > 0,
        }

    def calculate_cluster_features(self) -> Dict:
        if len(self.leaves) < 3:
            return {"cluster_leave_count": 0}

        sorted_leaves = sorted(
            [l for l in self.leaves if self._parse_date(l.get("from_date"))],
            key=lambda x: self._parse_date(x.get("from_date")),
        )

        clusters = []
        used = set()

        for i, anchor in enumerate(sorted_leaves):
            anchor_date = self._parse_date(anchor.get("from_date"))
            window_end = anchor_date + timedelta(days=30)

            cluster = [anchor]
            for other in sorted_leaves[i + 1 :]:
                other_date = self._parse_date(other.get("from_date"))
                if other_date <= window_end:
                    cluster.append(other)
                else:
                    break

            if len(cluster) >= 3:
                names = tuple(l.get("name") for l in cluster)
                if not any(n in used for n in names if n):
                    clusters.append({"count": len(cluster)})
                    used.update(n for n in names if n)

        return {"cluster_leave_count": len(clusters)}

    def calculate_frequency_manipulation(self) -> Dict:
        if not self.leaves:
            return {"frequency_manipulation_count": 0}

        weekday_counts = Counter()
        for leave in self.leaves:
            fd = self._parse_date(leave.get("from_date"))
            if fd:
                weekday_counts[fd.weekday()] += 1

        flags = sum(1 for c in weekday_counts.values() if c >= 3)
        return {"frequency_manipulation_count": flags}

    def calculate_duration_manipulation(self) -> Dict:
        if not self.leaves:
            return {"duration_manipulation_count": 0}

        patterns = 0
        duration_counts = Counter()
        half_day_count = 0

        for leave in self.leaves:
            from_date = self._parse_date(leave.get("from_date"))
            to_date = self._parse_date(leave.get("to_date"))
            if from_date and to_date:
                dur = (to_date - from_date).days + 1
                duration_counts[dur] += 1

            if leave.get("half_day") == 1 or leave.get("custom_short_leave") == 1:
                half_day_count += 1

        patterns += sum(1 for c in duration_counts.values() if c >= 3)
        if half_day_count >= 3:
            patterns += 1

        return {"duration_manipulation_count": patterns}

    def calculate_type_misuse(self) -> Dict:
        if not self.leaves:
            return {"type_misuse_count": 0}

        reason_types = {}
        for leave in self.leaves:
            reason = str(leave.get("custom_leave_reason") or "").lower().strip()
            ltype = leave.get("leave_type", "Unknown")
            if reason:
                reason_types.setdefault(reason, set()).add(ltype)

        patterns = sum(1 for types in reason_types.values() if len(types) >= 3)
        return {"type_misuse_count": patterns}

    def check_data_integrity(self) -> Dict:
        flags = []
        if not self.leaves:
            flags.append("missing_leave_data")
        if not self.holidays:
            flags.append("missing_holidays")

        missing_fields = sum(
            1
            for l in self.leaves
            if not l.get("from_date") or not l.get("to_date") or not l.get("leave_type")
        )
        if missing_fields:
            flags.append(f"missing_critical_fields:{missing_fields}")

        return {"data_integrity_flags": flags, "is_valid": len(flags) == 0}

    def to_dict(self) -> Dict:
        policy_mode = self.get_policy_mode()
        integrity = self.check_data_integrity()

        return {
            "policy_mode": policy_mode,
            "vague_reason_count": self.count_vague_reasons(policy_mode)["vague_count"],
            "bridge_leave_count": self.detect_bridge_leaves()["bridge_count"],
            "cluster_leave_count": self.calculate_cluster_features()[
                "cluster_leave_count"
            ],
            "frequency_manipulation_count": self.calculate_frequency_manipulation()[
                "frequency_manipulation_count"
            ],
            "duration_manipulation_count": self.calculate_duration_manipulation()[
                "duration_manipulation_count"
            ],
            "type_misuse_count": self.calculate_type_misuse()["type_misuse_count"],
            "data_integrity_flags": integrity["data_integrity_flags"],
            "data_integrity_valid": integrity["is_valid"],
            "total_leaves": len(self.leaves),
            "total_holidays": len(self.holidays),
        }


@frappe.whitelist()
def leave_analysis(doc=None, employee=None, name=None, from_date=None):
    if doc:
        employee = doc.employee
        name = doc.name
        from_date = doc.from_date

        is_active = frappe.db.get_value(
            "AI Prompt", {"linked_doc": doc.doctype}, "is_active"
        )

        if not is_active:
            print("Not Active")
            return

    reference_date = add_years(getdate(from_date), -1)

    query = """
        SELECT
            name,
            employee_name,
            from_date,
            to_date,
            project,
            total_leave_days,
            leave_type,
            half_day,
            custom_policy_table,
            custom_leave_reason,
            description,
            custom_short_leave,
            leave_balance,
            custom_paid_days_status,
            status,
            leave_approver_name
        FROM
            `tabLeave Application`
        WHERE
            employee = %s
            AND from_date >= %s
            AND status = "Approved"
        ORDER BY
            from_date DESC
    """
    leaves = frappe.db.sql(query, (employee, reference_date), as_dict=True)

    holiday_list = frappe.db.get_value("Employee", employee, "holiday_list")
    holidays = []
    if holiday_list:
        holidays = frappe.db.get_all(
            "Holiday",
            filters={"parent": holiday_list, "holiday_date": [">=", reference_date]},
            fields=["holiday_date", "description as holiday_reason"],
        )

    engineer = LeaveFeatureEngineer(leaves, holidays)
    features = engineer.to_dict()

    prompt_template = frappe.db.get_value(
        "AI Prompt", {"linked_doc": doc.doctype}, "prompt"
    )

    if prompt_template:
        prompt = str(prompt_template).replace(
            "{calculated_features}", json.dumps(features, indent=2)
        )

    # print(prompt)
    result = get_model_response(prompt)
    print(result)

    if not result or result == "Failed":
        frappe.msgprint("AI analysis failed.")
        return {"status": "failed"}

    try:
        content = json.loads(result["content"])
    except json.JSONDecodeError:
        frappe.msgprint("Invalid AI response format.")
        return {"status": "failed"}

    index = result.get("index")

    if content:
        required_keys = ["ai_risk_score", "ai_risk_band", "ai_rule_flags", "verdict"]
        if not all(k in content for k in required_keys):
            frappe.msgprint("AI response missing required fields.")
            return {"status": "failed"}

        frappe.db.sql(
            """
            UPDATE `tabLeave Application`
            SET 
                custom_ai_risk_score = %(score)s,
                custom_ai_risk_band = %(band)s,
                custom_ai_rule_flag = %(flags)s,
                custom_ai_model = %(model)s,
                custom_ai_scored_on = %(now)s,
                custom_ai_feature_snapshot = %(snapshot)s,
                custom_verdict = %(verdict)s,
                custom_key_number = %(custom_key_number)s,
                modified = %(now)s
            WHERE 
                name = %(name)s
            """,
            {
                "score": content.get("ai_risk_score"),
                "band": content.get("ai_risk_band"),
                "flags": "\n".join(content.get("ai_rule_flags", [])),
                "model": "MM-FE",
                "now": now(),
                "snapshot": content.get("ai_summary", ""),
                "verdict": content.get("verdict"),
                "name": name,
                "custom_key_number": index,
            },
        )

    return {"status": "success"}


@frappe.whitelist()
def finance_analysis(name=None):
    data = frappe.db.get_all(
        "Project Ledger Entry",
        filters={"name": name},
        fields=[
            "project",
            "contract",
            "transaction_type",
            "transaction_name",
            "date",
            "item",
            "sub_item",
            "designation",
            "province",
        ],
    )

    meta = data[0]
    project = meta.project
    contract = meta.contract
    item = meta.sub_item
    province = meta.province
    current_date = meta.date

    ledgers = frappe.db.get_all(
        "Project Ledger Entry",
        filters={
            "project": project,
            "contract": contract,
            "sub_item": item,
            "delinked": ["=", 0],
        },
        fields=[
            "project",
            "contract",
            "transaction_type",
            "transaction_name",
            "date",
            "item",
            "sub_item",
            "quantity",
            "amount",
            "designation",
            "province",
        ],
    )

    orders = []
    active_dates = []

    for row in ledgers:
        if row.transaction_type == "Sales Order":
            so_data = frappe.db.get_value(
                "Sales Order",
                row.transaction_name,
                [
                    "custom_contract_start_date",
                    "custom_contract_end_date",
                    "custom_province_required",
                    "province",
                ],
                as_dict=True,
            )

            if not so_data:
                continue

            if province and so_data.province and province != so_data.province:
                continue

            if (
                so_data.custom_province_required
                and not so_data.province
                and not province
            ):
                frappe.throw(
                    f"Province is mandatory for Sales Order {row.transaction_name}"
                )

            if (
                so_data.custom_contract_start_date
                <= current_date
                <= so_data.custom_contract_end_date
            ):
                row.update(
                    {
                        "start": so_data.custom_contract_start_date,
                        "end": so_data.custom_contract_end_date,
                    }
                )
                orders.append(row)
                active_dates.append(
                    (
                        so_data.custom_contract_start_date,
                        so_data.custom_contract_end_date,
                    )
                )

    if not orders:
        frappe.throw("No active Sales Order found for the current date and dimensions.")

    invoices = []

    for row in ledgers:
        if row.transaction_type != "Sales Order":

            ledger_date = row.get("date")

            if ledger_date:
                date_range = any(
                    start <= ledger_date <= end for start, end in active_dates
                )

                if date_range:
                    row.update({"date": ledger_date})
                    invoices.append(row)

    prompt = frappe.db.get_value(
        "AI Prompt", {"linked_doc": "Project Ledger Entry"}, "prompt"
    )
    prompt = str(prompt).replace("{item}", str(item))
    prompt = str(prompt).replace("{project}", str(project))
    prompt = str(prompt).replace("{contract}", str(contract))
    prompt = str(prompt).replace("{province}", str(province))
    prompt = str(prompt).replace("{orders}", str(orders))
    prompt = str(prompt).replace("{invoices}", str(invoices))
    print(prompt)

    result = get_model_response(prompt)
    print(result)
    return


@frappe.whitelist()
def dimension_analysis(doc, method):
    if doc.doctype == "Employee":
        if doc.branch == "SBC - Staff":
            return

        if not doc.project:
            return

        emp_name = doc.name
        project = doc.project
        contract = frappe.db.get_value("Employee", emp_name, "contract")
        province = doc.custom_provinces
        designation = doc.designation

        sales_orders = frappe.db.sql(
            """
            SELECT
                so.name,
                so.project,
                so.contract,
                so.province,
                so.creation,
                so.transaction_date
            FROM `tabSales Order` so
            INNER JOIN (
                SELECT
                    province,
                    MAX(transaction_date) AS latest_creation
                FROM `tabSales Order`
                WHERE project = %s
                AND docstatus = 1
                GROUP BY province
            ) latest
                ON so.province = latest.province
            AND so.creation = latest.latest_creation
            WHERE so.project = %s
            AND so.docstatus = 1  
            ORDER BY so.transaction_date DESC, so.creation DESC;
        """,
            (project, project),
            as_dict=True,
        )

        if not sales_orders:
            return

        filtered_contract = []

        for so in sales_orders:
            so_contract = so.get("contract")
            if (contract or "") == (so_contract or ""):
                filtered_contract.append(so)

        if not filtered_contract:
            frappe.throw(
                f"No Sales Orders found for Employee: <b>{emp_name}</b> with contract <b>{contract or '(blank)'}</b>. "
                f"Please ensure the Sales Order has the correct contract assigned."
            )

        emp_province = province
        filtered_province = []

        for so in filtered_contract:
            so_province = so.get("province")
            if (emp_province or "") == (so_province or ""):
                filtered_province.append(so)

        if not filtered_province:
            frappe.throw(
                f"No Sales Orders found for Employee: <b>{emp_name}</b> with province <b>{emp_province or '(blank)'}</b>. "
                f"Please ensure the Sales Order has the correct province assigned."
            )

        matched_items = []

        for so in filtered_province:
            items = frappe.db.sql(
                """
                SELECT
                    si.item_name,
                    si.custom_designation
                FROM `tabSales Order Item` si
                WHERE si.parent = %s AND si.item_name = "Position"
            """,
                (so.name),
                as_dict=True,
            )

            for item in items:
                item_designation = item.get("custom_designation")
                if item_designation == designation:
                    matched_entry["custom_designation"] = item.get("custom_designation")
                    matched_items.append(matched_entry)

        if not matched_items:
            frappe.throw(
                f"No Sales Order items found with designation '<b>{designation}</b>' "
                f"matching Employee: <b>{emp_name}</b>. Please check the Sales Order item designations."
            )

        return

    elif doc.doctype == "Sales Order":
        if not doc.project:
            return

        so_project = doc.project
        so_contract = doc.contract
        so_province = doc.province

        so_designations = set()
        for item in doc.get("items"):
            if item.item_name == "Position" and item.custom_designation:
                so_designations.add(item.custom_designation)

        employees = frappe.get_all(
            "Employee",
            filters={
                "project": so_project,
                "status": "Active",
                "branch": ["!=", "SBC - Staff"],
            },
            fields=["name", "contract", "custom_provinces", "designation"],
        )

        missing_designations = {}

        for emp in employees:
            if emp.contract == so_contract and emp.custom_provinces == so_province:
                designation = emp.designation

                if designation and designation not in so_designations:
                    if designation not in missing_designations:
                        missing_designations[designation] = []
                    missing_designations[designation].append(emp.name)

        if missing_designations:
            error_msg = (
                f"Cannot save Sales Order. Existing employees for Project <b>{so_project}</b> "
                f"(Contract: {so_contract or 'Blank'}, Province: {so_province or 'Blank'}) "
                f"require the following designations to be added as 'Position' items:<br><br>"
            )

            for desig, emps in missing_designations.items():
                error_msg += f"• <b>{desig}</b> (Required by: {', '.join(emps)})<br>"

            error_msg += "<br>Please add these positions before saving."
            frappe.throw(error_msg)


@frappe.whitelist()
def expense_analysis(name=None):
    exp = frappe.get_doc("Employee Expense Claim", name)

    emp = exp.employee
    mode_of_travel = exp.mode_of_travel

    vehicle_type = None
    ticket_amount = None
    travel_auth = []
    travel_info = []
    other_exp = []
    dsa_per_day = None
    total_dsa_amount = None
    total_other_expense_amount = None

    if mode_of_travel == "Rented Vehicle":
        vehicle_type = exp.vehicle_type
    if mode_of_travel == "By Air":
        ticket_amount = exp.ticket_amount

    if exp.travel_authorisation_request_form:
        travel_form = frappe.get_doc(
            "Travel Authorisation Request Form", exp.travel_authorisation_request_form
        )
        travel_auth.append(
            {
                "employee": travel_form.employee,
                "status": travel_form.docstatus,
                "purpose_of_travel": travel_form.purpose_of_travel,
                "travel_details": [
                    {
                        "from_date": detail.from_date,
                        "to_date": detail.to_date,
                        "from_time": detail.from_time,
                        "to_time": detail.to_time,
                        "source": detail.source,
                        "destination": detail.destination,
                        "description": detail.description,
                    }
                    for detail in travel_form.travel_information
                ],
            }
        )

    if exp.group_travel == 1:
        for info in exp.group_travel_information:
            travel_info.append(
                {
                    "travel_date": info.travel_date,
                    "from_time": info.from_time,
                    "to_time": info.to_time,
                    "source": info.source,
                    "destination": info.destination,
                    "km_travel": info.km_travel,
                }
            )
    else:
        for info in exp.travel_information:
            travel_info.append(
                {
                    "travel_date": info.travel_date,
                    "from_time": info.from_time,
                    "to_time": info.to_time,
                    "source": info.source,
                    "destination": info.destination,
                    "km_travel": info.km_travel,
                }
            )

    for other in exp.other_expenses:
        other_exp.append(
            {
                "type": other.type,
                "detail": other.detail,
                "amount": other.amount,
                "pay_to_employee": other.pay_to_employee,
                "vehicle_travel_cost": other.vehicle_travel_cost,
                "account": other.account,
            }
        )

    dsa_per_day = exp.dsa_per_day
    total_dsa_amount = exp.total_dsa_amount
    total_other_expense_amount = exp.total_other_expense_amount

    prompt = frappe.db.get_value(
        "AI Prompt", {"linked_doc": "Employee Expense Claim"}, "prompt"
    )
    prompt = prompt + (
        f"""
        \n\n
        Input Data: 
        
        Employee: {emp}, 
        Mode of Travel: {mode_of_travel}, 
        Vehicle Type: {vehicle_type}, 
        Ticket Amount: {ticket_amount}, 
        Travel Authorization: {travel_auth}, 
        Travel Information: {travel_info}, 
        Other Expenses: {other_exp},
        DSA Per Day: {dsa_per_day},
        Total DSA Amount: {total_dsa_amount},
        Total Other Expense Amount: {total_other_expense_amount}
    """
    )

    # print(prompt)

    result = get_model_response(prompt=prompt, type="expense")
    print(result)
    return


def get_model_response(prompt=None, type=None):
    if not prompt:
        return "Failed"

    api_keys = frappe.db.get_all("AI Keys", fields=["key"])
    success_data = None

    for index, entry in enumerate(api_keys):
        key = entry.get("key")
        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            response_data = completion.model_dump()
            print(response_data)
            content = response_data["choices"][0]["message"]["content"]
            success_data = "success"
            return {"content": content, "index": index + 1}

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                continue
            continue

    if not success_data:
        frappe.throw("Please try again later after some time.")


@frappe.whitelist()
def fetch_school_data(name=None, file=None, cr_name=None, district=None):
    file_url = None

    if name:
        beneficiary_doc = frappe.get_doc("Beneficiary Master", name)
        for f in beneficiary_doc.beneficiary_attachments:
            if f.get("title") == "Requirement Form":
                file_url = f.get("attachment")
                break

        if not file_url:
            frappe.throw("No PDF attachment with title 'Requirement Form' found.")
    elif file:
        file_url = file
    else:
        frappe.throw("No attachment Found")

    file_doc = frappe.get_doc("File", {"file_url": file_url})
    file_path = file_doc.get_full_path()

    with open(file_path, "rb") as f:
        file_content = f.read()

    pdf_doc = fitz.open(stream=file_content, filetype="pdf")
    images = []

    for page_num in range(0, min(1, len(pdf_doc))):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if img.width > 800:
            ratio = 800 / img.width
            new_height = int(img.height * ratio)
            img = img.resize((800, new_height), Image.LANCZOS)

        img = img.convert("L")
        img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=50, optimize=True)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(img_str)

    pdf_doc.close()

    prompt = frappe.db.get_value(
        "AI Prompt", {"linked_doc": "Beneficiary Master"}, "prompt"
    )

    print(f"Prompt chars: {len(prompt)}, Images: {len(images)}", "AI Analytics Debug")

    api_keys = frappe.db.get_all("AI Keys", "key")
    if not api_keys:
        frappe.throw("No API keys configured in AI Keys.")

    success = False
    data = None

    for idx, entry in enumerate(api_keys):
        key = entry.get("key")
        key_name = entry.get("name")
        if not key:
            continue

        current_count = frappe.db.get_value("AI Keys", key_name, "count") or 0
        content = [{"type": "text", "text": prompt}]

        for img_str in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                }
            )

        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
                response_format={"type": "json_object"},
                reasoning_effort="none",
                reasoning_format="hidden",
            )

            response_data = completion.model_dump()
            raw_content = response_data["choices"][0]["message"]["content"]
            data = _extract_json(raw_content)

            if not data or not data.get("parent_data"):
                frappe.db.set_value("AI Keys", key_name, {"status": "In Waiting"})
                print(
                    f"Key {idx+1}: JSON extraction failed.\n"
                    f"Raw content (first 500 chars):\n{raw_content[:500]}",
                    "AI Analytics Debug",
                )
                continue

            frappe.db.set_value(
                "AI Keys", key_name, {"status": "Active", "count": current_count + 1}
            )

            success = True
            break

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                frappe.db.set_value("AI Keys", key_name, {"status": "Expired"})
                print(f"Key {idx+1}: Rate limit error (429).", "AI Analytics Debug")
            elif (
                "413" in error_str
                or "too large" in error_str
                or "requested" in error_str
            ):
                print(
                    f"Key {idx+1}: Request too large (413). Try reducing image size.",
                    "AI Analytics Debug",
                )
                continue
            elif any(
                code in error_str
                for code in [
                    "502",
                    "503",
                    "504",
                    "server error",
                    "bad gateway",
                    "gateway timeout",
                    "service unavailable",
                ]
            ):
                frappe.db.set_value("AI Keys", key_name, {"status": "In Waiting"})
                print(f"Key {idx+1}: Server error", "AI Analytics Debug")
            else:
                frappe.db.set_value("AI Keys", key_name, {"status": "In Waiting"})
                print(f"Key {idx+1}: Unexpected error — {str(e)}", "AI Analytics Debug")
            continue

    if not success:
        frappe.throw("Please try again later after some time.")
    parent = data.get("parent_data", {})

    if name:
        beneficiary_doc.beneficiary_name = (
            parent.get("school_name") or beneficiary_doc.beneficiary_name
        )
        beneficiary_doc.district = parent.get("district") or beneficiary_doc.district
        beneficiary_doc.tehsil = parent.get("tehsil") or beneficiary_doc.tehsil
        beneficiary_doc.uc = parent.get("uc") or beneficiary_doc.uc
        beneficiary_doc.ref_code = parent.get("emis_code") or beneficiary_doc.ref_code
        beneficiary_doc.school_level = (
            parent.get("school_level") or beneficiary_doc.school_level
        )
        beneficiary_doc.school_gender = (
            parent.get("gender") or beneficiary_doc.school_gender
        )
        beneficiary_doc.total_existing_class_rooms = (
            parent.get("class_rooms")
            if parent.get("class_rooms") is not None
            else beneficiary_doc.total_existing_class_rooms
        )
        beneficiary_doc.total_working_teachers = (
            parent.get("total_teachers")
            if parent.get("total_teachers") is not None
            else beneficiary_doc.total_working_teachers
        )
        beneficiary_doc.total_enrollment = (
            parent.get("total_enrollment")
            if parent.get("total_enrollment") is not None
            else beneficiary_doc.total_enrollment
        )
        beneficiary_doc.acc_number = (
            parent.get("bank_account") or beneficiary_doc.acc_number
        )
        beneficiary_doc.branch_name = (
            parent.get("bank_city") or beneficiary_doc.branch_name
        )

        parsed_date = _safe_date(parent.get("submission_date"))
        if parsed_date:
            beneficiary_doc.sip_submission_date = parsed_date

        beneficiary_doc.set("requirements_table", [])
        for req in data.get("requirements", []):
            beneficiary_doc.append(
                "requirements_table",
                {
                    "facility_name": req.get("facility", ""),
                    "current_status": req.get("status", ""),
                    "requirements": req.get("requirement", ""),
                    "unit_cost": req.get("unit_cost", 0) or 0,
                    "total_cost": req.get("total_cost", 0) or 0,
                    "priority": req.get("priority", 0) or 0,
                    "responsibility_assigned": req.get("responsibility", ""),
                    "timeline": "2026-06-30",
                },
            )

        beneficiary_doc.save()
        frappe.db.commit()
    else:
        gend_ = parent.get("gender")
        if gend_ in ("F", "Girls"):
            gend_ = "Female"
        if gend_ in ("M", "Boys"):
            gend_ = "Male"
        sch_level = "Primary"
        if parent.get("school_level"):
            sch_level = parent.get("school_level")
        if parent.get("school_name", "").lower().startswith("ggps"):
            gend_ = "Female"
        new_doc = frappe.new_doc("Beneficiary Master")
        new_doc.province = 3
        new_doc.project = "Financial Support to PTCs for SDPs - KP - 43459411"
        new_doc.contract_name = "Financial Support to PTCs for SDPs - KP - 43459411"
        new_doc.beneficiary_name = parent.get("school_name")
        new_doc.beneficiary_type = "School"
        new_doc.ref_code = parent.get("emis_code")
        if district:
            new_doc.district = district

        boys_count = parent.get("boys")
        if boys_count:
            new_doc.male_enrollment = boys_count
        girls_count = parent.get("girls")
        if girls_count:
            new_doc.female_enrollment = girls_count

        new_doc.tehsil = parent.get("tehsil")
        new_doc.uc = parent.get("uc")
        new_doc.school_level = sch_level
        new_doc.school_gender = gend_
        new_doc.total_existing_class_rooms = parent.get("class_rooms")
        new_doc.total_working_teachers = parent.get("total_teachers")
        new_doc.total_enrollment = parent.get("total_enrollment")
        new_doc.eligibility_status = "Pending"
        new_doc.approved_amount = 1296187
        new_doc.payment_rule = "Installments"
        new_doc.num_installements = 2
        new_doc.payment_mode = "Online Transfer"
        new_doc.acc_number = parent.get("bank_account")
        if parent.get("bank_name"):
            new_doc.bank_name = parent.get("bank_name")
        new_doc.branch_name = parent.get("bank_city")

        submission_date = _safe_date(parent.get("submission_date"))
        if submission_date:
            new_doc.sip_submission_date = submission_date

        for req in data.get("requirements", []):
            new_doc.append(
                "requirements_table",
                {
                    "facility_name": req.get("facility", ""),
                    "current_status": req.get("status", ""),
                    "requirements": req.get("requirement", ""),
                    "unit_cost": req.get("unit_cost", 0) or 0,
                    "total_cost": req.get("total_cost", 0) or 0,
                    "priority": req.get("priority", 0) or 0,
                    "responsibility_assigned": req.get("responsibility", ""),
                    "timeline": "2026-06-30",
                },
            )

        new_doc.insert(ignore_permissions=True)

        if cr_name:
            frappe.db.set_value(
                "Registration Table",
                cr_name,
                {"status": "Done", "master_form": new_doc.name},
            )

    return {"status": "success"}


def _extract_json(raw_content):
    if not raw_content:
        return None

    raw_content = raw_content.strip()

    def _try_parse(json_str):
        if not json_str:
            return None
        json_str = json_str.strip()

        fixed = re.sub(r"(:\s*)0([1-9]\d*)(\s*[,}\]])", r"\1\2\3", json_str)

        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    patterns = [
        r"```json\s*\n?([\s\S]*?)\n?\s*```",
        r"```\s*\n?([\s\S]*?)\n?\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_content)
        if match:
            result = _try_parse(match.group(1))
            if result:
                return result

    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = _try_parse(raw_content[start : end + 1])
        if result:
            return result

    if raw_content.startswith("{"):
        result = _try_parse(raw_content)
        if result:
            return result

    return None


def _safe_date(date_str):
    if not date_str or not str(date_str).strip():
        return None
    date_str = str(date_str).strip()
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%d.%m.%y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    try:
        return frappe.utils.getdate(date_str)
    except Exception:
        return None
