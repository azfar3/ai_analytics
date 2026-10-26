import re
import json
import fitz
import base64
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional

import frappe
from frappe.utils import add_years, getdate, now, today


@frappe.whitelist()
def extract_pdf_images(
    file_url: str, max_pages: int = 2, dpi_scale: float = 2.0
) -> str:
    try:
        file_name = frappe.db.get_value("File", {"file_url": file_url}, "name")
        if not file_name:
            return json.dumps(
                {"status": "error", "message": f"File resolve error: File {file_url} not found"}
            )
        file_doc = frappe.get_doc("File", file_name)
        file_path = file_doc.get_full_path()
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"File resolve error: {str(e)}"}
        )

    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
    except Exception as e:
        return json.dumps({"status": "error", "message": f"File read error: {str(e)}"})

    try:
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        images = []

        for page_num in range(min(max_pages, len(pdf_doc))):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale))
            img_data = pix.tobytes("jpeg")
            img_str = base64.b64encode(img_data).decode("utf-8")
            images.append(f"data:image/jpeg;base64,{img_str}")

        pdf_doc.close()
        return json.dumps(
            {"status": "success", "images": images, "pages_processed": len(images)}
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": f"PDF render error: {str(e)}"})


@frappe.whitelist()
def parse_extracted_json(raw_content: str) -> str:
    if not raw_content:
        return json.dumps({"status": "error", "message": "Empty content provided."})

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
                return json.dumps({"status": "success", "data": result})

    start = raw_content.find("{")
    end = raw_content.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = _try_parse(raw_content[start : end + 1])
        if result:
            return json.dumps({"status": "success", "data": result})

    if raw_content.startswith("{"):
        result = _try_parse(raw_content)
        if result:
            return json.dumps({"status": "success", "data": result})

    return json.dumps(
        {"status": "error", "message": "No valid JSON object found in content."}
    )


@frappe.whitelist()
def insert_beneficiary_from_extraction(
    extracted_data: dict, district: str = None, cr_name: str = None
) -> str:
    try:
        parent = extracted_data.get("parent_data", {})
        requirements = extracted_data.get("requirements", [])

        gend_ = parent.get("gender")
        if gend_ in ("F", "Girls"):
            gend_ = "Female"
        if gend_ in ("M", "Boys"):
            gend_ = "Male"
        if parent.get("school_name", "").lower().startswith("ggps"):
            gend_ = "Female"

        sch_level = parent.get("school_level", "Primary")

        submission_date = _safe_date_tool(parent.get("submission_date"))

        existing_name = None
        emis_code = parent.get("emis_code")
        if emis_code:
            existing = frappe.db.get_value(
                "Beneficiary Master", {"ref_code": emis_code}, "name"
            )
            if existing:
                existing_name = existing

        if existing_name:
            doc = frappe.get_doc("Beneficiary Master", existing_name)
            doc.beneficiary_name = parent.get("school_name") or doc.beneficiary_name
            doc.district = parent.get("district") or doc.district
            doc.tehsil = parent.get("tehsil") or doc.tehsil
            doc.uc = parent.get("uc") or doc.uc
            doc.ref_code = emis_code or doc.ref_code
            doc.school_level = sch_level or doc.school_level
            doc.school_gender = gend_ or doc.school_gender
            if parent.get("class_rooms") is not None:
                doc.total_existing_class_rooms = parent.get("class_rooms")
            if parent.get("total_teachers") is not None:
                doc.total_working_teachers = parent.get("total_teachers")
            if parent.get("total_enrollment") is not None:
                doc.total_enrollment = parent.get("total_enrollment")
            doc.acc_number = parent.get("bank_account") or doc.acc_number
            doc.branch_name = parent.get("bank_city") or doc.branch_name
            if submission_date:
                doc.sip_submission_date = submission_date
        else:
            doc = frappe.new_doc("Beneficiary Master")
            doc.province = 3
            doc.project = "Financial Support to PTCs for SDPs - KP - 43459411"
            doc.contract_name = "Financial Support to PTCs for SDPs - KP - 43459411"
            doc.beneficiary_name = parent.get("school_name")
            doc.beneficiary_type = "School"
            doc.ref_code = emis_code
            doc.district = district
            doc.tehsil = parent.get("tehsil")
            doc.uc = parent.get("uc")
            doc.school_level = sch_level
            doc.school_gender = gend_
            doc.total_existing_class_rooms = parent.get("class_rooms")
            doc.total_working_teachers = parent.get("total_teachers")
            doc.total_enrollment = parent.get("total_enrollment")
            doc.eligibility_status = "Pending"
            doc.approved_amount = 1296187
            doc.payment_rule = "Installments"
            doc.num_installements = 2
            doc.payment_mode = "Online Transfer"
            doc.acc_number = parent.get("bank_account")
            doc.bank_name = parent.get("bank_name")
            doc.branch_name = parent.get("bank_city")
            if submission_date:
                doc.sip_submission_date = submission_date

            if parent.get("boys"):
                doc.male_enrollment = parent.get("boys")
            if parent.get("girls"):
                doc.female_enrollment = parent.get("girls")

        doc.set("requirements_table", [])
        for req in requirements:
            doc.append(
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

        if existing_name:
            doc.save(ignore_permissions=True)
        else:
            doc.insert(ignore_permissions=True)

        if cr_name and not existing_name:
            frappe.db.set_value(
                "Registration Table",
                cr_name,
                {
                    "status": "Done",
                    "master_form": doc.name,
                },
            )

        frappe.db.commit()
        return json.dumps(
            {
                "status": "success",
                "beneficiary": doc.name,
                "mode": "update" if existing_name else "create",
            }
        )

    except Exception as e:
        frappe.db.rollback()
        return json.dumps({"status": "error", "message": str(e)})


def _safe_date_tool(date_str):
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
            from datetime import datetime

            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    try:
        return frappe.utils.getdate(date_str)
    except Exception:
        return None


@frappe.whitelist()
def fetch_expense_claim_structure(claim_name: str) -> str:
    try:
        exp = frappe.get_doc("Employee Expense Claim", claim_name)

        travel_auth = []
        if exp.travel_authorisation_request_form:
            taf = frappe.get_doc(
                "Travel Authorisation Request Form",
                exp.travel_authorisation_request_form,
            )
            travel_auth.append(
                {
                    "employee": taf.employee,
                    "status": taf.docstatus,
                    "purpose_of_travel": taf.purpose_of_travel,
                    "travel_details": [
                        {
                            "from_date": str(d.from_date),
                            "to_date": str(d.to_date),
                            "from_time": str(d.from_time) if d.from_time else None,
                            "to_time": str(d.to_time) if d.to_time else None,
                            "source": d.source,
                            "destination": d.destination,
                            "description": d.description,
                        }
                        for d in taf.travel_information
                    ],
                }
            )

        travel_info = []
        source_field = (
            exp.group_travel_information
            if exp.group_travel == 1
            else exp.travel_information
        )
        for info in source_field:
            travel_info.append(
                {
                    "travel_date": str(info.travel_date),
                    "from_time": str(info.from_time) if info.from_time else None,
                    "to_time": str(info.to_time) if info.to_time else None,
                    "source": info.source,
                    "destination": info.destination,
                    "km_travel": info.km_travel,
                }
            )

        other_exp = []
        for other in exp.other_expenses:
            other_exp.append(
                {
                    "type": other.type,
                    "detail": other.detail,
                    "amount": float(other.amount or 0),
                    "pay_to_employee": other.pay_to_employee,
                    "vehicle_travel_cost": other.vehicle_travel_cost,
                    "account": other.account,
                }
            )

        payload = {
            "employee": exp.employee,
            "mode_of_travel": exp.mode_of_travel,
            "vehicle_type": (
                exp.vehicle_type if exp.mode_of_travel == "Rented Vehicle" else None
            ),
            "ticket_amount": (
                float(exp.ticket_amount or 0)
                if exp.mode_of_travel == "By Air"
                else None
            ),
            "travel_authorization": travel_auth,
            "travel_information": travel_info,
            "other_expenses": other_exp,
            "dsa_per_day": float(exp.dsa_per_day or 0),
            "total_dsa_amount": float(exp.total_dsa_amount or 0),
            "total_other_expense_amount": float(exp.total_other_expense_amount or 0),
            "claim_total": float(exp.total_claimed_amount or 0),
            "status": exp.status,
            "docstatus": exp.docstatus,
        }

        return json.dumps({"status": "success", "claim": payload})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def fetch_extracted_invoice_data(reference_doctype: str, reference_name: str) -> str:
    try:
        logs = frappe.db.get_all(
            "AI Transaction Log",
            filters={
                "reference_doctype": reference_doctype,
                "reference_name": reference_name,
                "status": "Success",
                "agent": "pdf_data_extractor",
            },
            fields=["name", "output_payload"],
            order_by="creation DESC",
            limit=1,
        )

        if not logs:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No extracted invoice data found for this reference.",
                }
            )

        output = json.loads(logs[0].output_payload or "{}")
        return json.dumps({"status": "success", "invoice_data": output})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def compute_expense_variance(claim_structure: dict, invoice_data: dict) -> str:
    try:
        variances = []

        claim_dates = {
            d["travel_date"] for d in claim_structure.get("travel_information", [])
        }
        invoice_dates = set()
        for inv in invoice_data.get("invoices", []):
            if inv.get("date"):
                invoice_dates.add(str(inv.get("date")))

        date_variance = {
            "field": "travel_dates",
            "claim_dates": sorted(list(claim_dates)),
            "invoice_dates": sorted(list(invoice_dates)),
            "missing_in_invoices": sorted(list(claim_dates - invoice_dates)),
            "missing_in_claim": sorted(list(invoice_dates - claim_dates)),
            "match_ratio": len(claim_dates & invoice_dates)
            / max(len(claim_dates | invoice_dates), 1),
        }
        variances.append(date_variance)

        claim_total = claim_structure.get("claim_total", 0)
        invoice_total = sum(
            float(inv.get("total_amount", 0) or 0)
            for inv in invoice_data.get("invoices", [])
        )

        amount_variance = {
            "field": "total_amount",
            "claim_total": claim_total,
            "invoice_total": invoice_total,
            "difference": round(claim_total - invoice_total, 2),
            "variance_percentage": round(
                (abs(claim_total - invoice_total) / max(claim_total, 1)) * 100, 2
            ),
        }
        variances.append(amount_variance)

        claim_dsa = claim_structure.get("total_dsa_amount", 0)
        invoice_dsa = sum(
            float(inv.get("dsa_amount", 0) or 0)
            for inv in invoice_data.get("invoices", [])
        )
        if claim_dsa > 0 or invoice_dsa > 0:
            variances.append(
                {
                    "field": "dsa_amount",
                    "claim_dsa": claim_dsa,
                    "invoice_dsa": invoice_dsa,
                    "difference": round(claim_dsa - invoice_dsa, 2),
                }
            )

        claim_other = {
            e["type"]: e["amount"] for e in claim_structure.get("other_expenses", [])
        }
        invoice_other = {}
        for inv in invoice_data.get("invoices", []):
            for item in inv.get("line_items", []):
                item_type = item.get("type", "Unknown")
                invoice_other[item_type] = invoice_other.get(item_type, 0) + float(
                    item.get("amount", 0) or 0
                )

        all_types = set(claim_other.keys()) | set(invoice_other.keys())
        type_variances = []
        for t in all_types:
            type_variances.append(
                {
                    "type": t,
                    "claim_amount": claim_other.get(t, 0),
                    "invoice_amount": invoice_other.get(t, 0),
                    "difference": round(
                        claim_other.get(t, 0) - invoice_other.get(t, 0), 2
                    ),
                }
            )

        variances.append({"field": "expense_by_type", "details": type_variances})

        has_variance = any(
            v.get("difference", 0) != 0
            or v.get("missing_in_invoices")
            or v.get("missing_in_claim")
            for v in variances
            if isinstance(v, dict)
        )

        report = {
            "status": "success",
            "has_variance": has_variance,
            "variance_severity": (
                "high"
                if amount_variance.get("variance_percentage", 0) > 20
                else (
                    "medium"
                    if amount_variance.get("variance_percentage", 0) > 5
                    else "low"
                )
            ),
            "variances": variances,
            "recommendation": (
                "Manual review required" if has_variance else "Auto-approve eligible"
            ),
        }

        return json.dumps(report)

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def update_expense_verdict(claim_name: str, verdict: dict) -> str:
    try:
        frappe.db.set_value(
            "Employee Expense Claim",
            claim_name,
            {
                "custom_ai_verdict": verdict.get("verdict"),
                "custom_ai_risk_score": verdict.get("risk_score"),
                "custom_ai_variance_summary": json.dumps(
                    verdict.get("variances", []), default=str
                ),
                "custom_ai_scored_on": now(),
            },
        )
        frappe.db.commit()
        return json.dumps(
            {"status": "success", "message": f"Verdict updated for {claim_name}"}
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def fetch_leave_ledger_fragment(
    employee: str,
    from_date: str = None,
    lookback_years: int = 1,
    limit: int = 50,
) -> str:
    try:
        if not from_date:
            from_date = today()

        reference_date = add_years(getdate(from_date), -lookback_years)

        leaves = frappe.db.sql(
            """
            SELECT
                name,
                employee_name,
                from_date,
                to_date,
                total_leave_days,
                leave_type,
                half_day,
                custom_policy_table,
                custom_leave_reason,
                description,
                custom_short_leave,
                leave_balance,
                status,
                leave_approver_name
            FROM `tabLeave Application`
            WHERE employee = %s
              AND from_date >= %s
              AND from_date <= %s
              AND status = "Approved"
            ORDER BY from_date DESC
            LIMIT %s
        """,
            (employee, reference_date, from_date, limit),
            as_dict=True,
        )

        holiday_list = frappe.db.get_value("Employee", employee, "holiday_list")
        holidays = []
        if holiday_list:
            holidays = frappe.db.get_all(
                "Holiday",
                filters={
                    "parent": holiday_list,
                    "holiday_date": [">=", reference_date],
                },
                fields=["holiday_date", "description as holiday_reason"],
                limit=limit,
            )

        return json.dumps(
            {
                "status": "success",
                "leaves": leaves,
                "holidays": holidays,
                "lookback_from": str(reference_date),
                "lookback_to": str(from_date),
                "count": len(leaves),
            }
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def compute_leave_features(employee: str, from_date: str = None) -> str:
    try:
        if not from_date:
            from_date = today()

        reference_date = add_years(getdate(from_date), -1)

        leaves = frappe.db.sql(
            """
            SELECT
                name, from_date, to_date, total_leave_days, leave_type,
                custom_policy_table, custom_leave_reason, description,
                half_day, custom_short_leave, project
            FROM `tabLeave Application`
            WHERE employee = %s AND from_date >= %s AND from_date <= %s AND status = "Approved"
        """,
            (employee, reference_date, from_date),
            as_dict=True,
        )

        holiday_list = frappe.db.get_value("Employee", employee, "holiday_list")
        holidays = []
        if holiday_list:
            holidays = frappe.db.get_all(
                "Holiday",
                filters={
                    "parent": holiday_list,
                    "holiday_date": [">=", reference_date],
                },
                fields=["holiday_date", "description as holiday_reason"],
            )

        engineer = LeaveFeatureEngineer(leaves, holidays)
        features = engineer.to_dict()

        return json.dumps({"status": "success", "features": features})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


class LeaveFeatureEngineer:
    VAGUE_KEYWORDS = {"other", "personal", "urgent", "not well", "n/a", "na", "none"}

    def __init__(self, leaves, holidays):
        self.leaves = leaves or []
        self.holidays = holidays or []
        self._holiday_dates = self._extract_holiday_dates()

    def _extract_holiday_dates(self):
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
            try:
                return getdate(date_val)
            except Exception:
                return None
        return None

    @staticmethod
    def _is_weekend(d):
        return d.weekday() >= 5

    @staticmethod
    def _next_business_day(d):
        next_day = d + timedelta(days=1)
        while LeaveFeatureEngineer._is_weekend(next_day):
            next_day += timedelta(days=1)
        return next_day

    @staticmethod
    def _prev_business_day(d):
        prev_day = d - timedelta(days=1)
        while LeaveFeatureEngineer._is_weekend(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_policy_mode(self):
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

    def count_vague_reasons(self, mode=None):
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

    def detect_bridge_leaves(self):
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

    def calculate_cluster_features(self):
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

    def calculate_frequency_manipulation(self):
        if not self.leaves:
            return {"frequency_manipulation_count": 0}
        weekday_counts = Counter()
        for leave in self.leaves:
            fd = self._parse_date(leave.get("from_date"))
            if fd:
                weekday_counts[fd.weekday()] += 1
        flags = sum(1 for c in weekday_counts.values() if c >= 3)
        return {"frequency_manipulation_count": flags}

    def calculate_duration_manipulation(self):
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

    def calculate_type_misuse(self):
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

    def check_data_integrity(self):
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

    def to_dict(self):
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
def update_leave_verdict(leave_name: str, verdict: dict) -> str:
    try:
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
                custom_key_number = %(key_number)s,
                modified = %(now)s
            WHERE name = %(name)s
        """,
            {
                "score": verdict.get("ai_risk_score"),
                "band": verdict.get("ai_risk_band"),
                "flags": "\n".join(verdict.get("ai_rule_flags", [])),
                "model": verdict.get("ai_model", "AGENT-203"),
                "now": now(),
                "snapshot": verdict.get("ai_summary", ""),
                "verdict": verdict.get("verdict"),
                "name": leave_name,
                "key_number": verdict.get("key_number", 0) or 0,
            },
        )
        frappe.db.commit()
        frappe.publish_realtime(
            "doc_update", {"doctype": "Leave Application", "name": leave_name}
        )
        return json.dumps(
            {"status": "success", "message": f"Verdict updated for {leave_name}"}
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@frappe.whitelist()
def get_current_date() -> str:
    return json.dumps({"status": "success", "date": str(today())})


@frappe.whitelist()
def search_frappe_records(
    doctype: str, filters: dict = None, fields: list = None, limit: int = 20
) -> str:
    try:
        filters = filters or {}
        fields = fields or ["name"]
        results = frappe.db.get_all(
            doctype, filters=filters, fields=fields, limit=limit
        )
        return json.dumps(
            {"status": "success", "count": len(results), "records": results}
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
