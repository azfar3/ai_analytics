import frappe
from frappe.utils import flt, getdate
import os

class ExpenseValidator:
    def __init__(self, exp_name, structured_evidence=None):
        self.exp = frappe.get_doc("Employee Expense Claim", exp_name)
        self.evidence_matrix = []
        self.findings = []
        self.warnings = []
        self.missing = []
        self.structured_evidence = structured_evidence or []
        self.travel_auth = None

    def validate(self):
        mode = self.exp.mode_of_travel

        if mode == "Personal Vehicle":
            self._validate_personal_vehicle()
        elif mode == "Rented Vehicle":
            self._validate_rented_vehicle()
        elif mode in ["Air", "Train"]:
            self._validate_air_transport()
        elif mode == "Public Transport":
            self._validate_public_transport()
        elif mode == "Shared Transport":
            self._validate_shared_transport()
            
        self._validate_dsa()

        # Completeness gate logic
        is_complete = True
        for m in self.evidence_matrix:
            if m["required"] and m["status"] in ["NOT_FOUND", "NOT_VERIFIABLE", "AMBIGUOUS", "FAIL"]:
                is_complete = False
                self.missing.append({
                    "requirement": m["requirement"],
                    "source": m["document"],
                    "reason": m["reason"] or f"{m['requirement']} could not be verified."
                })

        return {
            "ready_for_ai": is_complete,
            "status": "COMPLETE" if is_complete else "INCOMPLETE",
            "missing": self.missing,
            "warnings": self.warnings,
            "findings": self.findings,
            "evidence_matrix": self.evidence_matrix
        }

    def _add_matrix_entry(self, requirement, required, status, document, evidence_ref, extracted_facts, reason=None):
        if status == "PASS" and not evidence_ref:
            frappe.log_error(f"Internal Error: Attempted to PASS {requirement} with null evidence_reference")
            status = "FAIL"
            reason = "Internal system error: Null evidence reference for PASS"

        entry = {
            "requirement": requirement,
            "required": required,
            "status": status,
            "document": document,
            "evidence_reference": evidence_ref,
            "extracted_facts": extracted_facts or {},
            "validation": {},
            "reason": reason
        }
        self.evidence_matrix.append(entry)

    def _check_travel_auth(self):
        has_auth = bool(self.exp.travel_authorisation_request_form)
        auth_valid = False
        evidence_ref = None
        doc_name = "Travel Request Form"
        
        if has_auth:
            auth_status = frappe.db.get_value("Travel Authorisation Request Form", self.exp.travel_authorisation_request_form, "docstatus")
            auth_valid = (auth_status == 1)
            evidence_ref = self.exp.travel_authorisation_request_form
        
        if not auth_valid:
            # Check structured evidence if linked doc is missing or not submitted
            for ev in self.structured_evidence:
                if ev.get("document_type") == "TRAVEL_AUTHORIZATION":
                    has_auth = True
                    auth_valid = True
                    evidence_ref = ev.get("source_attachment")
                    doc_name = ev.get("source_attachment")
                    break
        
        if has_auth and auth_valid:
            self._add_matrix_entry("travel_authorization", True, "PASS", doc_name, evidence_ref, {}, None)
            self.travel_auth = {"status": "PASS", "evidence_reference": evidence_ref}
        else:
            self._add_matrix_entry("travel_authorization", True, "NOT_FOUND", None, None, {}, "Missing or unapproved Travel Authorization.")

    def _get_evidence_by_type(self, doc_type):
        for ev in self.structured_evidence:
            if ev.get("document_type") == doc_type:
                return ev
        return None

    def _validate_personal_vehicle(self):
        self._check_travel_auth()
        
        travel_report = self._get_evidence_by_type("TRAVEL_REPORT")
        start_odo = None
        end_odo = None
        report_src = None
        
        if travel_report:
            report_src = travel_report["source_attachment"]
            fields = {f["field"]: f for f in travel_report.get("extracted_fields", [])}
            start_odo = fields.get("start_odometer")
            end_odo = fields.get("end_odometer")

        # Fallback to check across all evidence if not found in report fields
        if not start_odo or start_odo.get("status") != "VERIFIED":
            for ev in self.structured_evidence:
                f_map = {f["field"]: f for f in ev.get("extracted_fields", [])}
                if "start_odometer" in f_map and f_map["start_odometer"].get("status") == "VERIFIED":
                    start_odo = f_map["start_odometer"]
                    if not report_src: report_src = ev.get("source_attachment")
                    break

        if not end_odo or end_odo.get("status") != "VERIFIED":
            for ev in self.structured_evidence:
                f_map = {f["field"]: f for f in ev.get("extracted_fields", [])}
                if "end_odometer" in f_map and f_map["end_odometer"].get("status") == "VERIFIED":
                    end_odo = f_map["end_odometer"]
                    if not report_src: report_src = ev.get("source_attachment")
                    break

        # Travel report document requirement
        if travel_report:
            self._add_matrix_entry("travel_report", True, "PASS", travel_report["source_attachment"], travel_report["source_attachment"], {}, None)
        else:
            self._add_matrix_entry("travel_report", True, "NOT_FOUND", None, None, {}, "Travel Report document not found.")

        # Start Odometer
        if start_odo and start_odo.get("status") == "VERIFIED":
            ref = start_odo.get("source") or report_src
            self._add_matrix_entry("start_odometer", True, "PASS", report_src, ref, {"start_odometer": start_odo}, None)
        else:
            self._add_matrix_entry("start_odometer", True, "NOT_FOUND", report_src, None, {}, "Start odometer reading missing or unverified from image.")

        # End Odometer
        if end_odo and end_odo.get("status") == "VERIFIED":
            ref = end_odo.get("source") or report_src
            self._add_matrix_entry("end_odometer", True, "PASS", report_src, ref, {"end_odometer": end_odo}, None)
        else:
            self._add_matrix_entry("end_odometer", True, "NOT_FOUND", report_src, None, {}, "End odometer reading missing or unverified from image.")

        # Vehicle Photo check
        veh_photo = None
        veh_src = None
        for ev in self.structured_evidence:
            f_map = {f["field"]: f for f in ev.get("extracted_fields", [])}
            if "vehicle_photo" in f_map and f_map["vehicle_photo"].get("status") == "VERIFIED":
                veh_photo = f_map["vehicle_photo"]
                veh_src = f_map["vehicle_photo"].get("source") or ev.get("source_attachment")
                break

        if veh_photo and veh_photo.get("status") == "VERIFIED":
            self._add_matrix_entry("vehicle_photo", True, "PASS", veh_src, veh_src, veh_photo, None)
        else:
            self._add_matrix_entry("vehicle_photo", True, "NOT_FOUND", report_src, None, {}, "Vehicle photo missing in travel report or attachments.")
        
        # GPS evidence conditional
        gps_ev = self._get_evidence_by_type("GPS_EVIDENCE")
        if gps_ev:
            self._add_matrix_entry("gps_evidence", True, "PASS", gps_ev["source_attachment"], gps_ev["source_attachment"], {}, None)
        else:
            # Check if gps flag in travel report
            if travel_report and "gps" in {f["field"]: f for f in travel_report.get("extracted_fields", [])}:
                self._add_matrix_entry("gps_evidence", True, "PASS", travel_report["source_attachment"], travel_report["source_attachment"], {}, None)
            else:
                self._add_matrix_entry("gps_evidence", True, "NOT_FOUND", None, None, {}, "GPS evidence is missing.")
        
        # Toll Receipts conditional
        has_toll_claim = any(getattr(o, "type", None) == "Toll Tax" for o in (getattr(self.exp, "other_expenses", None) or []))
        toll_ev = self._get_evidence_by_type("TOLL_RECEIPT")
        if has_toll_claim:
            if toll_ev:
                self._add_matrix_entry("toll_receipt", True, "PASS", toll_ev["source_attachment"], toll_ev["source_attachment"], {}, None)
            else:
                self._add_matrix_entry("toll_receipt", True, "NOT_FOUND", None, None, {}, "Toll claim made but no toll receipt found.")

    def _validate_rented_vehicle(self):
        self._check_travel_auth()
        
        # Add Travel Report to matrix if present
        travel_report = self._get_evidence_by_type("TRAVEL_REPORT")
        if travel_report:
            extracted = {f["field"]: f for f in travel_report.get("extracted_fields", [])}
            self._add_matrix_entry("travel_report", True, "PASS", travel_report["source_attachment"], travel_report["source_attachment"], extracted, None)
        else:
            self._add_matrix_entry("travel_report", True, "NOT_FOUND", None, None, {}, "Travel Report is missing.")

        vendor_log = self._get_evidence_by_type("VENDOR_TRAVEL_LOG")
        if vendor_log:
            self._add_matrix_entry("vendor_travel_log", True, "PASS", vendor_log["source_attachment"], vendor_log["source_attachment"], {}, None)
        else:
            # Fallback: Many times the vendor log is included in the vendor invoice
            vendor_invoice = self._get_evidence_by_type("VENDOR_INVOICE")
            if vendor_invoice:
                self._add_matrix_entry("vendor_travel_log", True, "PASS", vendor_invoice["source_attachment"], vendor_invoice["source_attachment"], {}, "Using Vendor Invoice as Vendor Travel Log fallback.")
            else:
                self._add_matrix_entry("vendor_travel_log", True, "NOT_FOUND", None, None, {}, "Vendor travel log is missing.")

        vendor_invoice = self._get_evidence_by_type("VENDOR_INVOICE")
        if vendor_invoice:
            fields = {f["field"]: f for f in vendor_invoice.get("extracted_fields", [])}
            amt = fields.get("amount")
            
            if not amt or amt["status"] != "EXTRACTED":
                vision_ev = self._get_evidence_by_type("VISION_EXTRACTION")
                if vision_ev:
                    vision_fields = {f["field"]: f for f in vision_ev.get("extracted_fields", [])}
                    vision_amt = vision_fields.get("amount")
                    if vision_amt and vision_amt["value"]:
                        amt = vision_amt

            if amt and amt["status"] == "EXTRACTED":
                self._add_matrix_entry("vendor_invoice", True, "PASS", vendor_invoice["source_attachment"], vendor_invoice["source_attachment"], {"amount": amt}, None)
            else:
                self._add_matrix_entry("vendor_invoice", True, "NOT_VERIFIABLE", vendor_invoice["source_attachment"], None, {}, "Vendor invoice found but amount could not be extracted.")
        else:
            self._add_matrix_entry("vendor_invoice", True, "NOT_FOUND", None, None, {}, "Vendor invoice is missing.")

    def _validate_air_transport(self):
        self._check_travel_auth()
        ticket = self._get_evidence_by_type("TICKET")
        if ticket:
            self._add_matrix_entry("air_ticket", True, "PASS", ticket["source_attachment"], ticket["source_attachment"], {}, None)
        else:
            self._add_matrix_entry("air_ticket", True, "NOT_FOUND", None, None, {}, "Air ticket is missing.")
            
        boarding = self._get_evidence_by_type("BOARDING_PASS")
        if boarding:
            self._add_matrix_entry("boarding_pass", False, "PASS", boarding["source_attachment"], boarding["source_attachment"], {}, None)

    def _validate_public_transport(self):
        ticket = self._get_evidence_by_type("TICKET")
        if ticket:
            self._add_matrix_entry("ticket_or_receipt", True, "PASS", ticket["source_attachment"], ticket["source_attachment"], {}, None)
        else:
            self._add_matrix_entry("ticket_or_receipt", True, "NOT_FOUND", None, None, {}, "Ticket or receipt is missing.")

    def _validate_shared_transport(self):
        travel_cost = flt(self.exp.total_travel_cost)
        other_exp = flt(self.exp.total_other_expense_amount)
        if travel_cost > 0 or other_exp > 0:
            self.findings.append({
                "rule": "SHARED_TRANSPORT_CONFLICT",
                "severity": "HIGH",
                "finding": "Shared transport selected, individual travel/other expense claims are not allowed."
            })
            self._add_matrix_entry("shared_transport_rules", True, "FAIL", "Claim Amounts", None, {}, "Prohibited individual expenses claimed under Shared Transport.")

    def _validate_dsa(self):
        from_date = getdate(self.exp.from_date) if self.exp.from_date else None
        to_date = getdate(self.exp.to_date) if self.exp.to_date else None
        total_dsa = flt(self.exp.total_dsa_amount)

        if from_date and to_date and from_date == to_date:
            if total_dsa > 0:
                self.findings.append({
                    "rule": "DSA_SINGLE_DAY",
                    "severity": "HIGH",
                    "finding": f"Single day trip ({from_date}) claimed DSA of {total_dsa}. Single day trips are not eligible for DSA."
                })
                self._add_matrix_entry("dsa_validity", True, "FAIL", "Claim Amounts", None, {}, "DSA claimed on single day trip.")
