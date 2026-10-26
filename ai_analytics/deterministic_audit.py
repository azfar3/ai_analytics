import frappe
from frappe.utils import flt, getdate
import hashlib
import os
import re

class DeterministicAuditEngine:
    def __init__(self, exp_name, validator):
        self.exp = frappe.get_doc("Employee Expense Claim", exp_name)
        self.validator = validator
        self.findings = []
        self.deductions = []
        self.score = 100

    def add_finding(self, rule, severity, category, description, evidence=None, values=None, deduction=0):
        if any(f["rule"] == rule for f in self.findings):
            return
            
        self.findings.append({
            "rule": rule,
            "severity": severity,
            "category": category,
            "description": description,
            "evidence": evidence or [],
            "values": values or {},
            "deterministic": True
        })
        
        if deduction > 0:
            self.deductions.append({
                "rule": rule,
                "amount": deduction,
                "reason": description
            })
            self.score -= deduction
            if self.score < 0:
                self.score = 0

    def run_audit(self):
        self.check_missing_dates()
        self.check_odometer()
        self.check_vehicle_photo()
        self.reconcile_dates()
        self.reconcile_amounts()
        self.check_dsa()
        self.detect_duplicates()
        
        # Add any findings from Phase 1 Completeness Gate (like missing evidence or early DSA/Shared transport failures)
        for val_finding in self.validator.findings:
            self.add_finding(
                rule=val_finding.get("rule", "VALIDATION_FAILED"),
                severity=val_finding.get("severity", "HIGH"),
                category="COMPLIANCE",
                description=val_finding.get("finding", ""),
                deduction=20 # Default deduction for early failures
            )
            
        for missing in self.validator.missing:
            self.add_finding(
                rule=f"MISSING_{missing['requirement'].upper()}",
                severity="HIGH",
                category="DOCUMENTATION",
                description=missing["reason"],
                deduction=20
            )

        return {
            "fact_sheet": {
                "claim": {
                    "employee": self.exp.employee,
                    "mode": self.exp.mode_of_travel,
                    "from_date": str(self.exp.from_date) if self.exp.from_date else None,
                    "to_date": str(self.exp.to_date) if self.exp.to_date else None,
                    "grand_total": flt(self.exp.grand_total)
                }
            },
            "findings": self.findings,
            "scoring": {
                "starting_score": 100,
                "deductions": self.deductions,
                "final_score": self.score
            },
            "llm_required": len(self.findings) > 0 or not self.validator.validate()["ready_for_ai"]
        }

    def check_missing_dates(self):
        if not self.exp.from_date or not self.exp.to_date:
            self.add_finding(
                rule="MISSING_CLAIM_DATES",
                severity="HIGH",
                category="COMPLIANCE",
                description="The Employee Expense Claim is missing a 'From Date' or 'To Date'.",
                deduction=20
            )

    def _get_evidence_field(self, doc_type, field_name):
        for ev in self.validator.structured_evidence:
            if ev.get("document_type") == doc_type:
                for f in ev.get("extracted_fields", []):
                    if f["field"] == field_name and f["status"] in ["EXTRACTED", "VERIFIED"]:
                        return f["value"]
        # Fallback to check across any doc_type
        for ev in self.validator.structured_evidence:
            for f in ev.get("extracted_fields", []):
                if f["field"] == field_name and f["status"] in ["EXTRACTED", "VERIFIED"]:
                    return f["value"]
        return None

    def check_odometer(self):
        if self.exp.mode_of_travel != "Personal Vehicle":
            return
            
        start_odo = self._get_evidence_field("TRAVEL_REPORT", "start_odometer")
        end_odo = self._get_evidence_field("TRAVEL_REPORT", "end_odometer")
        
        # Check text vs image discrepancy if both are present
        start_text = self._get_evidence_field("TRAVEL_REPORT", "start_odometer_text")
        start_img = self._get_evidence_field("TRAVEL_REPORT", "start_odometer_image")
        if start_text is not None and start_img is not None:
            if abs(flt(start_text) - flt(start_img)) > 5:
                self.add_finding(
                    rule="ODOMETER_TEXT_IMAGE_MISMATCH",
                    severity="HIGH",
                    category="COMPLIANCE",
                    description=f"Start odometer reading in report text ({start_text}) differs from vehicle dashboard image reading ({start_img}).",
                    deduction=25
                )

        end_text = self._get_evidence_field("TRAVEL_REPORT", "end_odometer_text")
        end_img = self._get_evidence_field("TRAVEL_REPORT", "end_odometer_image")
        if end_text is not None and end_img is not None:
            if abs(flt(end_text) - flt(end_img)) > 5:
                self.add_finding(
                    rule="ODOMETER_TEXT_IMAGE_MISMATCH",
                    severity="HIGH",
                    category="COMPLIANCE",
                    description=f"End odometer reading in report text ({end_text}) differs from vehicle dashboard image reading ({end_img}).",
                    deduction=25
                )

        if start_odo is not None and end_odo is not None:
            try:
                start_odo = flt(start_odo)
                end_odo = flt(end_odo)
                odometer_km = end_odo - start_odo
                
                claimed_km = 0
                travel_info_source = (getattr(self.exp, "group_travel_information", None) or []) if getattr(self.exp, "group_travel", 0) == 1 else (getattr(self.exp, "travel_information", None) or [])
                for row in travel_info_source:
                    claimed_km += flt(getattr(row, "km_travel", 0))
                    
                diff = abs(claimed_km - odometer_km)
                
                values = {
                    "claimed_km": claimed_km,
                    "start_odometer": start_odo,
                    "end_odometer": end_odo,
                    "odometer_km": odometer_km,
                    "difference_km": diff
                }
                
                if claimed_km > odometer_km and diff > 5:
                    self.add_finding(
                        rule="KM_MISMATCH_OVERCLAIM",
                        severity="HIGH",
                        category="FINANCIAL",
                        description=f"Claimed KM ({claimed_km}) exceeds supported odometer movement ({odometer_km}) by {diff} KM.",
                        values=values,
                        deduction=20
                    )
                elif claimed_km < odometer_km and diff > 5:
                    self.add_finding(
                        rule="KM_MISMATCH_UNDERCLAIM",
                        severity="LOW",
                        category="FINANCIAL",
                        description=f"Claimed KM ({claimed_km}) is less than supported odometer movement ({odometer_km}) by {diff} KM.",
                        values=values,
                        deduction=0
                    )
            except Exception:
                pass

    def check_vehicle_photo(self):
        if self.exp.mode_of_travel not in ["Personal Vehicle", "Rented Vehicle"]:
            return

        plate_in_photo = self._get_evidence_field("TRAVEL_REPORT", "license_plate")
        
        # Check against claim vehicle details if available
        auth_veh_no = getattr(self.exp, "vehicle_no", getattr(self.exp, "vehicle_number", None))
        if not auth_veh_no and self.validator.travel_auth and self.validator.travel_auth.get("status") == "PASS":
            auth_name = self.validator.travel_auth.get("evidence_reference")
            if auth_name and auth_name.startswith("TAR"):
                try:
                    auth = frappe.get_doc("Travel Authorisation Request Form", auth_name)
                    auth_veh_no = getattr(auth, "vehicle_no", getattr(auth, "vehicle_number", None))
                except Exception:
                    pass

        if plate_in_photo and auth_veh_no:
            norm_photo = re.sub(r'[^A-Za-z0-9]', '', str(plate_in_photo)).upper()
            norm_auth = re.sub(r'[^A-Za-z0-9]', '', str(auth_veh_no)).upper()
            if norm_photo and norm_auth and norm_photo != norm_auth:
                self.add_finding(
                    rule="VEHICLE_PLATE_MISMATCH",
                    severity="HIGH",
                    category="COMPLIANCE",
                    description=f"Vehicle license plate in photo ({plate_in_photo}) does not match authorized vehicle ({auth_veh_no}).",
                    deduction=20
                )

    def reconcile_dates(self):
        exp_from = getdate(self.exp.from_date) if self.exp.from_date else None
        exp_to = getdate(self.exp.to_date) if self.exp.to_date else None
        
        if self.validator.travel_auth and self.validator.travel_auth.get("status") == "PASS":
            auth_name = self.validator.travel_auth.get("evidence_reference")
            if auth_name and auth_name.startswith("TAR"):
                try:
                    auth = frappe.get_doc("Travel Authorisation Request Form", auth_name)
                    auth_from = None
                    auth_to = None
                    if auth.travel_information:
                        auth_from = getdate(auth.travel_information[0].from_date)
                        auth_to = getdate(auth.travel_information[-1].to_date)
                        
                    if exp_from and auth_from and exp_from < auth_from:
                        self.add_finding(
                            rule="DATE_MISMATCH_START",
                            severity="MEDIUM",
                            category="COMPLIANCE",
                            description=f"Claim start date ({exp_from}) is earlier than approved travel date ({auth_from}).",
                            deduction=20
                        )
                    if exp_to and auth_to and exp_to > auth_to:
                        self.add_finding(
                            rule="DATE_MISMATCH_END",
                            severity="MEDIUM",
                            category="COMPLIANCE",
                            description=f"Claim end date ({exp_to}) is later than approved travel date ({auth_to}).",
                            deduction=20
                        )
                except Exception:
                    pass
                
        # Check day-wise dates within bounds
        travel_info_source = (getattr(self.exp, "group_travel_information", None) or []) if getattr(self.exp, "group_travel", 0) == 1 else (getattr(self.exp, "travel_information", None) or [])
        for row in travel_info_source:
            row_date = getdate(getattr(row, "travel_date", None))
            if row_date:
                if (exp_from and row_date < exp_from) or (exp_to and row_date > exp_to):
                    self.add_finding(
                        rule="DATE_MISMATCH_DAY_WISE",
                        severity="HIGH",
                        category="COMPLIANCE",
                        description=f"Day-wise travel date {row_date} is outside claim bounds ({exp_from} to {exp_to}).",
                        deduction=15
                    )

    def reconcile_amounts(self):
        claimed_travel = flt(self.exp.total_travel_cost)
        sum_travel = 0
        travel_info_source = (getattr(self.exp, "group_travel_information", None) or []) if getattr(self.exp, "group_travel", 0) == 1 else (getattr(self.exp, "travel_information", None) or [])
        for row in travel_info_source:
            sum_travel += flt(getattr(row, "amount", getattr(row, "fare", 0)))
            
        if self.exp.mode_of_travel not in ["Personal Vehicle", "Rented Vehicle"] and abs(claimed_travel - sum_travel) > 1:
            self.add_finding(
                rule="AMOUNT_MISMATCH_TRAVEL",
                severity="MEDIUM",
                category="FINANCIAL",
                description=f"Total claimed travel cost ({claimed_travel}) does not match sum of travel rows ({sum_travel}).",
                values={"claimed": claimed_travel, "supported": sum_travel, "difference": abs(claimed_travel - sum_travel)},
                deduction=10
            )

    def check_dsa(self):
        exp_from = getdate(self.exp.from_date) if self.exp.from_date else None
        exp_to = getdate(self.exp.to_date) if self.exp.to_date else None
        total_dsa = flt(self.exp.total_dsa_amount)
        dsa_per_day = flt(self.exp.dsa_per_day)
        
        if exp_from and exp_to and total_dsa > 0 and dsa_per_day > 0:
            days = (exp_to - exp_from).days
            if days == 0:
                days = 1
            expected_dsa = days * dsa_per_day
            if abs(total_dsa - expected_dsa) > 1:
                self.add_finding(
                    rule="DSA_AMOUNT_MISMATCH",
                    severity="HIGH",
                    category="FINANCIAL",
                    description=f"Claimed DSA ({total_dsa}) does not match expected calculation ({days} days * {dsa_per_day} = {expected_dsa}).",
                    values={"claimed": total_dsa, "expected": expected_dsa, "difference": abs(total_dsa - expected_dsa)},
                    deduction=15
                )

    def detect_duplicates(self):
        hashes = {}
        
        for ev in self.validator.structured_evidence:
            url = ev.get("source_attachment")
            if not url:
                continue
                
            sha256_hash = ev.get("file_hash")
            
            # Fallback if file_hash was not precomputed in structured_evidence
            if not sha256_hash and url.startswith("/"):
                try:
                    file_name = frappe.db.get_value("File", {"file_url": url}, "name")
                    if file_name:
                        file_doc = frappe.get_doc("File", file_name)
                        file_path = file_doc.get_full_path()
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                sha256_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    pass
            
            if sha256_hash:
                filename = url.split("/")[-1]
                if sha256_hash in hashes:
                    self.add_finding(
                        rule="DUPLICATE_EVIDENCE",
                        severity="LOW",
                        category="WARNING",
                        description=f"Duplicate file detected: {filename} is identical to another attached file.",
                        evidence=[url, hashes[sha256_hash]],
                        deduction=0
                    )
                else:
                    hashes[sha256_hash] = url

