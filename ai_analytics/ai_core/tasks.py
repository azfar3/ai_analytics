import json
import frappe
from frappe.utils import now, today
from ai_analytics.ai_core.orchestrator import FrappeAgentOrchestrator

QUEUE_LONG = "long"


def _update_transaction_log(log_name: str, status: str, **fields) -> None:
    try:
        frappe.db.set_value(
            "AI Transaction Log",
            log_name,
            {"status": status, "modified": now(), **fields},
        )
        frappe.db.commit()
    except Exception:
        frappe.log_error(title="AI Transaction Log Update Failed")


@frappe.whitelist()
def enqueue_agent_task(
    agent_name: str,
    reference_doctype: str,
    reference_name: str,
    user_input_data: dict,
    command_name: str = None,
    session_name: str = None,
    create_session: bool = False,
    enqueue_after_commit: bool = False,
) -> dict:

    if not frappe.db.exists("AI Agents", agent_name):
        frappe.throw(f"AI Agents '{agent_name}' not found in registry.")

    if create_session and not session_name:
        from ai_analytics.ai_core.session_manager import SessionManager

        session_name = SessionManager.create_session(
            agent_name=agent_name,
            initial_skills=user_input_data.get("skills_pipeline", []),
        )

    log_doc = frappe.get_doc(
        {
            "doctype": "AI Transaction Log",
            "agent": agent_name,
            "status": "Queued",
            "reference_doctype": reference_doctype,
            "reference_name": reference_name,
            "input_payload": json.dumps(user_input_data, default=str),
        }
    ).insert(ignore_permissions=True)

    job = frappe.enqueue(
        method="ai_analytics.ai_core.tasks._run_agent_worker",
        queue=QUEUE_LONG,
        timeout=1800,
        job_name=f"ai-agent-{agent_name}-{reference_doctype}-{reference_name}-{now()}",
        agent_name=agent_name,
        reference_doctype=reference_doctype,
        reference_name=reference_name,
        user_input_data=user_input_data,
        log_name=log_doc.name,
        command_name=command_name,
        session_name=session_name,
        enqueue_after_commit=enqueue_after_commit,
    )

    return {
        "status": "queued",
        "job_id": job.id,
        "log_name": log_doc.name,
        "agent": agent_name,
        "session": session_name,
    }


def _run_agent_worker(
    agent_name: str,
    reference_doctype: str,
    reference_name: str,
    user_input_data: dict,
    log_name: str,
    command_name: str = None,
    session_name: str = None,
) -> dict:
    frappe.db.commit()

    _update_transaction_log(log_name, "Pending")

    try:
        orchestrator = FrappeAgentOrchestrator(
            agent_name=agent_name,
            reference_doctype=reference_doctype,
            reference_name=reference_name,
            session_id=session_name,
            command_name=command_name,
        )
        result = orchestrator.run(user_input_data, log_name=log_name)

        frappe.log_error(
            title=f"AI Result Debug [{log_name}]",
            message=json.dumps(result, default=str),
        )

        if not result:
            _update_transaction_log(
                log_name,
                "Failed",
                execution_trace=json.dumps(
                    {"error": "Orchestrator returned empty result.", "trace": []}
                ),
            )
            return {"status": "failed", "log_name": log_name}

        # Manually apply the result back to the document
        if reference_doctype == "Leave Application" and isinstance(result, dict):
            from ai_analytics.ai_core.tools import update_leave_verdict

            update_res = update_leave_verdict(reference_name, result)
            frappe.log_error(title="AI DB Update Debug", message=update_res)
        elif reference_doctype == "Employee Expense Claim" and isinstance(result, dict):
            from ai_analytics.ai_core.tools import update_expense_verdict

            update_expense_verdict(reference_name, result)

        _update_transaction_log(
            log_name,
            "Success",
        )
        return {"status": "success", "result": result, "log_name": log_name}

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        _update_transaction_log(
            log_name,
            "Failed",
            execution_trace=json.dumps(
                {
                    "error": str(e),
                    "trace": tb.split("\n"),
                }
            ),
        )
        frappe.log_error(title="AI Agent Worker - Unhandled Exception", message=tb)
        return {"status": "failed", "log_name": log_name}


@frappe.whitelist()
def enqueue_leave_anomaly_agent(
    employee: str, leave_application_name: str, from_date: str
) -> dict:
    return enqueue_agent_task(
        agent_name="leave_anomaly_detector",
        reference_doctype="Leave Application",
        reference_name=leave_application_name,
        user_input_data={
            "employee": employee,
            "leave_application": leave_application_name,
            "from_date": from_date,
            "task": "evaluate_leave_patterns",
        },
        command_name="analyze_single",
    )


@frappe.whitelist()
def enqueue_expense_verification_agent(expense_claim_name: str) -> dict:
    return enqueue_agent_task(
        agent_name="expense_claim_verifier",
        reference_doctype="Employee Expense Claim",
        reference_name=expense_claim_name,
        user_input_data={
            "expense_claim": expense_claim_name,
            "task": "verify_expense_claim",
        },
    )


@frappe.whitelist()
def enqueue_pdf_extraction_agent(
    file_url: str,
    target_doctype: str = "Beneficiary Master",
    reference_name: str = None,
) -> dict:
    return enqueue_agent_task(
        agent_name="pdf_data_extractor",
        reference_doctype=target_doctype,
        reference_name=reference_name or file_url,
        user_input_data={
            "file_url": file_url,
            "target_doctype": target_doctype,
            "task": "extract_structured_data",
        },
    )


@frappe.whitelist()
def leave_application(doc, method=None):
    try:
        if isinstance(doc, str):
            if doc.strip().startswith("{"):
                doc_dict = json.loads(doc)
                doc = frappe.get_doc("Leave Application", doc_dict.get("name"))
            else:
                doc = frappe.get_doc("Leave Application", doc)
        elif isinstance(doc, dict):
            doc = frappe.get_doc("Leave Application", doc.get("name"))

        if (
            not doc
            or not getattr(doc, "employee", None)
            or not getattr(doc, "name", None)
        ):
            return

        if method is None:
            # Called manually from client script - run synchronously
            user_input_data = {
                "employee": doc.employee,
                "leave_application": doc.name,
                "from_date": str(doc.from_date) if doc.from_date else str(today()),
                "task": "evaluate_leave_patterns",
            }

            log_doc = frappe.get_doc(
                {
                    "doctype": "AI Transaction Log",
                    "agent": "leave_anomaly_detector",
                    "status": "Queued",
                    "reference_doctype": "Leave Application",
                    "reference_name": doc.name,
                    "input_payload": json.dumps(user_input_data, default=str),
                }
            ).insert(ignore_permissions=True)
            frappe.db.commit()

            result = _run_agent_worker(
                agent_name="leave_anomaly_detector",
                reference_doctype="Leave Application",
                reference_name=doc.name,
                user_input_data=user_input_data,
                log_name=log_doc.name,
                command_name="analyze_single",
            )
            return {
                "status": "success",
                "message": "Analysis completed successfully",
                "result": result,
            }
        else:
            # Called from hook - run asynchronously
            return enqueue_leave_anomaly_agent(
                employee=doc.employee,
                leave_application_name=doc.name,
                from_date=str(doc.from_date) if doc.from_date else str(today()),
            )

    except Exception:
        import traceback

        if isinstance(doc, dict):
            doc_name = doc.get("name", "Unknown")
        else:
            doc_name = getattr(
                doc,
                "name",
                str(doc)[:50] + "..." if isinstance(doc, str) else "Unknown",
            )

        frappe.log_error(
            title=f"AI after_insert Hook Error [{doc_name}]"[:140],
            message=traceback.format_exc(),
        )


def expense_claim(doc, method):
    try:
        if not doc.name:
            return

        if doc.docstatus == 1 or doc.status in ("Pending", "Submitted"):
            enqueue_expense_verification_agent(expense_claim_name=doc.name)

    except Exception:
        import traceback

        frappe.log_error(
            title=f"AI on_update Hook Error [{doc.name}]",
            message=traceback.format_exc(),
        )


def beneficiary_master(doc, method):
    try:
        file_url = None
        for f in doc.beneficiary_attachments or []:
            if f.get("title") == "Requirement Form":
                file_url = f.get("attachment")
                break

        if not file_url:
            frappe.log_error(
                title=f"AI on_submit Hook Skip [{doc.name}]",
                message="No 'Requirement Form' attachment found.",
            )
            return

        enqueue_pdf_extraction_agent(
            file_url=file_url,
            target_doctype="Beneficiary Master",
            reference_name=doc.name,
        )

    except Exception:
        import traceback

        frappe.log_error(
            title=f"AI on_submit Hook Error [{doc.name}]",
            message=traceback.format_exc(),
        )


@frappe.whitelist()
def renew_ai_keys():
    """Renew all AI keys that are not Expired at midnight."""
    keys = frappe.db.get_all("AI Keys", fields=["name"])
    for key in keys:
        frappe.db.set_value("AI Keys", key.name, {"count": 0, "status": "Active"})
    frappe.db.commit()
