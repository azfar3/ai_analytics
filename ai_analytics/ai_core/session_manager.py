import json
import uuid
import frappe
from frappe.utils import now


class SessionManager:

    @staticmethod
    def create_session(
        agent_name: str,
        user: str = None,
        initial_skills: list = None,
        metadata: dict = None,
    ) -> str:
        session_id = str(uuid.uuid4())

        doc = frappe.get_doc(
            {
                "doctype": "AI Session",
                "session_id": session_id,
                "agent": agent_name,
                "user": user or frappe.session.user,
                "status": "Active",
                "session_metadata": json.dumps(metadata or {}, default=str),
            }
        ).insert(ignore_permissions=True)

        if initial_skills:
            for skill_name in initial_skills:
                if frappe.db.exists("AI Skills", skill_name):
                    doc.append(
                        "active_skills", {"skill": skill_name, "status": "Pending"}
                    )
            doc.save(ignore_permissions=True)

        return doc.name

    @staticmethod
    def get_session(session_name: str):
        return frappe.get_doc("AI Session", session_name)

    @staticmethod
    def update_context(session_name: str, role: str, content: dict):
        doc = frappe.get_doc("AI Session", session_name)

        context = json.loads(doc.context_window or "[]")
        context.append({"role": role, "content": content, "timestamp": now()})

        if len(context) > 20:
            context = context[-20:]

        doc.context_window = json.dumps(context, default=str)
        doc.save(ignore_permissions=True)

        return context

    @staticmethod
    def close_session(session_name: str, status: str = "Completed"):
        """Close a session."""
        doc = frappe.get_doc("AI Session", session_name)
        doc.status = status
        doc.save(ignore_permissions=True)
        return doc.name
