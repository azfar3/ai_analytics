import json
import re
import frappe
from groq import Groq
from frappe.utils import now
from typing import Dict, List, Optional, Any

MAX_TOOL_LOOPS = 5


class FrappeAgentOrchestrator:
    def __init__(
        self,
        agent_name: str,
        reference_doctype: str,
        reference_name: str,
        session_id: str = None,
        command_name: str = None,
    ):
        if not frappe.db.exists("AI Agents", agent_name):
            raise frappe.DoesNotExistError(f"AI Agents '{agent_name}' not found.")

        self.agent = frappe.get_doc("AI Agents", agent_name)
        self.ref_doctype = reference_doctype
        self.ref_name = reference_name
        self.session_id = session_id
        self.command_name = command_name

        self.session_doc = None
        if session_id:
            try:
                self.session_doc = frappe.get_doc("AI Session", session_id)
            except frappe.DoesNotExistError:
                self.session_doc = None

        self.client = self._get_active_groq_client()
        self.tools_registry = self._load_tools_registry()
        self.skills_registry = self._load_skills_registry()
        self.log_doc = None

    def _get_active_groq_client(self) -> Groq:
        api_keys = frappe.db.get_all(
            "AI Keys",
            filters={"status": ["!=", "Expired"]},
            fields=["key", "name", "count"],
            order_by="count asc",
        )
        if not api_keys:
            frappe.throw("No active API keys found.")

        for entry in api_keys:
            key = entry.get("key")
            if key:
                try:
                    return Groq(api_key=key)
                except Exception:
                    continue
        frappe.throw("All API keys exhausted.")

    def _load_tools_registry(self) -> dict:
        registry = {}
        child_rows = self.agent.get("allowed_tools") or []

        if not child_rows:
            frappe.log_error(
                title="AI Agent No Tools",
                message=f"Agent '{self.agent.name}' has no tools configured",
            )
            return registry

        for row in child_rows:
            if not frappe.db.exists("AI Tool", row.tool):
                continue
            tool_doc = frappe.get_doc("AI Tool", row.tool)

            params_raw = getattr(tool_doc, "parameters_json", None) or "{}"
            try:
                params = json.loads(params_raw)
            except json.JSONDecodeError:
                params = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }

            registry[tool_doc.tool_name] = {
                "method_path": tool_doc.python_method,
                "description": tool_doc.description,
                "parameters": params,
            }
        return registry

    def _load_skills_registry(self) -> dict:
        registry = {}

        if self.session_doc:
            for row in self.session_doc.get("active_skills", []):
                if row.skill and frappe.db.exists("AI Skills", row.skill):
                    skill_doc = frappe.get_doc("AI Skills", row.skill)
                    registry[skill_doc.skill_name] = {
                        "description": skill_doc.description,
                        "category": skill_doc.category,
                        "tools": [t.tool for t in skill_doc.get("required_tools", [])],
                        "input_schema": skill_doc.input_schema,
                        "output_schema": skill_doc.output_schema,
                        "is_async": skill_doc.is_async,
                        "execution_order": skill_doc.execution_order,
                    }
            return registry

        available_tools = set(self.tools_registry.keys())
        all_skills = frappe.db.get_all("AI Skills", fields=["name", "skill_name"])

        for skill in all_skills:
            skill_doc = frappe.get_doc("AI Skills", skill.name)
            skill_tools = [t.tool for t in skill_doc.get("required_tools", [])]
            tool_names = []
            for tool_id in skill_tools:
                if frappe.db.exists("AI Tool", tool_id):
                    tool_name = frappe.db.get_value("AI Tool", tool_id, "tool_name")
                    tool_names.append(tool_name)

            if all(t in available_tools for t in tool_names):
                registry[skill_doc.skill_name] = {
                    "description": skill_doc.description,
                    "category": skill_doc.category,
                    "tools": tool_names,
                    "input_schema": skill_doc.input_schema,
                    "output_schema": skill_doc.output_schema,
                    "is_async": skill_doc.is_async,
                    "execution_order": skill_doc.execution_order,
                }

        return registry

    def resolve_command(self, user_input: dict) -> Optional[Any]:
        if self.command_name:
            if frappe.db.exists("AI Commands", self.command_name):
                return frappe.get_doc("AI Commands", self.command_name)
            return None

        user_text = json.dumps(user_input)
        commands = frappe.db.get_all(
            "AI Commands",
            filters={"enabled": 1},
            fields=["name", "command_name", "trigger_pattern"],
        )

        for cmd in commands:
            if cmd.trigger_pattern and re.search(
                cmd.trigger_pattern, user_text, re.IGNORECASE
            ):
                return frappe.get_doc("AI Commands", cmd.name)

        if self.ref_doctype == "Leave Application":
            if frappe.db.exists("AI Commands", "analyze_single"):
                return frappe.get_doc("AI Commands", "analyze_single")

        return None

    def execute_skill_pipeline(self, command_doc, user_input: dict) -> dict:
        skill_rows = sorted(
            command_doc.get("required_skills", []),
            key=lambda x: x.execution_order or 10,
        )

        pipeline_results = {}
        context = {"user_input": user_input, "command": command_doc.command_name}

        for row in skill_rows:
            skill_name = row.skill
            if not frappe.db.exists("AI Skills", skill_name):
                continue

            skill_doc = frappe.get_doc("AI Skills", skill_name)

            if self.session_doc:
                self._update_session_skill_status(skill_doc.skill_name, "Running")

            skill_result = self._execute_skill(skill_doc, context, pipeline_results)
            pipeline_results[skill_doc.skill_name] = skill_result

            if self.session_doc:
                status = (
                    "Completed" if skill_result.get("status") == "success" else "Failed"
                )
                self._update_session_skill_status(skill_doc.skill_name, status)

            if skill_result.get("status") != "success" and not skill_doc.is_async:
                break

        return pipeline_results

    def _execute_skill(self, skill_doc, context: dict, previous_results: dict) -> dict:
        try:
            skill_context = {
                "skill": skill_doc.skill_name,
                "category": skill_doc.category,
                "description": skill_doc.description,
                "context": context,
                "previous_results": previous_results,
            }

            if skill_doc.category == "Data Retrieval":
                return self._execute_data_retrieval_skill(skill_doc, context)

            if skill_doc.category == "Feature Engineering":
                return self._execute_feature_engineering_skill(skill_doc, context)

            return self._execute_reasoning_skill(skill_doc, skill_context)

        except Exception as e:
            return {"status": "error", "message": str(e), "skill": skill_doc.skill_name}

    def _execute_data_retrieval_skill(self, skill_doc, context: dict) -> dict:
        results = {}
        for tool_row in skill_doc.get("required_tools", []):
            tool_name = frappe.db.get_value("AI Tool", tool_row.tool, "tool_name")
            if tool_name in self.tools_registry:
                args = self._build_tool_args(tool_name, context)
                result = self._execute_tool_direct(tool_name, args)
                results[tool_name] = (
                    json.loads(result) if isinstance(result, str) else result
                )
        return {"status": "success", "data": results, "skill": skill_doc.skill_name}

    def _execute_feature_engineering_skill(self, skill_doc, context: dict) -> dict:
        for tool_row in skill_doc.get("required_tools", []):
            tool_name = frappe.db.get_value("AI Tool", tool_row.tool, "tool_name")
            if tool_name == "compute_leave_features":
                employee = context.get("user_input", {}).get("employee")
                from_date = context.get("user_input", {}).get("from_date")
                if employee:
                    result = self._execute_tool_direct(
                        tool_name, {"employee": employee, "from_date": from_date}
                    )
                    return {
                        "status": "success",
                        "features": json.loads(result),
                        "skill": skill_doc.skill_name,
                    }
        return {
            "status": "error",
            "message": "No valid tool found for feature engineering",
        }

    def _execute_reasoning_skill(self, skill_doc, skill_context: dict) -> dict:
        return {
            "status": "success",
            "context": skill_context,
            "skill": skill_doc.skill_name,
        }

    def _build_tool_args(self, tool_name: str, context: dict) -> dict:
        user_input = context.get("user_input", {})

        arg_map = {
            "fetch_leave_ledger_fragment": {
                "employee": user_input.get("employee"),
                "from_date": user_input.get("from_date"),
                "lookback_years": 1,
                "limit": 50,
            },
            "compute_leave_features": {
                "employee": user_input.get("employee"),
                "from_date": user_input.get("from_date"),
            },
            "get_current_date": {},
        }

        return arg_map.get(tool_name, {})

    def _execute_tool_direct(self, name: str, args: dict) -> str:
        if name not in self.tools_registry:
            return json.dumps(
                {"status": "error", "message": f"Tool '{name}' not available"}
            )
        try:
            method_path = self.tools_registry[name]["method_path"]
            method = frappe.get_attr(method_path)
            output = method(**args)
            if isinstance(output, str):
                return output
            return json.dumps({"status": "success", "data": output}, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def _compile_tools_spec(self) -> list:
        specs = []
        for name, meta in self.tools_registry.items():
            spec = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta.get("parameters")
                    or {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            }
            specs.append(spec)
        return specs

    def run(self, user_input_data: dict, log_name: str = None) -> dict:
        if log_name:
            self.log_doc = frappe.get_doc("AI Transaction Log", log_name)
        else:
            self.log_doc = frappe.get_doc(
                {
                    "doctype": "AI Transaction Log",
                    "agent": self.agent.name,
                    "status": "Pending",
                    "reference_doctype": self.ref_doctype,
                    "reference_name": self.ref_name,
                    "input_payload": json.dumps(user_input_data, default=str),
                }
            ).insert(ignore_permissions=True)

        command_doc = self.resolve_command(user_input_data)

        skill_results = {}
        if command_doc:
            skill_results = self.execute_skill_pipeline(command_doc, user_input_data)
            user_input_data["_skill_results"] = skill_results

        system_prompt = self._build_system_prompt(command_doc)

        clean_input = self._sanitize_input(user_input_data)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(clean_input, default=str)},
        ]

        tools_spec = self._compile_tools_spec()
        execution_trace_log = []

        try:
            loops = 0
            while loops < MAX_TOOL_LOOPS:
                loops += 1

                api_kwargs = {
                    "model": self.agent.model_version,
                    "messages": messages,
                    "temperature": self.agent.temperature or 0.1,
                    "max_completion_tokens": self.agent.max_tokens or 2048,
                }
                if tools_spec:
                    api_kwargs["tools"] = tools_spec
                    api_kwargs["tool_choice"] = "auto"

                completion = self.client.chat.completions.create(**api_kwargs)
                response_message = completion.choices[0].message

                messages.append(self._message_to_dict(response_message))

                if getattr(response_message, "tool_calls", None):
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments

                        tool_result = self._execute_tool(tool_name, tool_args)
                        execution_trace_log.append(
                            {
                                "loop": loops,
                                "tool": tool_name,
                                "args": tool_args,
                                "result": tool_result,
                                "source": "llm_tool_call",
                            }
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_result,
                            }
                        )
                    continue

                raw_response = response_message.content
                parsed_json = self._safe_parse_json(raw_response)
                usage = completion.usage

                self._finalize_log(
                    status="Success",
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    execution_trace=execution_trace_log,
                    output_payload=parsed_json,
                    skill_results=skill_results,
                )
                return parsed_json

            raise TimeoutError("Agent exceeded max tool calling loops limit.")

        except Exception as e:
            self._finalize_log(
                status="Failed",
                execution_trace=execution_trace_log,
                error=str(e),
                skill_results=skill_results,
            )
            frappe.log_error(
                title=f"AI Orchestrator Failure [{self.agent.name}]", message=str(e)
            )
            return {}

    def _build_system_prompt(self, command_doc=None) -> str:
        base_prompt = self.agent.system_prompt or ""

        if self.skills_registry:
            skills_desc = "\n## Available Skills\n"
            for name, meta in self.skills_registry.items():
                skills_desc += f"- **{name}**: {meta['description']} (Category: {meta['category']})\n"
            base_prompt += skills_desc

        if command_doc:
            base_prompt += f"\n## Active Command: {command_doc.command_name}\n{command_doc.description}\n"
            if command_doc.confirmation_required:
                base_prompt += (
                    "\nThis command requires human confirmation before execution.\n"
                )

        if self.session_doc:
            base_prompt += (
                f"\n## Session Context\nSession ID: {self.session_doc.name}\n"
            )

        return base_prompt

    def _execute_tool(self, name: str, arguments_str: str) -> str:
        if name not in self.tools_registry:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{name}' is not in the agent's allowed_tools registry.",
                }
            )
        try:
            args = json.loads(arguments_str) if arguments_str else {}
            method_path = self.tools_registry[name]["method_path"]
            method = frappe.get_attr(method_path)
            output = method(**args)
            if isinstance(output, str):
                return output
            return json.dumps({"status": "success", "data": output}, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def _update_session_skill_status(self, skill_name: str, status: str):
        if not getattr(self, "session_doc", None):  # Safe check
            return
        for row in self.session_doc.get("active_skills", []):
            if frappe.db.get_value("AI Skills", row.skill, "skill_name") == skill_name:
                row.status = status
                break
        self.session_doc.save(ignore_permissions=True)

    @staticmethod
    def _message_to_dict(msg):
        d = {"role": msg.role, "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    @staticmethod
    def _safe_parse_json(raw: str) -> dict:
        if not raw:
            return {"raw_response": "", "parsed": False}
        raw = raw.strip()
        for pattern in [
            r"```json\s*\n?([\s\S]*?)\n?\s*```",
            r"```\s*\n?([\s\S]*?)\n?\s*```",
        ]:
            m = re.search(pattern, raw)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw, "parsed": False}

    @staticmethod
    def _sanitize_input(data: dict) -> dict:
        import datetime

        def clean_value(v):
            if isinstance(v, dict):
                return {
                    k: clean_value(val)
                    for k, val in v.items()
                    if not k.startswith("_") and k not in ("session_doc", "log_doc")
                }
            elif isinstance(v, list):
                return [clean_value(item) for item in v]
            elif isinstance(v, (str, int, float, bool, type(None))):
                return v
            elif isinstance(v, datetime.date):
                return str(v)
            else:
                return str(v)

        return clean_value(data)

    def _finalize_log(self, status: str, **kwargs):
        if not self.log_doc:
            return
        self.log_doc.status = status
        if "prompt_tokens" in kwargs:
            self.log_doc.prompt_tokens = kwargs.get("prompt_tokens")
        if "completion_tokens" in kwargs:
            self.log_doc.completion_tokens = kwargs.get("completion_tokens")
        if "total_tokens" in kwargs:
            self.log_doc.total_tokens = kwargs.get("total_tokens")
        if "execution_trace" in kwargs:
            self.log_doc.execution_trace = json.dumps(
                kwargs.get("execution_trace"), indent=2, default=str
            )
        if "output_payload" in kwargs:
            self.log_doc.output_payload = json.dumps(
                kwargs.get("output_payload"), indent=2, default=str
            )
        if "error" in kwargs:
            self.log_doc.error_message = kwargs.get("error")
        self.log_doc.save(ignore_permissions=True)
        frappe.db.commit()
