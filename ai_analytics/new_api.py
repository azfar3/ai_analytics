import os
import io
import re
import json
import base64
import hashlib
import logging
import functools
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from collections import Counter, defaultdict
from contextlib import contextmanager

import frappe
from frappe.utils import add_years, getdate, now, today, cint, flt
from frappe.utils.background_jobs import enqueue

try:
    import fitz

    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    frappe.log_error("PyMuPDF not installed. PDF processing disabled.")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# CONFIGURATION & CONSTANTS
class AIConfig:

    # Model configurations
    DEFAULT_TEXT_MODEL = "openai/gpt-oss-20b"
    DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
    FALLBACK_MODELS = ["openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"]

    # Token & request limits
    MAX_COMPLETION_TOKENS = 1024
    MAX_PROMPT_TOKENS = 4000
    TEMPERATURE = 0.0
    TOP_P = 1.0

    # Image processing
    MAX_IMAGE_WIDTH = 800
    IMAGE_QUALITY = 50
    MAX_PDF_PAGES = 1

    # Retry & circuit breaker
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2
    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_TIMEOUT = 300

    # Caching
    CACHE_TTL_PROMPTS = 3600
    CACHE_TTL_FEATURES = 1800
    CACHE_TTL_RESPONSES = 600

    # Batch processing
    BATCH_SIZE = 10

    # Analysis thresholds
    BRIDGE_LEAVE_FLAG_THRESHOLD = 1
    FREQUENCY_MANIPULATION_THRESHOLD = 3
    DURATION_MANIPULATION_THRESHOLD = 3
    CLUSTER_MIN_SIZE = 3
    CLUSTER_WINDOW_DAYS = 30
    TYPE_MISUSE_THRESHOLD = 3


class RiskBand(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class PolicyMode(Enum):
    TYPE_BASED = "TYPE_BASED"
    REASON_BASED = "REASON_BASED"
    MIXED = "MIXED"


class AnalysisType(Enum):
    LEAVE_RISK = "leave_risk"
    FINANCE_AUDIT = "finance_audit"
    EXPENSE_VALIDATE = "expense_validate"
    SCHOOL_EXTRACT = "school_extract"
    DIMENSION_CHECK = "dimension_check"


# LOGGING & MONITORING
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"ai_analytics.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# CIRCUIT BREAKER PATTERN
class CircuitBreaker:

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, name: str):
        with cls._lock:
            if name not in cls._instances:
                instance = super().__new__(cls)
                instance.name = name
                instance.failure_count = 0
                instance.last_failure_time = None
                instance.state = "CLOSED"
                cls._instances[name] = instance
            return cls._instances[name]

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed > AIConfig.CIRCUIT_BREAKER_TIMEOUT:
                    self.state = "HALF_OPEN"
                    return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= AIConfig.CIRCUIT_BREAKER_THRESHOLD:
            self.state = "OPEN"


# CACHING UTILITIES
def cache_result(ttl_seconds: int = 1800):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [func.__module__, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            cache_key = f"ai_analytics_cache:{cache_key}"

            # Try to get from cache
            try:
                cached = frappe.cache().get_value(cache_key)
                if cached is not None:
                    return json.loads(cached)
            except Exception:
                pass

            # Execute and cache
            result = func(*args, **kwargs)
            try:
                frappe.cache().set_value(
                    cache_key,
                    json.dumps(result, default=str),
                    expires_in_sec=ttl_seconds,
                )
            except Exception:
                pass

            return result

        return wrapper

    return decorator


def invalidate_cache(pattern: str = "ai_analytics_cache:*"):
    try:
        frappe.cache().delete_keys(pattern)
    except Exception as e:
        frappe.log_error(f"Cache invalidation failed: {str(e)}")


# PROMPT MANAGEMENT
class PromptManager:
    _prompt_cache = {}
    _cache_lock = threading.Lock()

    @classmethod
    def get_prompt(cls, linked_doc: str) -> Optional[str]:
        try:
            prompt_data = frappe.db.get_value(
                "AI Prompt",
                {"linked_doc": linked_doc},
                ["prompt", "is_active"],
                as_dict=True,
            )
            if not prompt_data:
                return None
            if not prompt_data.get("is_active"):
                get_logger("PromptManager").info(
                    f"AI Prompt for {linked_doc} is marked inactive."
                )
                return None
            return prompt_data.get("prompt")
        except Exception as e:
            get_logger("PromptManager").error(
                f"Failed to fetch prompt for {linked_doc}: {e}"
            )
            return None

    @classmethod
    def format_prompt(cls, template: str, **kwargs) -> str:
        formatted = template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                # Compress large data structures
                compressed = cls._compress_value(value)
                formatted = formatted.replace(placeholder, compressed)
        return formatted

    @classmethod
    def _compress_value(cls, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if len(value) > 10:
                return cls._summarize_list(value)
            return json.dumps(value, default=str)
        elif isinstance(value, dict):
            if len(str(value)) > 2000:
                return cls._summarize_dict(value)
            return json.dumps(value, default=str)
        return str(value)

    @classmethod
    def _summarize_list(cls, items: List) -> str:
        if not items:
            return "[]"

        sample = items[:3]
        summary = {
            "count": len(items),
            "sample": sample,
            "summary": cls._calculate_summary(items),
        }
        return json.dumps(summary, default=str)

    @classmethod
    def _summarize_dict(cls, data: Dict) -> str:
        essential_keys = list(data.keys())[:10]
        condensed = {k: data[k] for k in essential_keys}
        condensed["_note"] = f"Truncated from {len(data)} keys"
        return json.dumps(condensed, default=str)

    @classmethod
    def _calculate_summary(cls, items: List) -> Dict:
        summary = {}
        if items and isinstance(items[0], dict):
            for key in items[0].keys():
                values = [item.get(key) for item in items if isinstance(item, dict)]
                numeric = [v for v in values if isinstance(v, (int, float))]
                if numeric:
                    summary[key] = {
                        "sum": sum(numeric),
                        "avg": round(sum(numeric) / len(numeric), 2),
                        "min": min(numeric),
                        "max": max(numeric),
                    }
        return summary


# AI GATEWAY - OPTIMIZED LLM INTERACTION
@dataclass
class AIResponse:
    content: Dict[str, Any]
    model_used: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    key_index: int = 0
    success: bool = True
    error: Optional[str] = None


class AIGateway:
    def __init__(self):
        self.logger = get_logger("AIGateway")
        self.circuit_breaker = CircuitBreaker("ai_gateway")
        self._key_health = {}

    def _get_api_keys(self) -> List[Dict]:
        try:
            keys = frappe.db.get_all(
                "AI Keys",
                filters={"status": ["in", ["Active", "In Waiting"]]},
                fields=["name", "key", "status", "count"],
                order_by="count asc",
            )
            return sorted(keys, key=lambda k: (k.status != "Active", k.count or 0))
        except Exception as e:
            self.logger.error(f"Failed to fetch API keys: {e}")
            return []

    def _update_key_status(
        self, key_name: str, status: str, increment_count: bool = False
    ):
        try:
            updates = {"status": status}
            if increment_count:
                current = frappe.db.get_value("AI Keys", key_name, "count") or 0
                updates["count"] = current + 1
            frappe.db.set_value("AI Keys", key_name, updates)
        except Exception as e:
            self.logger.error(f"Failed to update key {key_name}: {e}")

    def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        response_format: Optional[Dict] = None,
        images: Optional[List[str]] = None,
        temperature: float = AIConfig.TEMPERATURE,
        max_tokens: int = AIConfig.MAX_COMPLETION_TOKENS,
        reasoning_effort: Optional[str] = None,
    ) -> AIResponse:
        if not self.circuit_breaker.can_execute():
            return AIResponse(
                content={},
                model_used="",
                success=False,
                error="Circuit breaker is OPEN. Service temporarily unavailable.",
            )

        if not HAS_GROQ:
            return AIResponse(
                content={},
                model_used="",
                success=False,
                error="Groq client not installed.",
            )

        api_keys = self._get_api_keys()
        if not api_keys:
            return AIResponse(
                content={},
                model_used="",
                success=False,
                error="No active API keys available.",
            )

        model = model or AIConfig.DEFAULT_TEXT_MODEL
        response_format = response_format or {"type": "json_object"}

        content = [{"type": "text", "text": prompt}]
        if images:
            for img_str in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                    }
                )

        messages = [{"role": "user", "content": content}]

        last_error = None
        start_time = datetime.now()

        for idx, key_entry in enumerate(api_keys):
            key = key_entry.get("key")
            key_name = key_entry.get("name")

            if not key:
                continue

            try:
                client = Groq(api_key=key)

                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": response_format,
                    "max_completion_tokens": max_tokens,
                    "top_p": AIConfig.TOP_P,
                    "stream": False,
                    "stop": None,
                }

                if reasoning_effort:
                    completion_kwargs["reasoning_effort"] = reasoning_effort
                    completion_kwargs["reasoning_format"] = "hidden"

                completion = client.chat.completions.create(**completion_kwargs)

                response_data = completion.model_dump()
                raw_content = response_data["choices"][0]["message"]["content"]

                parsed_content = self._parse_json_response(raw_content)

                if not parsed_content:
                    self._update_key_status(key_name, "In Waiting")
                    self.logger.warning(f"Key {idx+1}: JSON extraction failed. Raw content: {raw_content}")
                    frappe.log_error(f"JSON Parsing Failed. Raw: {raw_content}", "LLM Parsing Error")
                    continue

                self._update_key_status(key_name, "Active", increment_count=True)
                self.circuit_breaker.record_success()

                latency = (datetime.now() - start_time).total_seconds() * 1000
                usage = response_data.get("usage", {})

                return AIResponse(
                    content=parsed_content,
                    model_used=model,
                    tokens_used=usage.get("total_tokens", 0),
                    latency_ms=latency,
                    key_index=idx + 1,
                    success=True,
                )

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                if "429" in error_str or "rate limit" in error_str:
                    self._update_key_status(key_name, "Expired")
                    self.logger.warning(f"Key {idx+1}: Rate limit (429)")
                elif "413" in error_str or "too large" in error_str:
                    self.logger.warning(f"Key {idx+1}: Request too large (413)")
                elif any(
                    code in error_str
                    for code in ["502", "503", "504", "server error", "bad gateway"]
                ):
                    self._update_key_status(key_name, "In Waiting")
                    self.logger.warning(f"Key {idx+1}: Server error")
                else:
                    self._update_key_status(key_name, "In Waiting")
                    self.logger.error(f"Key {idx+1}: Unexpected error - {e}")

                continue

        self.circuit_breaker.record_failure()
        return AIResponse(
            content={},
            model_used="",
            success=False,
            error=f"All API keys exhausted. Last error: {last_error}",
        )

    def _parse_json_response(self, raw_content: str) -> Optional[Dict]:
        if not raw_content:
            return None

        raw_content = raw_content.strip()

        def try_parse(json_str: str) -> Optional[Dict]:
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
                result = try_parse(match.group(1))
                if result:
                    return result

        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            result = try_parse(raw_content[start : end + 1])
            if result:
                return result

        if raw_content.startswith("{"):
            result = try_parse(raw_content)
            if result:
                return result

        return None


# Global gateway instance
ai_gateway = AIGateway()


# FEATURE ENGINEERING - DETERMINISTIC LAYER (ZERO LLM TOKENS)
@dataclass
class LeaveFeatures:
    policy_mode: str
    vague_reason_count: int
    vague_reason_percentage: float
    vague_reason_flag: bool
    bridge_leave_count: int
    bridge_pre_holiday: int
    bridge_post_holiday: int
    bridge_flag: bool
    cluster_leave_count: int
    frequency_manipulation_count: int
    duration_manipulation_count: int
    type_misuse_count: int
    data_integrity_flags: List[str]
    data_integrity_valid: bool
    total_leaves: int
    total_days_consumed: float
    full_day_leaves: int
    half_day_leaves: int
    short_leaves: int
    bridge_leave_full_days: int
    bridge_leave_half_days: int
    pending_applications_count: int
    remaining_leave_balance: float
    total_holidays: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FinanceSummary:
    project: str
    contract: str
    item: str
    province: Optional[str]
    current_date: date
    active_orders_count: int
    active_orders_total: float
    active_date_ranges: List[Tuple[date, date]]
    invoices_count: int
    invoices_total: float

    def to_prompt_format(self) -> str:
        return json.dumps(
            {
                "project": self.project,
                "contract": self.contract,
                "item": self.item,
                "province": self.province,
                "current_date": str(self.current_date),
                "active_orders": {
                    "count": self.active_orders_count,
                    "total_value": round(self.active_orders_total, 2),
                    "date_ranges": [f"{s} to {e}" for s, e in self.active_date_ranges],
                },
                "invoices": {
                    "count": self.invoices_count,
                    "total_value": round(self.invoices_total, 2),
                },
            }
        )


class LeaveFeatureEngineer:
    VAGUE_KEYWORDS = frozenset(
        {
            "other",
            "personal",
            "urgent",
            "not well",
            "n/a",
            "na",
            "none",
            "sick",
            "unwell",
            "family",
        }
    )

    def __init__(
        self,
        leaves: List[Dict],
        holidays: List[Dict],
        employee: Optional[str] = None,
        leave_name: Optional[str] = None,
    ):
        self.leaves = leaves or []
        self.holidays = holidays or []
        self.employee = employee
        self.leave_name = leave_name
        self._holiday_dates = self._extract_holiday_dates()
        self._parsed_leaves = self._pre_parse_leaves()

    def _extract_holiday_dates(self) -> set:
        dates = set()
        for h in self.holidays:
            hd = h.get("holiday_date")
            if hd:
                parsed = self._parse_date(hd)
                if parsed:
                    dates.add(parsed)
        return dates

    def _pre_parse_leaves(self) -> List[Dict]:
        parsed = []
        for leave in self.leaves:
            from_date = self._parse_date(leave.get("from_date"))
            to_date = self._parse_date(leave.get("to_date"))
            if from_date and to_date:
                is_half = cint(leave.get("half_day")) == 1
                is_short = cint(leave.get("custom_short_leave")) == 1
                total_days = flt(leave.get("total_leave_days"))
                if not total_days:
                    if is_short:
                        total_days = 0.25
                    elif is_half:
                        total_days = 0.5
                    else:
                        total_days = (to_date - from_date).days + 1.0

                parsed.append(
                    {
                        **leave,
                        "_from_date": from_date,
                        "_to_date": to_date,
                        "_duration": (to_date - from_date).days + 1,
                        "_is_half": is_half,
                        "_is_short": is_short,
                        "_total_days": total_days,
                    }
                )
        return parsed

    @staticmethod
    def _parse_date(date_val) -> Optional[date]:
        if isinstance(date_val, date) and not isinstance(date_val, datetime):
            return date_val
        if isinstance(date_val, str):
            try:
                return getdate(date_val)
            except Exception:
                return None
        if isinstance(date_val, datetime):
            return date_val.date()
        return None

    @staticmethod
    def _is_weekend(d: date) -> bool:
        return d.weekday() >= 5

    @classmethod
    def _next_business_day(cls, d: date) -> date:
        next_day = d + timedelta(days=1)
        while cls._is_weekend(next_day):
            next_day += timedelta(days=1)
        return next_day

    @classmethod
    def _prev_business_day(cls, d: date) -> date:
        prev_day = d - timedelta(days=1)
        while cls._is_weekend(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_policy_mode(self) -> str:
        if not self.leaves:
            return PolicyMode.TYPE_BASED.value

        reason_based_count = sum(
            1
            for l in self.leaves
            if cint(l.get("custom_policy_table")) == 1
            or l.get("project") == "ComNet - HR"
        )
        total = len(self.leaves)

        if reason_based_count == total:
            return PolicyMode.REASON_BASED.value
        elif reason_based_count == 0:
            return PolicyMode.TYPE_BASED.value
        return PolicyMode.MIXED.value

    def count_vague_reasons(
        self, mode: Optional[str] = None
    ) -> Tuple[int, float, bool]:
        if mode is None:
            mode = self.get_policy_mode()

        if mode == PolicyMode.TYPE_BASED.value:
            return 0, 0.0, False

        count = 0
        for leave in self.leaves:
            reason = str(leave.get("custom_leave_reason") or "").lower()
            desc = str(leave.get("description") or "").lower()
            combined = f"{reason} {desc}"
            if any(kw in combined for kw in self.VAGUE_KEYWORDS):
                count += 1

        total = len(self.leaves)
        percentage = round((count / total * 100), 2) if total else 0.0
        return count, percentage, count > 0

    def detect_bridge_leaves(self) -> Tuple[int, int, int, bool, int, int]:
        if not self._parsed_leaves or not self._holiday_dates:
            return 0, 0, 0, False, 0, 0

        pre_count = post_count = 0
        bridge_full = bridge_half = 0

        for leave in self._parsed_leaves:
            from_date = leave["_from_date"]
            to_date = leave["_to_date"]
            is_partial = leave["_is_half"] or leave["_is_short"]

            is_bridge = False
            day_after = self._next_business_day(to_date)
            if day_after in self._holiday_dates:
                pre_count += 1
                is_bridge = True

            day_before = self._prev_business_day(from_date)
            if day_before in self._holiday_dates:
                post_count += 1
                is_bridge = True

            if is_bridge:
                if is_partial:
                    bridge_half += 1
                else:
                    bridge_full += 1

        total = pre_count + post_count
        return (
            total,
            pre_count,
            post_count,
            total >= AIConfig.BRIDGE_LEAVE_THRESHOLD,
            bridge_full,
            bridge_half,
        )

    def calculate_cluster_features(self) -> int:
        if not self._parsed_leaves:
            return 0

        sorted_leaves = sorted(self._parsed_leaves, key=lambda l: l["_from_date"])
        cluster_count = 0
        window_days = timedelta(days=AIConfig.CLUSTER_WINDOW_DAYS)

        for i, start_leave in enumerate(sorted_leaves):
            window_end = start_leave["_from_date"] + window_days
            leaves_in_window = sum(
                1 for l in sorted_leaves[i:] if l["_from_date"] <= window_end
            )
            if leaves_in_window >= AIConfig.CLUSTER_MIN_LEAVES:
                cluster_count += 1

        return cluster_count

    def calculate_frequency_manipulation(self) -> int:
        if not self._parsed_leaves:
            return 0

        weekday_counts = Counter(l["_from_date"].weekday() for l in self._parsed_leaves)
        return sum(
            1
            for c in weekday_counts.values()
            if c >= AIConfig.FREQUENCY_MANIPULATION_THRESHOLD
        )

    def calculate_duration_manipulation(self) -> int:
        if not self._parsed_leaves:
            return 0

        patterns = 0
        duration_counts = Counter()
        half_day_count = 0

        for leave in self._parsed_leaves:
            duration_counts[leave["_duration"]] += 1
            if leave["_is_half"] or leave["_is_short"]:
                half_day_count += 1

        patterns += sum(
            1
            for c in duration_counts.values()
            if c >= AIConfig.DURATION_MANIPULATION_THRESHOLD
        )
        if half_day_count >= AIConfig.DURATION_MANIPULATION_THRESHOLD:
            patterns += 1

        return patterns

    def calculate_type_misuse(self) -> int:
        if not self._parsed_leaves:
            return 0

        reason_types = defaultdict(set)
        for leave in self.leaves:
            reason = str(leave.get("custom_leave_reason") or "").lower().strip()
            if reason:
                reason_types[reason].add(leave.get("leave_type", "Unknown"))

        return sum(
            1
            for types in reason_types.values()
            if len(types) >= AIConfig.TYPE_MISUSE_THRESHOLD
        )

    def count_pending_applications(self) -> int:
        if not self.employee:
            return 0
        filters = {"employee": self.employee, "status": "Open", "docstatus": 0}
        if self.leave_name:
            filters["name"] = ["!=", self.leave_name]
        return frappe.db.count("Leave Application", filters)

    def fetch_remaining_leave_balance(self) -> float:
        if not self.employee:
            return 0.0
        try:
            allocations = frappe.get_all(
                "Leave Allocation",
                filters={"employee": self.employee, "docstatus": 1},
                fields=["total_leaves_allocated"],
            )
            total_allocated = sum(flt(a.total_leaves_allocated) for a in allocations)
            consumed = sum(
                flt(l.get("total_leave_days", 1.0))
                for l in self.leaves
                if l.get("status") == "Approved"
            )
            return round(max(0.0, total_allocated - consumed), 2)
        except Exception:
            return 0.0

    def check_data_integrity(self) -> Tuple[List[str], bool]:
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

        return flags, len(flags) == 0

    def compute(self) -> LeaveFeatures:
        policy_mode = self.get_policy_mode()
        vague_count, vague_pct, vague_flag = self.count_vague_reasons(policy_mode)
        bridge_total, bridge_pre, bridge_post, bridge_flag, bridge_full, bridge_half = (
            self.detect_bridge_leaves()
        )
        integrity_flags, integrity_valid = self.check_data_integrity()

        total_days = sum(l.get("_total_days", 1.0) for l in self._parsed_leaves)
        half_day_count = sum(1 for l in self._parsed_leaves if l.get("_is_half"))
        short_leave_count = sum(1 for l in self._parsed_leaves if l.get("_is_short"))
        full_day_count = len(self._parsed_leaves) - half_day_count - short_leave_count

        return LeaveFeatures(
            policy_mode=policy_mode,
            vague_reason_count=vague_count,
            vague_reason_percentage=vague_pct,
            vague_reason_flag=vague_flag,
            bridge_leave_count=bridge_total,
            bridge_pre_holiday=bridge_pre,
            bridge_post_holiday=bridge_post,
            bridge_flag=bridge_flag,
            cluster_leave_count=self.calculate_cluster_features(),
            frequency_manipulation_count=self.calculate_frequency_manipulation(),
            duration_manipulation_count=self.calculate_duration_manipulation(),
            type_misuse_count=self.calculate_type_misuse(),
            data_integrity_flags=integrity_flags,
            data_integrity_valid=integrity_valid,
            total_leaves=len(self.leaves),
            total_days_consumed=round(total_days, 2),
            full_day_leaves=full_day_count,
            half_day_leaves=half_day_count,
            short_leaves=short_leave_count,
            bridge_leave_full_days=bridge_full,
            bridge_leave_half_days=bridge_half,
            pending_applications_count=self.count_pending_applications(),
            remaining_leave_balance=self.fetch_remaining_leave_balance(),
            total_holidays=len(self.holidays),
        )


class ImageProcessor:
    @staticmethod
    def process_pdf_page(pdf_content: bytes, page_num: int = 0) -> Optional[str]:
        if not HAS_FITZ or not HAS_PIL:
            return None

        try:
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")

            if page_num >= len(pdf_doc):
                pdf_doc.close()
                return None

            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_doc.close()

            # Resize if too large
            if img.width > AIConfig.MAX_IMAGE_WIDTH:
                ratio = AIConfig.MAX_IMAGE_WIDTH / img.width
                new_height = int(img.height * ratio)
                img = img.resize((AIConfig.MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)

            # Convert to grayscale then back to RGB (better compression)
            img = img.convert("L").convert("RGB")

            # Compress to JPEG
            buffer = io.BytesIO()
            img.save(
                buffer, format="JPEG", quality=AIConfig.IMAGE_QUALITY, optimize=True
            )

            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            get_logger("ImageProcessor").error(f"PDF processing failed: {e}")
            return None

    @staticmethod
    def process_pdf_pages(
        pdf_content: bytes, max_pages: int = AIConfig.MAX_PDF_PAGES
    ) -> List[str]:
        if not HAS_FITZ:
            return []

        try:
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
            pages = min(max_pages, len(pdf_doc))
            pdf_doc.close()

            images = []
            for i in range(pages):
                img = ImageProcessor.process_pdf_page(pdf_content, i)
                if img:
                    images.append(img)

            return images
        except Exception as e:
            get_logger("ImageProcessor").error(f"Multi-page PDF processing failed: {e}")
            return []


# ANALYSIS ORCHESTRATORS
class BaseAnalyzer:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.gateway = ai_gateway

    def _handle_error(self, message: str, throw: bool = False) -> Dict:
        self.logger.error(message)
        if throw:
            frappe.throw(message)
        return {"status": "failed", "error": message}

    def _validate_ai_response(
        self, response: AIResponse, required_keys: List[str]
    ) -> Optional[Dict]:
        if not response.success:
            self.logger.error(f"AI call failed: {response.error}")
            return None

        content = response.content
        missing = [k for k in required_keys if k not in content]
        if missing:
            self.logger.error(f"AI response missing keys: {missing}")
            return None

        return content


class LeaveAnalyzer(BaseAnalyzer):
    REQUIRED_AI_KEYS = ["ai_risk_score", "ai_risk_band", "ai_rule_flags", "verdict"]

    def analyze(
        self,
        doc: Optional[Any] = None,
        employee: Optional[str] = None,
        name: Optional[str] = None,
        from_date: Optional[str] = None,
        reference_years: int = 1,
    ) -> Dict:
        try:
            # If doc parameter is passed, extract properties
            if doc:
                if isinstance(doc, str):
                    doc = frappe.get_doc(
                        json.loads(doc) if doc.startswith("{") else doc
                    )
                employee = getattr(doc, "employee", employee)
                name = getattr(doc, "name", name)
                from_date = getattr(doc, "from_date", from_date)

                is_active = frappe.db.get_value(
                    "AI Prompt", {"linked_doc": doc.doctype}, "is_active"
                )
                if not is_active:
                    self.logger.info(f"AI Prompt for {doc.doctype} is inactive.")
                    return {"status": "inactive"}

            if not employee or not name:
                return self._handle_error(
                    "Employee and Leave Application name are required"
                )

            # Print debug header
            print(
                "\n=================== [AI ANALYTICS - LEAVE ANALYSIS] ==================="
            )
            print(
                f"[DEBUG] Analyzing Leave Application: {name} | Employee: {employee} | From Date: {from_date}"
            )

            # Calculate reference date
            reference_date = add_years(getdate(from_date or today()), -reference_years)
            print(f"[DEBUG] Reference Date for historical analysis: {reference_date}")

            # Fetch leave data
            leaves = self._fetch_leaves(employee, reference_date)
            holidays = self._fetch_holidays(employee, reference_date)
            print(
                f"[DEBUG] Fetched {len(leaves)} past leaves and {len(holidays)} holiday records."
            )

            # Feature engineering (deterministic, zero tokens)
            engineer = LeaveFeatureEngineer(
                leaves, holidays, employee=employee, leave_name=name
            )
            features = engineer.compute()
            print(
                f"[DEBUG] Calculated Features Snapshot:\n{json.dumps(features.to_dict(), indent=2)}"
            )

            # Get prompt template
            prompt_template = PromptManager.get_prompt("Leave Application")
            if not prompt_template:
                print(
                    "[DEBUG ERROR] Prompt template for 'Leave Application' not found or inactive!"
                )
                return self._handle_error(
                    "No prompt template found for Leave Application"
                )

            print(
                f"[DEBUG] Prompt template retrieved successfully (length: {len(prompt_template)} chars)."
            )

            # Format prompt with compressed features
            prompt = PromptManager.format_prompt(
                prompt_template,
                calculated_features=json.dumps(features.to_dict(), indent=2),
            )
            print(f"[DEBUG] Final prompt ready. Sending request to AI Gateway...")

            # AI analysis
            response = self.gateway.call(prompt=prompt)
            print(
                f"[DEBUG] AI Gateway Response: Success={response.success} | Model={response.model_used} | Key Index={response.key_index} | Tokens={response.tokens_used} | Latency={response.latency_ms:.2f}ms"
            )

            content = self._validate_ai_response(response, self.REQUIRED_AI_KEYS)

            if not content:
                print(
                    f"[DEBUG ERROR] AI analysis validation failed or returned invalid format. Error: {response.error}"
                )
                return self._handle_error(
                    "AI analysis failed or returned invalid format"
                )

            print(f"[DEBUG] AI Result Content:\n{json.dumps(content, indent=2)}")

            # Update document
            self._update_leave_document(name, content, response)
            print(
                f"[DEBUG] Successfully updated tabLeave Application document '{name}'."
            )
            print(
                "========================================================================\n"
            )

            return {
                "status": "success",
                "risk_score": content.get("ai_risk_score"),
                "risk_band": content.get("ai_risk_band"),
                "verdict": content.get("verdict"),
                "features": features.to_dict(),
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
            }

        except Exception as e:
            return self._handle_error(f"Leave analysis error: {str(e)}")

    def _fetch_leaves(self, employee: str, reference_date: date) -> List[Dict]:
        return frappe.db.sql(
            """
            SELECT
                name, employee_name, from_date, to_date, project,
                total_leave_days, leave_type, half_day, custom_policy_table,
                custom_leave_reason, description, custom_short_leave,
                leave_balance, custom_paid_days_status, status, leave_approver_name
            FROM `tabLeave Application`
            WHERE employee = %s
                AND from_date >= %s
                AND status = 'Approved'
            ORDER BY from_date DESC
        """,
            (employee, reference_date),
            as_dict=True,
        )

    def _fetch_holidays(self, employee: str, reference_date: date) -> List[Dict]:
        holiday_list = frappe.db.get_value("Employee", employee, "holiday_list")
        if not holiday_list:
            return []

        return frappe.db.get_all(
            "Holiday",
            filters={"parent": holiday_list, "holiday_date": [">=", reference_date]},
            fields=["holiday_date", "description as holiday_reason"],
        )

    def _update_leave_document(self, name: str, content: Dict, response: AIResponse):
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
                "score": content.get("ai_risk_score"),
                "band": content.get("ai_risk_band"),
                "flags": "\n".join(content.get("ai_rule_flags", [])),
                "model": "MM-FE",
                "now": now(),
                "snapshot": content.get("ai_summary", ""),
                "verdict": content.get("verdict"),
                "name": name,
                "key_number": response.key_index,
            },
        )
        frappe.db.commit()

    def batch_analyze(
        self,
        employee_leaves: List[Tuple[str, str]],
        from_date: Optional[str] = None,
        background: bool = False,
    ) -> List[Dict]:
        if background:
            job = enqueue(
                "ai_analytics.leave_batch_worker",
                employee_leaves=employee_leaves,
                from_date=from_date,
                queue="long",
                timeout=1500,
            )
            return [{"status": "queued", "job_id": job}]

        results = []
        for emp, name in employee_leaves:
            result = self.analyze(employee=emp, name=name, from_date=from_date)
            results.append(result)

        return results


class FinanceAnalyzer(BaseAnalyzer):
    def analyze(self, name: str) -> Dict:
        try:
            # Fetch metadata
            meta = self._fetch_ledger_meta(name)
            if not meta:
                return self._handle_error(f"Ledger entry {name} not found", throw=True)

            # Fetch related ledgers
            ledgers = self._fetch_related_ledgers(meta)

            # Process orders and invoices
            orders, active_dates, invoices = self._process_transactions(ledgers, meta)

            if not orders:
                return self._handle_error(
                    "No active Sales Order found for the current date and dimensions.",
                    throw=True,
                )

            # Build compressed summary
            summary = FinanceSummary(
                project=meta.project,
                contract=meta.contract,
                item=meta.sub_item,
                province=meta.province,
                current_date=meta.date,
                active_orders_count=len(orders),
                active_orders_total=sum(flt(o.amount) for o in orders),
                active_date_ranges=active_dates,
                invoices_count=len(invoices),
                invoices_total=sum(flt(i.amount) for i in invoices),
            )

            # Get prompt and call AI
            prompt_template = PromptManager.get_prompt("Project Ledger Entry")
            if not prompt_template:
                return self._handle_error(
                    "No prompt template found for Project Ledger Entry"
                )

            prompt = PromptManager.format_prompt(
                prompt_template,
                item=meta.sub_item,
                project=meta.project,
                contract=meta.contract,
                province=meta.province,
                summary=summary.to_prompt_format(),
            )

            response = self.gateway.call(prompt=prompt)

            if not response.success:
                return self._handle_error(
                    f"Finance AI analysis failed: {response.error}"
                )

            return {
                "status": "success",
                "analysis": response.content,
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
            }

        except Exception as e:
            return self._handle_error(f"Finance analysis error: {str(e)}")

    def _fetch_ledger_meta(self, name: str) -> Optional[Any]:
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
        if not data:
            return None

        # Convert to object-like access
        return type("LedgerMeta", (), data[0])()

    def _fetch_related_ledgers(self, meta: Any) -> List[Dict]:
        return frappe.db.get_all(
            "Project Ledger Entry",
            filters={
                "project": meta.project,
                "contract": meta.contract,
                "sub_item": meta.sub_item,
                "delinked": 0,
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

    def _process_transactions(
        self, ledgers: List[Dict], meta: Any
    ) -> Tuple[List[Dict], List[Tuple[date, date]], List[Dict]]:
        orders = []
        active_dates = []

        for row in ledgers:
            if row.transaction_type != "Sales Order":
                continue

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

            # Province validation
            if meta.province and so_data.province and meta.province != so_data.province:
                continue

            if (
                so_data.custom_province_required
                and not so_data.province
                and not meta.province
            ):
                frappe.throw(
                    f"Province is mandatory for Sales Order {row.transaction_name}"
                )

            # Date validation
            if (
                so_data.custom_contract_start_date
                <= meta.date
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

        # Process invoices
        invoices = []
        for row in ledgers:
            if row.transaction_type == "Sales Order":
                continue

            ledger_date = row.get("date")
            if ledger_date and any(
                start <= ledger_date <= end for start, end in active_dates
            ):
                invoices.append(row)

        return orders, active_dates, invoices


from ai_analytics.attachment_processor import UniversalAttachmentProcessor


class ExpenseAnalyzer(BaseAnalyzer):
    def analyze(self, name: str) -> Dict:
        """Synchronous entry point."""
        return self.analyze_expense_claim(name)

    def analyze_expense_claim(self, name: str) -> Dict:
        exp = None
        try:
            print(
                "\n=================== [AI ANALYTICS - EXPENSE CLAIM AUDIT] ==================="
            )
            print(
                f"[DEBUG EXPENSE] Initiating AI Audit for Employee Expense Claim: {name}"
            )
            exp = frappe.get_doc("Employee Expense Claim", name)
            start_time = datetime.now()

            # Mark as in progress if doc status fields exist
            if hasattr(exp, "ai_audit_status"):
                exp.db_set("ai_audit_status", "In Progress")

            # 1. Collect Attachments
            file_urls = []
            file_docs = frappe.get_all(
                "File",
                filters={
                    "attached_to_doctype": "Employee Expense Claim",
                    "attached_to_name": exp.name,
                },
                fields=["file_url", "file_name"]
            )
            for f in file_docs:
                if f.get("file_url"):
                    file_urls.append({"url": f.get("file_url"), "name": f.get("file_name"), "source": "Main Form"})

            if hasattr(exp, "travel_documents"):
                for d in getattr(exp, "travel_documents", None) or []:
                    file_url = getattr(d, "file", getattr(d, "file_url", getattr(d, "attachment", None)))
                    doc_type = getattr(d, "document", getattr(d, "document_type", ""))
                    comments = getattr(d, "comments", getattr(d, "description", ""))
                    if file_url:
                        file_urls.append({
                            "url": file_url, 
                            "explicit_type": f"{str(doc_type).lower()} {str(comments).lower()}"
                        })
                        
            if hasattr(exp, "supporting_bills"):
                for d in getattr(exp, "supporting_bills", None) or []:
                    file_url = getattr(d, "file", getattr(d, "file_url", getattr(d, "attachment", None)))
                    doc_type = getattr(d, "document", getattr(d, "document_type", getattr(d, "type", "")))
                    comments = getattr(d, "comments", getattr(d, "description", ""))
                    if file_url:
                        file_urls.append({
                            "url": file_url, 
                            "explicit_type": f"{str(doc_type).lower()} {str(comments).lower()}"
                        })

            print(f"[DEBUG EXPENSE] Attachment Collection | Found {len(file_urls)} total file attachments.")

            # 2. Process attachments into content-aware structured evidence + compressed base64 images (if needed)
            attachment_result = UniversalAttachmentProcessor.process_attachments(file_urls)
            images = attachment_result["images_for_vision"]
            structured_evidence = attachment_result["structured_evidence"]
            
            print(f"[DEBUG EXPENSE] Content-Aware Extraction | Extracted {len(structured_evidence)} structured evidence items. Required {len(images)} vision payload images.")

            # Phase 1: Evidence Completeness Gate
            from ai_analytics.expense_validator import ExpenseValidator
            
            # 4. Multi-Batch Map-Reduce Processing (Handling 5-Image Limit)
            batch_size = 4
            image_batches = [
                images[i : i + batch_size] for i in range(0, len(images), batch_size)
            ]
            if image_batches:
                print(f"[DEBUG EXPENSE] Batch Chunking | Total Batches to Process: {len(image_batches)} (Batch Size: {batch_size})")

            extracted_items = []
            total_tokens = 0

            for idx, batch in enumerate(image_batches, 1):
                print(f"[DEBUG EXPENSE] Executing Vision Batch #{idx}/{len(image_batches)} (Count: {len(batch)} images)...")
                mini_prompt = (
                    "Analyze each provided document or evidence image carefully. "
                    "IMPORTANT: You MUST return exactly one object in the 'items' array for EACH image provided, in the exact same order.\n"
                    "Categorize each image into one of:\n"
                    "- 'START_ODOMETER': Photo of vehicle dashboard/speedometer at trip start / initial reading.\n"
                    "- 'END_ODOMETER': Photo of vehicle dashboard/speedometer at trip end / final reading.\n"
                    "- 'ODOMETER': General vehicle dashboard/speedometer showing kilometers/mileage.\n"
                    "- 'VEHICLE_PHOTO': Exterior photo of vehicle (car, motorcycle, van, etc.) showing vehicle or registration plate.\n"
                    "- 'RECEIPT': Fuel slip, toll ticket, fare ticket, invoice, hotel bill.\n"
                    "- 'OTHER': Any other document/page.\n\n"
                    "Fields for each object:\n"
                    "{\n"
                    "  'category': 'START_ODOMETER' | 'END_ODOMETER' | 'ODOMETER' | 'VEHICLE_PHOTO' | 'RECEIPT' | 'OTHER',\n"
                    "  'odometer_reading': <numeric kilometer reading from meter dial, or null>,\n"
                    "  'vehicle_type': <e.g. 'Sedan', 'Motorcycle', 'Van' or null>,\n"
                    "  'license_plate': <visible registration number or null>,\n"
                    "  'receipt_type': 'Ticket' | 'Fuel' | 'Toll' | 'Hotel' | 'Invoice' | null,\n"
                    "  'vendor_name': <string or null>,\n"
                    "  'date': <string or null>,\n"
                    "  'total_amount': <number or null>\n"
                    "}\n"
                    "Return strict JSON with key 'items': [...]"
                )
                
                # Pass just the base64 strings to the model
                images_only = [img_dict["image_data"] for img_dict in batch]
                
                res = self.gateway.call(
                    prompt=mini_prompt,
                    model=AIConfig.DEFAULT_VISION_MODEL,
                    images=images_only,
                    temperature=0.1,
                )
                
                if res.success and isinstance(res.content, dict):
                    item_list = res.content.get("items", res.content.get("receipts", []))
                    if isinstance(item_list, list):
                        # Ensure lengths match
                        if len(item_list) < len(batch):
                            item_list.extend([{} for _ in range(len(batch) - len(item_list))])
                        elif len(item_list) > len(batch):
                            item_list = item_list[:len(batch)]
                            
                        for i, r in enumerate(item_list):
                            if r and any(r.values()):
                                r["_source_url"] = batch[i]["url"]
                                r["_source_page"] = batch[i].get("page", 1)
                                r["_source_index"] = batch[i].get("image_index", 1)
                                r["_doc_type"] = batch[i].get("doc_type", "")
                                extracted_items.append(r)
                                
                        print(f"[DEBUG EXPENSE] Vision Batch #{idx} Success | Tokens: {res.tokens_used}")
                else:
                    print(f"[DEBUG EXPENSE ERROR] Vision Batch #{idx} Failed: {res.error}")
                total_tokens += getattr(res, "tokens_used", 0)

            # Merge vision extractions into structured_evidence
            if extracted_items:
                odometer_readings = []
                vehicle_photos = []

                for r in extracted_items:
                    source_url = r.get("_source_url", "Vision API")
                    source_page = r.get("_source_page", 1)
                    source_idx = r.get("_source_index", 1)
                    source_ref = f"{source_url}#img{source_idx}" if source_idx > 1 else f"{source_url}/page:{source_page}"
                    cat = str(r.get("category", "")).upper()
                    
                    existing_ev = next((ev for ev in structured_evidence if ev.get("source_attachment") == source_url), None)
                    fields_to_add = []
                    
                    # 1. Vehicle Photo
                    if cat == "VEHICLE_PHOTO" or r.get("vehicle_type") or r.get("license_plate"):
                        veh_info = {
                            "vehicle_type": r.get("vehicle_type"),
                            "license_plate": r.get("license_plate")
                        }
                        vehicle_photos.append(veh_info)
                        fields_to_add.append({
                            "field": "vehicle_photo",
                            "value": veh_info,
                            "status": "VERIFIED",
                            "source": source_ref
                        })
                        if r.get("license_plate"):
                            fields_to_add.append({
                                "field": "license_plate",
                                "value": r.get("license_plate"),
                                "status": "VERIFIED",
                                "source": source_ref
                            })

                    # 2. Odometer Readings from Vision Dial
                    odom_val = r.get("odometer_reading") or r.get("odometer_km")
                    if odom_val:
                        try:
                            # clean string if needed
                            if isinstance(odom_val, str):
                                odom_val = re.sub(r'[^\d.]', '', odom_val)
                            odom_val = float(odom_val)
                            odometer_readings.append({
                                "value": odom_val,
                                "category": cat,
                                "source_ref": source_ref,
                                "source_url": source_url
                            })
                        except Exception:
                            pass
                        
                    # 3. Financial Receipts & Invoices
                    if r.get("total_amount"):
                        fields_to_add.append({"field": "amount", "value": r.get("total_amount"), "status": "EXTRACTED", "source": source_ref})
                    if r.get("date"):
                        fields_to_add.append({"field": "date", "value": r.get("date"), "status": "EXTRACTED", "source": source_ref})
                    if r.get("vendor_name"):
                        fields_to_add.append({"field": "vendor", "value": r.get("vendor_name"), "status": "EXTRACTED", "source": source_ref})
                        
                    if existing_ev:
                        existing_ev.setdefault("extracted_fields", []).extend(fields_to_add)
                        if existing_ev.get("extraction_status") != "EXTRACTED":
                            existing_ev["extraction_status"] = "EXTRACTED"
                    else:
                        structured_evidence.append({
                            "document_type": "VISION_EXTRACTION",
                            "extraction_status": "EXTRACTED",
                            "source_attachment": source_url,
                            "extracted_text_content": "",
                            "extracted_fields": fields_to_add
                        })

                # Resolve start and end odometer from visual readings
                if odometer_readings:
                    start_reading = next((o for o in odometer_readings if o["category"] == "START_ODOMETER"), None)
                    end_reading = next((o for o in odometer_readings if o["category"] == "END_ODOMETER"), None)
                    
                    # If not explicitly marked as START/END, sort numerically
                    if not start_reading or not end_reading:
                        sorted_readings = sorted(odometer_readings, key=lambda x: x["value"])
                        if len(sorted_readings) >= 2:
                            if not start_reading: start_reading = sorted_readings[0]
                            if not end_reading: end_reading = sorted_readings[-1]
                        elif len(sorted_readings) == 1:
                            if not start_reading: start_reading = sorted_readings[0]

                    # Target TRAVEL_REPORT evidence if present, otherwise first evidence
                    target_ev = next((ev for ev in structured_evidence if ev.get("document_type") == "TRAVEL_REPORT"), None)
                    if not target_ev and structured_evidence:
                        target_ev = structured_evidence[0]
                        
                    if target_ev and start_reading:
                        target_ev.setdefault("extracted_fields", [])
                        # Mark text field if it already existed
                        existing_start = next((f for f in target_ev["extracted_fields"] if f["field"] == "start_odometer"), None)
                        if existing_start:
                            existing_start["field"] = "start_odometer_text"
                        # Set image-verified start odometer
                        target_ev["extracted_fields"].append({
                            "field": "start_odometer",
                            "value": start_reading["value"],
                            "status": "VERIFIED",
                            "source": start_reading["source_ref"]
                        })
                        target_ev["extracted_fields"].append({
                            "field": "start_odometer_image",
                            "value": start_reading["value"],
                            "status": "VERIFIED",
                            "source": start_reading["source_ref"]
                        })

                    if target_ev and end_reading:
                        target_ev.setdefault("extracted_fields", [])
                        existing_end = next((f for f in target_ev["extracted_fields"] if f["field"] == "end_odometer"), None)
                        if existing_end:
                            existing_end["field"] = "end_odometer_text"
                        # Set image-verified end odometer
                        target_ev["extracted_fields"].append({
                            "field": "end_odometer",
                            "value": end_reading["value"],
                            "status": "VERIFIED",
                            "source": end_reading["source_ref"]
                        })
                        target_ev["extracted_fields"].append({
                            "field": "end_odometer_image",
                            "value": end_reading["value"],
                            "status": "VERIFIED",
                            "source": end_reading["source_ref"]
                        })

                    # If vehicle photo found, also link to target_ev
                    if target_ev and vehicle_photos:
                        target_ev.setdefault("extracted_fields", []).append({
                            "field": "vehicle_photo",
                            "value": vehicle_photos[0],
                            "status": "VERIFIED",
                            "source": "Vision Extraction"
                        })

            validator = ExpenseValidator(name, structured_evidence=structured_evidence)
            validation_result = validator.validate()
            
            missing_items = validation_result.get("missing", [])
            if missing_items:
                print(f"[DEBUG EXPENSE] Completeness Gate: Identified {len(missing_items)} missing requirement(s). Continuing audit across all phases without stopping.")
            else:
                print("[DEBUG EXPENSE] Completeness Gate Passed: All required evidence present. Proceeding to Deterministic Audit...")

            # Phase 2: Deterministic Audit Engine
            from ai_analytics.deterministic_audit import DeterministicAuditEngine
            audit_engine = DeterministicAuditEngine(name, validator)
            audit_result = audit_engine.run_audit()
            
            print(f"[DEBUG EXPENSE] Deterministic Audit Completed | Findings: {len(audit_result['findings'])} | Score: {audit_result['scoring']['final_score']}")

            # Calculate and save individual document scores in child table 'travel_documents'
            doc_scores = self._calculate_and_save_document_scores(
                exp=exp,
                structured_evidence=structured_evidence,
                validator=validator,
                audit_result=audit_result,
            )

            # 5. Build Phase 4 Minimal AuditFactSheet
            ambiguities = [
                field for f in structured_evidence 
                for field in f.get('extracted_fields', []) 
                if field.get('status') == 'AMBIGUOUS'
            ]
            
            fact_sheet = {
                "claim": audit_result["fact_sheet"].get("claim"),
                "evidence_matrix": validator.evidence_matrix,
                "missing_evidence": validation_result.get("missing", []),
                "present_evidence": [
                    {
                        "document_type": e.get("document_type"),
                        "url": e.get("source_attachment", ""),
                        "status": e.get("extraction_status"),
                        "fields": [
                            {
                                "field": f.get("field"),
                                "value": f.get("value"),
                                "status": f.get("status")
                            }
                            for f in e.get("extracted_fields", [])
                        ]
                    }
                    for e in structured_evidence
                ],
                "document_scores": doc_scores,
                "deterministic_findings": audit_result["findings"],
                "ambiguities": ambiguities,
                "scoring": audit_result["scoring"],
                "policy_context": [
                    "Rule: Single day trips do not qualify for DSA.",
                    "Rule: Personal vehicle claims require Travel Report with start/end odometer readings and vehicle photo.",
                    "Rule: Physical odometer dashboard photos must support the claimed distance.",
                    "Rule: Claimed KM must not exceed supported Odometer KM."
                ]
            }

            if structured_evidence:
                fact_sheet["extracted_document_text"] = [
                    {
                        "document_type": e["document_type"],
                        "url": e.get("source_attachment", e.get("url", "")),
                        "raw_text": e.get("extracted_text_content", "No text extracted")
                    } for e in structured_evidence
                ]

            # The AI is always required to cross-verify the extracted evidence (vendor amounts, travel dates) against the claim.
            # The deterministic layer (Phase 1 and 2) only ensured the files existed and checked basic logic.
            # Phase 4 (AI Synthesis) is the final auditor.
            llm_required = True

            audit_output = {}
            # Synthesize Final LLM Prompt (Text Mode Reduce)
            prompt_template = PromptManager.get_prompt("Employee Expense Claim")
            if not prompt_template:
                prompt_template = (
                    "You are the final reasoning layer of an Employee Expense Claim audit system.\n"
                    "The system has already performed deterministic validation and attachment extraction.\n"
                    "You must evaluate the entire claim thoroughly across all phases, even if certain documents are missing.\n\n"
                    "AUDIT INSTRUCTIONS:\n"
                    "1. Review the Audit Fact Sheet thoroughly.\n"
                    "2. Explicitly report what documents/evidence were MISSING and what documents/evidence were PRESENT.\n"
                    "3. For all PRESENT documents and data, evaluate whether the present data is ACCURATE or INACCURATE/CONFLICTING against the claim fields (dates, amounts, odometer readings, vehicle details).\n"
                    "4. Review the individual Document Audit Scores assigned to each attached document.\n"
                    "5. Synthesize deterministic findings and policy rules into an objective verdict.\n"
                    "6. Recommend APPROVE, QUERY, REJECT, or ESCALATE:\n"
                    "   - If required documents are missing or major data discrepancies exist, you MUST NOT recommend APPROVE.\n"
                    "   - If missing documents can be queried from the employee, recommend QUERY or REJECT with specific reasons.\n"
                    "   - Only recommend APPROVE if all required documents are present and data is verified accurate.\n\n"
                    "Return strict JSON:\n"
                    "{\n"
                    '  "overall_status": "Compliant | Partially Compliant | Non-Compliant | High-Risk Review Required",\n'
                    '  "risk_level": "Low | Medium | High",\n'
                    '  "recommendation": "APPROVE | QUERY | REJECT | ESCALATE",\n'
                    '  "missing_documents": [\n'
                    '    {"document": "Requirement or Document Name", "reason": "Why it is missing or required"}\n'
                    '  ],\n'
                    '  "present_documents": [\n'
                    '    {"document": "Document Name", "data_accurate": true, "accuracy_analysis": "Assessment of data accuracy"}\n'
                    '  ],\n'
                    '  "key_findings": ["..."],\n'
                    '  "unresolved_issues": ["..."],\n'
                    '  "executive_summary": "Comprehensive 2-4 sentence summary explaining what was missing, what was present, and whether present data was accurate."\n'
                    "}"
                )
            
            # Append Vision extractions ONLY if any exist
            if extracted_items:
                fact_sheet["vision_extractions"] = extracted_items
                
            synthesis_prompt = f"{prompt_template}\n\n## Audit Fact Sheet\n```json\n{json.dumps(fact_sheet, default=str, indent=2)}\n```"

            print(f"[DEBUG EXPENSE] Phase 4 LLM Routing: Executing Final Synthesis LLM Evaluation...")
            print(f"[DEBUG EXPENSE] ==================== LLM PROMPT ====================")
            print(synthesis_prompt)
            print(f"[DEBUG EXPENSE] ====================================================")
            
            synthesis_res = self.gateway.call(prompt=synthesis_prompt)
            total_tokens += getattr(synthesis_res, "tokens_used", 0)

            print(f"[DEBUG EXPENSE] =================== LLM RESPONSE ===================")
            print(synthesis_res.content if synthesis_res.success else synthesis_res.error)
            print(f"[DEBUG EXPENSE] ====================================================")

            audit_output = (
                synthesis_res.content
                if synthesis_res.success and isinstance(synthesis_res.content, dict)
                else self._build_deterministic_fallback_audit(
                    fact_sheet=fact_sheet,
                    error_msg=synthesis_res.error or "Unknown Error",
                )
            )

            # Store fact sheet in audit output so we can render deterministic detailed report
            audit_output["_fact_sheet"] = fact_sheet


            latency_ms = (datetime.now() - start_time).total_seconds() * 1000.0

            # 6. Save Audit Results to Custom Fields on Document
            self._update_document_audit_fields(
                exp, audit_output, total_tokens, latency_ms
            )

            print(
                f"[DEBUG EXPENSE] Audit Finished Successfully | Rating: {audit_output.get('compliance_rating')} | Risk: {audit_output.get('risk_level')} | Total Tokens: {total_tokens} | Latency: {latency_ms:.2f}ms"
            )
            print(
                "========================================================================\n"
            )

            return {
                "status": "success",
                "compliance_rating": audit_output.get("overall_status", audit_output.get("compliance_rating")),
                "risk_level": audit_output.get("risk_level"),
                "analysis": audit_output,
                "tokens_used": total_tokens,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            print(
                f"[DEBUG EXPENSE EXCEPTION] Expense analysis failed for {name}: {str(e)}"
            )
            get_logger("ExpenseAnalyzer").error(f"Expense analysis error: {e}")
            if hasattr(exp, "ai_audit_status"):
                exp.db_set("ai_audit_status", "Failed")
            return self._handle_error(f"Expense analysis error: {str(e)}")

    def _run_zero_token_pre_check(self, exp) -> Dict:
        violations = []
        warnings = []

        # Rule 1: Travel Pre-Approval Check
        if not exp.travel_authorisation_request_form:
            violations.append(
                "Non-Compliant: Travel Authorisation Request Form is missing."
            )
        else:
            auth_status = frappe.db.get_value(
                "Travel Authorisation Request Form",
                exp.travel_authorisation_request_form,
                "docstatus",
            )
            if auth_status != 1:
                violations.append(
                    "Non-Compliant: Travel Authorisation Request Form is not approved (docstatus != 1)."
                )

        # Rule 2: Single Day Trip DSA = 0
        from_date = getdate(exp.from_date) if exp.from_date else None
        to_date = getdate(exp.to_date) if exp.to_date else None
        total_dsa = flt(exp.total_dsa_amount)

        if from_date and to_date and from_date == to_date:
            if total_dsa > 0:
                violations.append(
                    f"Non-Compliant: Single day trip ({from_date}) claimed DSA of PKR {total_dsa}. Single day trips are not eligible for DSA."
                )

        # Rule 3: Shared Transport Claim Check
        if exp.mode_of_travel == "Shared Transport":
            if (
                flt(exp.total_travel_cost) > 0
                or flt(exp.total_other_expense_amount) > 0
            ):
                violations.append(
                    "Non-Compliant: Shared transport selected, travel expense claims are not allowed."
                )

        # Rule 4: Vehicle / Odometer Document Check
        if exp.mode_of_travel in ["Personal Vehicle", "Rented Vehicle"]:
            attachments = self._collect_attachment_urls(exp)
            if not attachments:
                warnings.append(
                    f"Warning: Mode of travel is '{exp.mode_of_travel}' but no odometer/toll receipts attached."
                )

        return {"violations": violations, "warnings": warnings}

    def _collect_attachment_urls(self, exp) -> List[str]:
        urls = []
        # Main form attachments
        file_docs = frappe.get_all(
            "File",
            filters={
                "attached_to_doctype": "Employee Expense Claim",
                "attached_to_name": exp.name,
            },
            fields=["file_url"],
        )
        for f in file_docs:
            if f.file_url:
                urls.append(f.file_url)

        # Child table travel_documents
        if hasattr(exp, "travel_documents"):
            for d in getattr(exp, "travel_documents", []):
                file_url = getattr(d, "file", None) or getattr(d, "file_url", None) or getattr(d, "attachment", None)
                if file_url:
                    urls.append(file_url)

        return list(set(urls))

    def _calculate_and_save_document_scores(
        self,
        exp,
        structured_evidence: List[Dict],
        validator,
        audit_result: Dict,
    ) -> List[Dict]:
        """Calculates an audit compliance score (0-100) for each document attached in the child table
        'travel_documents', updates the 'score' field on each child row in memory and in the database,
        and returns a detailed score breakdown for the audit report.
        """
        doc_scores = []
        travel_docs = getattr(exp, "travel_documents", []) or []

        # Create lookups for structured evidence
        ev_by_url = {}
        for ev in structured_evidence:
            url = ev.get("source_attachment", "")
            if url:
                ev_by_url[url] = ev
                ev_by_url[os.path.basename(url)] = ev
                ev_by_url[url.strip().lower()] = ev
                ev_by_url[os.path.basename(url).strip().lower()] = ev

        findings = audit_result.get("findings", [])

        for d in travel_docs:
            doc_type = str(getattr(d, "document", "") or "Other").strip()
            file_url = getattr(d, "file", None) or getattr(d, "file_url", None) or getattr(d, "attachment", None) or ""
            row_name = getattr(d, "name", None)

            # 1. No file attached to row
            if not file_url:
                score = 0
                status = "FAIL"
                accuracy = "Missing Document"
                notes = "Document row listed but no file attached."
                doc_scores.append({
                    "row_name": row_name,
                    "document": doc_type,
                    "file": "None",
                    "score": 0,
                    "status": status,
                    "accuracy": accuracy,
                    "notes": notes,
                })
                if hasattr(d, "score"):
                    d.score = 0
                if row_name:
                    try:
                        frappe.db.set_value(
                            "Travel Claim Documents Table",
                            row_name,
                            "score",
                            0,
                            update_modified=False,
                        )
                    except Exception as e:
                        frappe.log_error(f"Error setting doc score: {e}")
                continue

            # 2. Match structured evidence
            ev = (
                ev_by_url.get(file_url)
                or ev_by_url.get(os.path.basename(file_url))
                or ev_by_url.get(file_url.strip().lower())
                or ev_by_url.get(os.path.basename(file_url).strip().lower())
            )

            score = 100
            deductions = []
            verified_points = []
            ev_type = ev.get("document_type", "") if ev else ""

            if not ev:
                score = 20
                status = "FAIL"
                accuracy = "Unreadable / Inaccessible"
                notes = "Attached file could not be read or extracted."
            else:
                fields = {f["field"]: f for f in ev.get("extracted_fields", [])}
                raw_text = ev.get("extracted_text_content", "")

                # Travel Report
                if doc_type in ["Travel Report", "Daily Travel Log"] or ev_type == "TRAVEL_REPORT":
                    start_odo = fields.get("start_odometer") or fields.get("start_odometer_image") or fields.get("start_odometer_text")
                    end_odo = fields.get("end_odometer") or fields.get("end_odometer_image") or fields.get("end_odometer_text")

                    # If not in this single ev, check across structured_evidence for TRAVEL_REPORT
                    if not start_odo:
                        for any_ev in structured_evidence:
                            if any_ev.get("document_type") == "TRAVEL_REPORT":
                                start_odo = next((f for f in any_ev.get("extracted_fields", []) if f["field"] in ["start_odometer", "start_odometer_image"]), None)
                                if start_odo:
                                    break

                    if not end_odo:
                        for any_ev in structured_evidence:
                            if any_ev.get("document_type") == "TRAVEL_REPORT":
                                end_odo = next((f for f in any_ev.get("extracted_fields", []) if f["field"] in ["end_odometer", "end_odometer_image"]), None)
                                if end_odo:
                                    break

                    if start_odo and start_odo.get("status") in ["EXTRACTED", "VERIFIED"] and start_odo.get("value") is not None:
                        verified_points.append(f"Start odometer ({start_odo.get('value')}) verified")
                    else:
                        score -= 25
                        deductions.append("Missing start odometer (-25)")

                    if end_odo and end_odo.get("status") in ["EXTRACTED", "VERIFIED"] and end_odo.get("value") is not None:
                        verified_points.append(f"End odometer ({end_odo.get('value')}) verified")
                    else:
                        score -= 25
                        deductions.append("Missing end odometer (-25)")

                    # Check for Odometer Mismatch Finding
                    for f in findings:
                        if f.get("rule") == "ODOMETER_TEXT_IMAGE_MISMATCH":
                            score -= 20
                            deductions.append("Report odometer text differs from photo (-20)")
                            break
                        if f.get("rule") == "KM_MISMATCH_OVERCLAIM":
                            score -= 25
                            deductions.append("Claimed distance exceeds odometer distance (-25)")
                            break

                    # Check Vehicle Photo
                    veh_photo = fields.get("vehicle_photo")
                    if not veh_photo:
                        for any_ev in structured_evidence:
                            if any_ev.get("document_type") == "TRAVEL_REPORT":
                                veh_photo = next((f for f in any_ev.get("extracted_fields", []) if f["field"] == "vehicle_photo"), None)
                                if veh_photo:
                                    break

                    if exp.mode_of_travel in ["Personal Vehicle", "Rented Vehicle"]:
                        if veh_photo and veh_photo.get("status") == "VERIFIED":
                            verified_points.append("Vehicle photo verified")
                        else:
                            score -= 15
                            deductions.append("Missing vehicle photo (-15)")

                        for f in findings:
                            if f.get("rule") == "VEHICLE_PLATE_MISMATCH":
                                score -= 20
                                deductions.append("Vehicle license plate mismatch (-20)")
                                break

                # Travel Authorization
                elif doc_type in ["Travel Authorization", "Approval Email"] or ev_type == "TRAVEL_AUTHORIZATION":
                    if validator.travel_auth and validator.travel_auth.get("status") == "PASS":
                        verified_points.append("Approved Travel Authorization verified")
                    else:
                        score -= 30
                        deductions.append("Travel Authorization missing or unapproved (-30)")

                    for f in findings:
                        if f.get("rule") in ["DATE_MISMATCH_START", "DATE_MISMATCH_END"]:
                            score -= 20
                            deductions.append("Travel dates conflict with authorization (-20)")
                            break

                # Supporting Bills / Receipts / Invoices / Tickets
                elif doc_type in ["Supporting Bills", "Expense Claim"] or ev_type in ["VENDOR_INVOICE", "TOLL_RECEIPT", "FUEL_RECEIPT", "TICKET"]:
                    amt = fields.get("amount")
                    if amt and amt.get("status") in ["EXTRACTED", "VERIFIED"] and amt.get("value") is not None:
                        verified_points.append(f"Total amount (PKR {amt.get('value')}) verified")
                    else:
                        score -= 30
                        deductions.append("Receipt total amount could not be extracted (-30)")

                    inv_date = fields.get("date") or fields.get("invoice_date")
                    if inv_date and inv_date.get("value"):
                        exp_from = getdate(exp.from_date) if exp.from_date else None
                        exp_to = getdate(exp.to_date) if exp.to_date else None
                        try:
                            d_val = getdate(inv_date.get("value"))
                            if (exp_from and d_val < exp_from) or (exp_to and d_val > exp_to):
                                score -= 25
                                deductions.append(f"Receipt date ({d_val}) is outside travel period (-25)")
                            else:
                                verified_points.append(f"Receipt date ({d_val}) matches travel period")
                        except Exception:
                            pass

                # Other / General
                else:
                    if len(raw_text.strip()) > 30 or fields:
                        verified_points.append("Document content verified")
                    else:
                        score -= 25
                        deductions.append("Insufficient or unverified document content (-25)")

                # Deduct if duplicate file detected
                for f in findings:
                    if f.get("rule") == "DUPLICATE_EVIDENCE" and file_url in f.get("evidence", []):
                        score -= 25
                        deductions.append("Duplicate attachment detected (-25)")
                        break

                score = max(0, min(100, score))
                status = "PASS" if score >= 80 else ("PARTIAL" if score >= 50 else "FAIL")
                accuracy = "Accurate" if score >= 80 else ("Partially Accurate" if score >= 50 else "Inaccurate / Non-compliant")

                notes_parts = []
                if verified_points:
                    notes_parts.append("; ".join(verified_points))
                if deductions:
                    notes_parts.append("Deductions: " + "; ".join(deductions))
                notes = " | ".join(notes_parts) if notes_parts else "Document verified."

            doc_scores.append({
                "row_name": row_name,
                "document": doc_type,
                "file": file_url,
                "score": score,
                "status": status,
                "accuracy": accuracy,
                "notes": notes,
            })

            # Save to child row in memory and in DB
            if hasattr(d, "score"):
                d.score = score
            if row_name:
                try:
                    frappe.db.set_value(
                        "Travel Claim Documents Table",
                        row_name,
                        "score",
                        score,
                        update_modified=False,
                    )
                except Exception as e:
                    frappe.log_error(f"Error setting doc score: {e}")

        return doc_scores

    def _build_deterministic_fallback_audit(
        self, fact_sheet: Dict, error_msg: str = ""
    ) -> Dict:
        missing_list = fact_sheet.get("missing_evidence", [])
        det_findings = fact_sheet.get("deterministic_findings", [])
        final_score = fact_sheet.get("scoring", {}).get("final_score", 100)
        ambiguities = fact_sheet.get("ambiguities", [])

        # Determine overall status, risk, and recommendation from deterministic facts
        if missing_list or final_score < 50:
            overall_status = "Non-Compliant"
            risk_level = "High"
            recommendation = "REJECT" if final_score < 40 else "QUERY"
        elif det_findings or final_score < 80 or ambiguities:
            overall_status = "Partially Compliant"
            risk_level = "Medium"
            recommendation = "QUERY"
        else:
            overall_status = "Compliant"
            risk_level = "Low"
            recommendation = "APPROVE"

        # Synthesize policy violations / breaches directly from deterministic gaps
        violations = []
        for m in missing_list:
            req = m.get("requirement", "Document")
            reason = m.get("reason", "Required by policy")
            violations.append(f"Missing mandatory evidence: {req} - {reason}")
        for f in det_findings:
            desc = f.get("description")
            if desc and desc not in violations:
                violations.append(desc)
        for a in ambiguities:
            violations.append(f"Data ambiguity on {a.get('field')}: {a.get('reason')}")

        # Synthesize required actions for employee / approver
        actions = []
        for m in missing_list:
            actions.append(f"Provide missing documentation for: {m.get('requirement')}.")
        for f in det_findings:
            desc = f.get("description", "")
            if "missing" in desc.lower():
                actions.append(f"Submit required documentation/data: {desc}")
            else:
                actions.append(f"Resolve policy discrepancy: {desc}")
        for a in ambiguities:
            actions.append(f"Clarify conflicting data for {a.get('field')}.")

        if not actions and violations:
            actions.append("Resolve the flagged policy discrepancies before re-submitting for approval.")

        # Build narrative executive summary
        summary_parts = []
        if error_msg:
            summary_parts.append(f"AI synthesis unavailable ({error_msg}); claim evaluated via deterministic audit engine.")
        else:
            summary_parts.append("Claim evaluated via deterministic audit engine.")

        summary_parts.append(
            f"The claim received an audit score of {final_score}/100 and was rated as {overall_status} "
            f"(Risk: {risk_level}, Recommendation: {recommendation})."
        )

        if missing_list:
            missing_names = ", ".join([m.get("requirement", "Document") for m in missing_list[:3]])
            if len(missing_list) > 3:
                missing_names += f" and {len(missing_list) - 3} other(s)"
            summary_parts.append(f"Mandatory evidence missing: {missing_names}.")

        if det_findings:
            top_findings = "; ".join([f.get("description", "") for f in det_findings[:2]])
            summary_parts.append(f"Policy findings identified: {top_findings}.")

        if overall_status == "Compliant":
            summary_parts.append("All attached documents and readings verified compliant.")
        else:
            summary_parts.append("Mandatory documentation and policy discrepancies must be rectified prior to approval.")

        exec_summary = " ".join(summary_parts)

        return {
            "overall_status": overall_status,
            "compliance_rating": overall_status,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "key_findings": actions or ["No further actions required."],
            "required_actions": actions,
            "policy_violations": violations,
            "ai_findings": violations,
            "unresolved_issues": violations,
            "audit_findings": actions,
            "executive_summary": exec_summary,
        }

    def _update_document_audit_fields(
        self, exp, audit_output: Dict, tokens_used: int, latency_ms: float
    ):
        # =====================================================================
        # Word / PDF document-style audit report (formal, print-friendly)
        # =====================================================================

        # --- Shared Word-style constants ---
        SEC_H = ('font-size: 13pt; font-weight: bold; margin: 18px 0 6px 0; '
                'color: #1F3864; border-bottom: 1px solid #BFBFBF; ')
        TH = ('border: 1px solid #404040; padding: 5px 8px; text-align: left; '
            'background-color: #D9D9D9; font-weight: bold;')
        TD = 'border: 1px solid #404040; padding: 5px; vertical-align: top;'
        TBL = 'width: 100%; border-collapse: collapse; margin-bottom: 14px; font-size: 10.5pt;'

        def _badge(status):
            """Color-code PASS/FAIL like Word's conditional formatting."""
            s = str(status).upper()
            if s.startswith("PASS"):
                return f'<span style="color:#006100;font-weight:bold;">{status}</span>'
            if s.startswith("FAIL"):
                return f'<span style="color:#9C0006;font-weight:bold;">{status}</span>'
            return f"<strong>{status}</strong>"

        def _score(score):
            """Color-code score: red < 50, amber < 80, green >= 80."""
            try:
                v = float(score)
            except (TypeError, ValueError):
                return f"<strong>{score} / 100</strong>"
            color = "#9C0006" if v < 50 else ("#9C6500" if v < 80 else "#006100")
            return f'<span style="color:{color};font-weight:bold;">{score} / 100</span>'

        fact_sheet = audit_output.get("_fact_sheet", {})
        missing_list = fact_sheet.get("missing_evidence", []) if fact_sheet else []
        det_f = fact_sheet.get("deterministic_findings", []) if fact_sheet else []
        final_score = fact_sheet.get("scoring", {}).get("final_score", 100) if fact_sheet else 100

        # ---- Map Phase 4 schema to standard fields ----
        rating = audit_output.get("overall_status", audit_output.get("compliance_rating", "Compliant"))
        risk = audit_output.get("risk_level", "Low")
        recommendation = audit_output.get("recommendation", "")

        # Grounding check: never let a claim with missing evidence or score < 70 be rated Compliant
        if (missing_list or final_score < 70) and rating == "Compliant":
            rating = "Non-Compliant" if final_score < 50 else "Partially Compliant"
            risk = "High" if final_score < 50 else "Medium"
            if recommendation == "APPROVE" or not recommendation:
                recommendation = "REJECT" if final_score < 50 else "QUERY"

        # Map 'overall_status' back to Frappe standard 'compliance_rating' options if needed
        if rating == "High-Risk Review Required":
            rating = "High Risk"

        # Collect violations from all potential keys
        raw_violations = (
            audit_output.get("policy_violations")
            or audit_output.get("ai_findings")
            or audit_output.get("unresolved_issues")
            or []
        )
        if isinstance(raw_violations, str):
            violations = [raw_violations]
        else:
            violations = list(raw_violations)

        # Collect findings/actions from all potential keys
        raw_findings = (
            audit_output.get("required_actions")
            or audit_output.get("audit_findings")
            or audit_output.get("key_findings")
            or []
        )
        if isinstance(raw_findings, str):
            findings = [raw_findings]
        else:
            findings = list(raw_findings)

        # SYNCHRONIZE WITH DETERMINISTIC GROUND TRUTH:
        # Build deterministic violations and required actions from missing items and findings
        deterministic_violations = []
        for m in missing_list:
            deterministic_violations.append(
                f"Missing mandatory evidence: {m.get('requirement', 'Document')} - {m.get('reason', 'Required by policy')}"
            )
        for f in det_f:
            desc = f.get("description")
            if desc and desc not in deterministic_violations:
                deterministic_violations.append(desc)

        if not violations and deterministic_violations:
            violations = list(deterministic_violations)
        elif deterministic_violations:
            for dv in deterministic_violations:
                if not any(dv.lower() in str(v).lower() for v in violations):
                    violations.append(dv)

        # If findings is empty or only has generic failure message, derive actionable next steps
        if not findings or all("synthesis failed" in str(f).lower() or "no active api" in str(f).lower() for f in findings):
            findings = []
            for m in missing_list:
                findings.append(f"Provide missing required documentation for: {m.get('requirement', '')}.")
            for f in det_f:
                findings.append(f"Resolve policy discrepancy: {f.get('description', '')}.")

        # Narrative Executive Summary Grounding:
        doc_ver = audit_output.get("executive_summary", audit_output.get("explanation", audit_output.get("document_verification", "")))
        if not doc_ver or "system encountered an error" in str(doc_ver).lower() or "no active api" in str(doc_ver).lower():
            err_ctx = " (AI synthesis offline)" if ("error" in str(doc_ver).lower() or "api" in str(doc_ver).lower()) else ""
            summary_parts = [
                f"Deterministic audit completed with score {final_score}/100. Verdict: {rating} "
                f"(Risk: {risk}, Recommendation: {recommendation}){err_ctx}."
            ]
            if missing_list:
                missing_names = ", ".join([m.get("requirement", "Document") for m in missing_list[:3]])
                if len(missing_list) > 3:
                    missing_names += f" and {len(missing_list) - 3} other(s)"
                summary_parts.append(f"Detected {len(missing_list)} missing required evidence item(s) ({missing_names}).")
            if det_f:
                summary_parts.append(f"Flagged {len(det_f)} policy exception(s).")
            if rating == "Compliant" and not missing_list:
                summary_parts.append("All attached documents and readings verified compliant.")
            else:
                summary_parts.append("Mandatory documentation and policy discrepancies must be rectified prior to approval.")
            doc_ver = " ".join(summary_parts)

        violations_html = (
            "".join([f"<li style='margin-bottom: 4px;'>{v}</li>" for v in violations])
            if violations
            else "<li style='margin-bottom: 4px;'>No critical policy breaches detected.</li>"
        )
        findings_html = (
            "".join([f"<li style='margin-bottom: 4px;'>{f}</li>" for f in findings])
            if findings
            else "<li style='margin-bottom: 4px;'>No further actions required.</li>"
        )

        doc_scores_html = ""
        checklist_html = ""
        ambiguities_html = ""
        claim_html = ""
        m_items = "<li style='margin-bottom: 4px;'>All required documents and evidence were submitted.</li>"
        p_items = "<li style='margin-bottom: 4px;'>No documents attached.</li>"
        deterministic_items = "<li style='margin-bottom: 4px;'>No deterministic anomalies detected.</li>"

        if fact_sheet:
            # 1. Travel Document Scores Table
            doc_scores = fact_sheet.get("document_scores", [])
            if doc_scores:
                doc_scores_rows = ""
                for ds in doc_scores:
                    file_display = ds.get("file", "")
                    if file_display and file_display != "None":
                        file_display = (
                            f"<a href='{file_display}' target='_blank' "
                            f"style='color:#1F3864;text-decoration:underline;'>"
                            f"{os.path.basename(file_display)}</a>"
                        )
                    else:
                        file_display = "None"
                    doc_scores_rows += f"""
                    <tr>
                        <td style="{TD}"><strong>{ds.get('document', '')}</strong></td>
                        <td style="{TD}">{file_display}</td>
                        <td style="{TD}">{_score(ds.get('score', 0))}</td>
                        <td style="{TD}">{ds.get('accuracy', '')}</td>
                        <td style="{TD}">{ds.get('notes', '')}</td>
                    </tr>"""
                doc_scores_html = f"""
                <h4 style="{SEC_H}">4. Travel Document Scores</h4>
                <table style="{TBL}">
                    <thead>
                        <tr>
                            <th style="{TH}">Document Type</th>
                            <th style="{TH}">Attached File</th>
                            <th style="{TH}">Score</th>
                            <th style="{TH}">Data Accuracy</th>
                            <th style="{TH}">Verification Details &amp; Deductions</th>
                        </tr>
                    </thead>
                    <tbody>{doc_scores_rows}</tbody>
                </table>"""

            # 2. Missing vs Present Documents Analysis
            missing_list = fact_sheet.get("missing_evidence", [])
            if missing_list:
                m_items = "".join([
                    f"<li style='margin-bottom: 4px;'><strong>{m.get('requirement', '')}</strong>: {m.get('reason', '')}</li>"
                    for m in missing_list
                ])

            present_list = fact_sheet.get("present_evidence", [])
            if present_list:
                p_items = ""
                for p in present_list:
                    doc_t = p.get("document_type", "Document")
                    url_b = os.path.basename(p.get("url", "")) or p.get("url", "")
                    fields_str = []
                    for f in p.get("fields", []):
                        f_name = str(f.get("field", "")).replace("_", " ").title()
                        f_val = f.get("value")
                        f_st = f.get("status", "")
                        fields_str.append(f"{f_name}: <strong>{f_val}</strong> ({f_st})")
                    fields_text = " | ".join(fields_str) if fields_str else "Document attached and extracted."
                    p_items += f"<li style='margin-bottom: 4px;'><strong>{doc_t}</strong> ({url_b}): {fields_text}</li>"

            # 3. Evidence Checklist
            checklist_rows = ""
            for item in fact_sheet.get("evidence_matrix", []):
                st = item.get("status", "FAIL")
                checklist_rows += f"""
                <tr>
                    <td style="{TD}">{item.get('requirement', '')}</td>
                    <td style="{TD}">{_badge(st)}</td>
                    <td style="{TD}">{item.get('reason', '')}</td>
                </tr>"""
            if checklist_rows:
                checklist_html = f"""
                <h4 style="{SEC_H}">6. Policy Checklist Results</h4>
                <table style="{TBL}">
                    <thead>
                        <tr>
                            <th style="{TH}">Requirement</th>
                            <th style="{TH}">Status</th>
                            <th style="{TH}">Notes</th>
                        </tr>
                    </thead>
                    <tbody>{checklist_rows}</tbody>
                </table>"""

            # 4. Deterministic Findings
            det_f = fact_sheet.get("deterministic_findings", [])
            if det_f:
                deterministic_items = "".join([
                    f"<li style='margin-bottom: 4px;'>{f.get('description')}</li>" for f in det_f
                ])

            # 5. Ambiguities
            ambiguities_list = fact_sheet.get("ambiguities", [])
            if ambiguities_list:
                amb_rows = ""
                for a in ambiguities_list:
                    amb_rows += f"""
                    <tr>
                        <td style="{TD}">{a.get('field', '')}</td>
                        <td style="{TD}">{a.get('value', '')}</td>
                        <td style="{TD}" >{a.get('source', '')}</td>
                        <td style="{TD}">{a.get('reason', '')}</td>
                    </tr>"""
                ambiguities_html = f"""
                <h4 style="{SEC_H}">8. Ambiguities &amp; Discrepancies</h4>
                <table style="{TBL}">
                    <thead>
                        <tr>
                            <th style="{TH}">Field</th>
                            <th style="{TH}">Value</th>
                            <th style="{TH}">Source</th>
                            <th style="{TH}">Reason</th>
                        </tr>
                    </thead>
                    <tbody>{amb_rows}</tbody>
                </table>"""
            else:
                ambiguities_html = f"""
                <h4 style="{SEC_H}">8. Ambiguities &amp; Discrepancies</h4>
                <p style="margin: 0 0 14px 0; text-align: justify;">No ambiguities or data conflicts detected.</p>"""

            # 6. Claim Overview
            claim_data = fact_sheet.get('claim', {})
            if claim_data:
                claim_rows = ""
                for k, v in claim_data.items():
                    formatted_key = str(k).replace('_', ' ').title()
                    val = "Not Provided" if v is None else str(v)
                    claim_rows += f"""
                    <tr>
                        <td style="{TD}; width: 30%;"><strong>{formatted_key}</strong></td>
                        <td style="{TD}">{val}</td>
                    </tr>"""
                claim_html = f"""
                <h4 style="{SEC_H}">9. Claim Overview</h4>
                <table style="{TBL}">
                    <thead>
                        <tr>
                            <th style="{TH}">Property</th>
                            <th style="{TH}">Value</th>
                        </tr>
                    </thead>
                    <tbody>{claim_rows}</tbody>
                </table>"""

        # ---- Missing-documents notice (Word callout box, not a UI card) ----
        missing_list = fact_sheet.get("missing_evidence", []) if fact_sheet else []
        missing_note = ""
        if missing_list:
            missing_note = f"""
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 14px;">
                <tr>
                    <td style="border: 1px solid #C00000; padding: 8px 12px; color: #9C0006; font-size: 10.5pt;">
                        <strong>Notice ({len(missing_list)} Missing Requirement(s)):</strong>
                        One or more required documents were not attached. See Section 5 for details.
                    </td>
                </tr>
            </table>"""

        # ---- Document header: label/value table (Word-style title block) ----
        rec_row = ""
        if recommendation:
            rec_row = (
                '<tr><td style="padding: 2px 0; width: 140px;">'
                '<strong>Recommendation:</strong></td>'
                f'<td style="padding: 2px 0;">{recommendation}</td></tr>'
            )
        report_date = str(now())[:16]

        summary_doc = f"""
        <div style="font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 11pt;
                    line-height: 1.5; color: #000000; padding: 24px 8px;
                    max-width: 800px; margin: 0 auto; text-align: left !important;">

            <!-- Document Header -->
            <div style="border-bottom: 3px double #000000; padding-bottom: 10px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0; font-size: 20pt; font-weight: bold; color: #1F3864;
                        letter-spacing: 0.3px; text-transform: uppercase;
                        font-family: 'Calibri Light', 'Segoe UI', Arial, sans-serif;">
                    Employee Expense Claim Audit Report
                </h2>
                <table style="width: 100%; border-collapse: collapse; font-size: 10.5pt; color: #333333;">
                    <tr>
                        <td style="padding: 2px; width: 140px;"><strong>Compliance Rating:</strong></td>
                        <td style="padding: 2px;">{rating}</td>
                    </tr>
                    <tr>
                        <td style="padding: 2px;"><strong>Risk Level:</strong></td>
                        <td style="padding: 2px;">{risk}</td>
                    </tr>
                    {rec_row}
                </table>
            </div>

            {missing_note}

            <h4 style="{SEC_H}">1. Executive Summary</h4>
            <p style="margin: 0 0 14px 0; text-align: justify;">{doc_ver}</p>

            <h4 style="{SEC_H}">2. Policy Breaches &amp; Violations</h4>
            <ul style="margin: 0 0 14px 0; padding-left: 28px; text-align: justify;">
                {violations_html}
            </ul>

            <h4 style="{SEC_H}">3. Audit Findings &amp; Required Actions</h4>
            <ul style="margin: 0 0 14px 0; padding-left: 28px; text-align: justify;">
                {findings_html}
            </ul>

            {doc_scores_html}

            <h4 style="{SEC_H}">5. Document Completeness &amp; Data Verification</h4>
            <p style="margin: 0 0 4px 0;"><strong>Missing Documents:</strong></p>
            <ul style="margin: 0 0 10px 0; padding-left: 28px;">
                {m_items}
            </ul>
            <p style="margin: 0 0 4px 0;"><strong>Present Documents &amp; Extracted Data:</strong></p>
            <ul style="margin: 0 0 14px 0; padding-left: 28px;">
                {p_items}
            </ul>

            {checklist_html}

            <h4 style="{SEC_H}">7. Deterministic System Findings</h4>
            <ul style="margin: 0 0 14px 0; padding-left: 28px; text-align: justify;">
                {deterministic_items}
            </ul>

            {ambiguities_html}

            {claim_html}

            <!-- Document Footer -->
            <div style="border-top: 1px solid #BFBFBF; margin-top: 28px; padding-top: 8px;
                        font-size: 9pt; color: #595959; text-align: center;">
                Employee Expense Claim Audit Report &mdash; Generated {report_date} &mdash; Confidential
            </div>
        </div>
        """

        # ---- Field updates (unchanged) ----
        updates = {}
        if hasattr(exp, "ai_audit_status"):
            updates["ai_audit_status"] = "Completed"
        if hasattr(exp, "compliance_rating"):
            updates["compliance_rating"] = rating
        if hasattr(exp, "risk_level"):
            updates["risk_level"] = risk
        if hasattr(exp, "tokens_used"):
            updates["tokens_used"] = tokens_used
        if hasattr(exp, "processing_time_ms"):
            updates["processing_time_ms"] = latency_ms
        if hasattr(exp, "audit_completed_on"):
            updates["audit_completed_on"] = now()
        if hasattr(exp, "audit_summary"):
            updates["audit_summary"] = summary_doc
        if hasattr(exp, "raw_audit_json"):
            updates["raw_audit_json"] = json.dumps(audit_output, indent=2, default=str)

        if updates:
            exp.db_set(updates)

    def _extract_expense_data(self, exp) -> Dict:
        data = {
            "employee": exp.employee,
            "mode_of_travel": exp.mode_of_travel,
            "vehicle_type": getattr(exp, "vehicle_type", None),
            "ticket_amount": flt(getattr(exp, "ticket_amount", 0)),
            "from_date": (
                str(exp.from_date) if getattr(exp, "from_date", None) else None
            ),
            "to_date": str(exp.to_date) if getattr(exp, "to_date", None) else None,
            "dsa_per_day": flt(getattr(exp, "dsa_per_day", 0)),
            "total_dsa_amount": flt(getattr(exp, "total_dsa_amount", 0)),
            "total_other_expense_amount": flt(
                getattr(exp, "total_other_expense_amount", 0)
            ),
            "total_travel_cost": flt(getattr(exp, "total_travel_cost", 0)),
            "grand_total": flt(getattr(exp, "grand_total", 0)),
            "travel_authorization": [],
            "travel_information": [],
            "other_expenses": [],
        }

        if getattr(exp, "travel_authorisation_request_form", None):
            travel_form = frappe.get_doc(
                "Travel Authorisation Request Form",
                exp.travel_authorisation_request_form,
            )
            data["travel_authorization"].append(
                {
                    "employee": travel_form.employee,
                    "status": travel_form.docstatus,
                    "purpose": getattr(travel_form, "purpose_of_travel", ""),
                    "details": [
                        {
                            "from_date": str(d.from_date),
                            "to_date": str(d.to_date),
                            "source": getattr(d, "source", ""),
                            "destination": getattr(d, "destination", ""),
                        }
                        for d in getattr(travel_form, "travel_information", [])
                    ],
                }
            )

        travel_info_source = (
            getattr(exp, "group_travel_information", [])
            if getattr(exp, "group_travel", 0) == 1
            else getattr(exp, "travel_information", [])
        )
        data["travel_information"] = [
            {
                "date": str(getattr(info, "travel_date", "")),
                "source": getattr(info, "source", ""),
                "destination": getattr(info, "destination", ""),
                "km": flt(getattr(info, "km_travel", 0)),
            }
            for info in travel_info_source
        ]

        data["other_expenses"] = [
            {
                "type": getattr(o, "type", ""),
                "amount": flt(getattr(o, "amount", 0)),
                "pay_to_employee": getattr(o, "pay_to_employee", 0),
            }
            for o in getattr(exp, "other_expenses", [])
        ]

        return data

    def _build_prompt(self, template: str, data: Dict) -> str:
        return f"{template}\n\n## Input Data\n```json\n{json.dumps(data, default=str, indent=2)}\n```"


class SchoolDataExtractor(BaseAnalyzer):
    def extract(
        self,
        name: Optional[str] = None,
        file: Optional[str] = None,
        cr_name: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Dict:
        try:
            print(
                "\n=================== [AI ANALYTICS - SCHOOL DATA EXTRACTION] ==================="
            )
            print(
                f"[DEBUG] Processing Extraction Request | Name: {name} | File: {file} | District Override: {district}"
            )

            # Resolve file URL
            file_url = self._resolve_file_url(name, file)
            if not file_url:
                print("[DEBUG ERROR] No attachment found for processing.")
                return self._handle_error("No attachment found", throw=True)

            print(f"[DEBUG] Attachment File URL resolved: {file_url}")

            # Process PDF (First page only as requested to minimize vision load)
            images = self._process_pdf(file_url)
            if not images:
                print(
                    "[DEBUG ERROR] Failed to convert PDF to compressed vision images."
                )
                return self._handle_error("Failed to process PDF", throw=True)

            print(
                f"[DEBUG] Successfully rendered PDF first page into compressed vision image (Count: {len(images)})."
            )

            # Get prompt
            prompt = PromptManager.get_prompt("Beneficiary Master")
            if not prompt:
                err_msg = "AI Prompt for 'Beneficiary Master' is inactive or not found."
                print(f"[DEBUG INACTIVE] {err_msg} Skipping extraction quietly.")
                frappe.log_error(
                    message=err_msg,
                    title="AI Analytics - School Data Extraction Inactive",
                )
                return {"status": "inactive", "message": err_msg}

            print(
                f"[DEBUG] Prompt template retrieved successfully (length: {len(prompt)} chars). Sending Vision AI request..."
            )

            # Vision AI call
            response = self.gateway.call(
                prompt=prompt,
                model=AIConfig.DEFAULT_VISION_MODEL,
                images=images,
                temperature=0.1,
                reasoning_effort="none",
            )

            print(
                f"[DEBUG] Vision AI Gateway Response: Success={response.success} | Model={response.model_used} | Key Index={response.key_index} | Tokens={response.tokens_used} | Latency={response.latency_ms:.2f}ms"
            )

            if not response.success:
                print(f"[DEBUG ERROR] Vision AI Gateway call failed: {response.error}")
                return self._handle_error(
                    f"Vision AI failed: {response.error}", throw=True
                )

            data = response.content
            if not data or not data.get("parent_data"):
                print(
                    f"[DEBUG ERROR] Vision AI returned invalid JSON structure: {data}"
                )
                return self._handle_error("Invalid AI response structure", throw=True)

            print(
                f"[DEBUG] Extracted Raw Parent Data:\n{json.dumps(data.get('parent_data', {}), indent=2)}"
            )
            print(
                f"[DEBUG] Extracted Requirements Count: {len(data.get('requirements', []))}"
            )

            # Save data with enhanced validation, fuzzy location matching, and deterministic cost checks
            if name:
                result = self._update_existing(name, data)
            else:
                result = self._create_new(data, cr_name, district)

            print(
                f"[DEBUG] Extraction process completed successfully. Output: {result}"
            )
            print(
                "========================================================================\n"
            )

            return {**result, "tokens_used": response.tokens_used}

        except Exception as e:
            print(f"[DEBUG EXCEPTION] School extraction failed: {str(e)}")
            return self._handle_error(f"School extraction error: {str(e)}")

    def _resolve_file_url(
        self, name: Optional[str], file: Optional[str]
    ) -> Optional[str]:
        if file:
            return file

        if name:
            beneficiary = frappe.get_doc("Beneficiary Master", name)
            for f in beneficiary.beneficiary_attachments:
                if f.get("title") == "Requirement Form":
                    return f.get("attachment")

        return None

    def _process_pdf(self, file_url: str) -> List[str]:
        file_name = frappe.db.get_value("File", {"file_url": file_url}, "name")
        if not file_name:
            return []
        file_doc = frappe.get_doc("File", file_name)
        file_path = file_doc.get_full_path()

        with open(file_path, "rb") as f:
            content = f.read()

        # Enforce rendering page 1 only to minimize token overhead
        return ImageProcessor.process_pdf_pages(content, max_pages=1)

    def _match_location(self, doctype: str, raw_name: Optional[str]) -> Optional[str]:
        """Fuzzy & exact database matching for District, Tehsil, and UC Link fields."""
        if not raw_name or not str(raw_name).strip():
            return None

        clean_raw = str(raw_name).strip()

        # 1. Exact match check
        if frappe.db.exists(doctype, clean_raw):
            return clean_raw

        # 2. Case-insensitive / LIKE match check
        matched = frappe.db.get_value(
            doctype, {"name": ["like", f"%{clean_raw}%"]}, "name"
        )
        if matched:
            print(
                f"[DEBUG FUZZY LOCATION] Matched '{clean_raw}' -> '{matched}' for {doctype}"
            )
            return matched

        # 3. Word-token similarity fallback
        first_word = clean_raw.split()[0] if clean_raw.split() else clean_raw
        fallback_matched = frappe.db.get_value(
            doctype, {"name": ["like", f"%{first_word}%"]}, "name"
        )
        if fallback_matched:
            print(
                f"[DEBUG FUZZY LOCATION] Fallback matched '{clean_raw}' -> '{fallback_matched}' for {doctype}"
            )
            return fallback_matched

        return clean_raw

    def _validate_fields(
        self, emis_code: Optional[str], bank_account: Optional[str]
    ) -> bool:
        """Validates critical EMIS code (max 5 digits) and Bank Account (IBAN format)."""
        valid_emis = (
            bool(re.match(r"^\d{1,5}$", str(emis_code).strip())) if emis_code else True
        )

        valid_bank = True
        if bank_account:
            try:
                from comnet_hr.comnet_hr.page.iban_validator.iban_validator import (
                    IBANValidator,
                )

                valid_bank, msg, _, _ = IBANValidator.validate_iban(bank_account)
            except Exception:
                valid_bank = bool(
                    re.match(r"^[A-Za-z0-9\-]{8,34}$", str(bank_account).strip())
                )

        if not valid_emis or not valid_bank:
            print(
                f"[DEBUG VALIDATION WARNING] Format mismatch - EMIS: '{emis_code}' (Valid max 5-digit: {valid_emis}) | Bank Acc: '{bank_account}' (Valid IBAN: {valid_bank})"
            )
            return False
        return True

    def _calculate_requirement_costs(self, req: Dict) -> Tuple[float, float]:
        """Deterministic zero-token math calculation for requirement unit and total costs."""
        unit_cost = flt(req.get("unit_cost") or 0)
        total_cost = flt(req.get("total_cost") or 0)
        qty = flt(req.get("quantity") or 1)

        # Re-verify total cost mathematically
        calculated_total = round(unit_cost * qty, 2)
        if total_cost == 0 and calculated_total > 0:
            total_cost = calculated_total
        elif abs(total_cost - calculated_total) > 1.0 and unit_cost > 0:
            print(
                f"[DEBUG ZERO-TOKEN MATH] Auto-corrected total_cost for '{req.get('facility')}' from {total_cost} to {calculated_total}"
            )
            total_cost = calculated_total

        return unit_cost, total_cost

    def _update_existing(self, name: str, data: Dict) -> Dict:
        parent = data.get("parent_data", {})
        doc = frappe.get_doc("Beneficiary Master", name)

        # Fuzzy location resolution
        matched_district = (
            self._match_location("District", parent.get("district")) or doc.district
        )
        matched_tehsil = (
            self._match_location("Tehsil", parent.get("tehsil")) or doc.tehsil
        )
        matched_uc = self._match_location("UC", parent.get("uc")) or doc.uc

        # Field validation
        is_valid = self._validate_fields(
            parent.get("emis_code"), parent.get("bank_account")
        )
        if not is_valid:
            doc.eligibility_status = "Requires Clarification"

        # Update fields with fallback
        updates = {
            "beneficiary_name": parent.get("school_name"),
            "district": matched_district,
            "tehsil": matched_tehsil,
            "uc": matched_uc,
            "ref_code": parent.get("emis_code"),
            "school_level": parent.get("school_level"),
            "school_gender": parent.get("gender"),
            "total_existing_class_rooms": parent.get("class_rooms"),
            "total_working_teachers": parent.get("total_teachers"),
            "total_enrollment": parent.get("total_enrollment"),
            "acc_number": parent.get("bank_account"),
            "branch_name": parent.get("bank_city"),
        }

        for field, value in updates.items():
            if value is not None:
                setattr(doc, field, value)

        # Date handling
        parsed_date = self._safe_date(parent.get("submission_date"))
        if parsed_date:
            doc.sip_submission_date = parsed_date

        # Requirements table with deterministic zero-token math
        doc.set("requirements_table", [])
        for req in data.get("requirements", []):
            unit_cost, total_cost = self._calculate_requirement_costs(req)
            doc.append(
                "requirements_table",
                {
                    "facility_name": req.get("facility", ""),
                    "current_status": req.get("status", ""),
                    "requirements": req.get("requirement", ""),
                    "unit_cost": unit_cost,
                    "total_cost": total_cost,
                    "priority": cint(req.get("priority")),
                    "responsibility_assigned": req.get("responsibility", ""),
                    "timeline": "2026-06-30",
                },
            )

        doc.save()
        frappe.db.commit()

        return {"status": "success", "name": doc.name, "action": "updated"}

    def _create_new(
        self, data: Dict, cr_name: Optional[str], district: Optional[str]
    ) -> Dict:
        parent = data.get("parent_data", {})

        # Normalize gender
        gender = parent.get("gender")
        if gender in ("F", "Girls"):
            gender = "Female"
        elif gender in ("M", "Boys"):
            gender = "Male"

        if parent.get("school_name", "").lower().startswith("ggps"):
            gender = "Female"

        # Normalize school level
        school_level = parent.get("school_level") or "Primary"

        # Fuzzy location resolution
        resolved_district = self._match_location(
            "District", district or parent.get("district")
        )
        resolved_tehsil = self._match_location("Tehsil", parent.get("tehsil"))
        resolved_uc = self._match_location("UC", parent.get("uc"))

        # Field validation
        is_valid = self._validate_fields(
            parent.get("emis_code"), parent.get("bank_account")
        )
        eligibility = "Pending" if is_valid else "Requires Clarification"

        doc = frappe.new_doc("Beneficiary Master")
        doc.province = 3
        doc.project = "Financial Support to PTCs for SDPs - KP - 43459411"
        doc.contract_name = "Financial Support to PTCs for SDPs - KP - 43459411"
        doc.beneficiary_name = parent.get("school_name")
        doc.beneficiary_type = "School"
        doc.ref_code = parent.get("emis_code")
        doc.district = resolved_district
        doc.tehsil = resolved_tehsil
        doc.uc = resolved_uc
        doc.school_level = school_level
        doc.school_gender = gender
        doc.total_existing_class_rooms = parent.get("class_rooms")
        doc.total_working_teachers = parent.get("total_teachers")
        doc.total_enrollment = parent.get("total_enrollment")
        doc.eligibility_status = eligibility
        doc.approved_amount = 1296187
        doc.payment_rule = "Installments"
        doc.num_installements = 2
        doc.payment_mode = "Online Transfer"
        doc.acc_number = parent.get("bank_account")
        doc.bank_name = parent.get("bank_name")
        doc.branch_name = parent.get("bank_city")

        # Enrollment breakdown
        if parent.get("boys"):
            doc.male_enrollment = parent.get("boys")
        if parent.get("girls"):
            doc.female_enrollment = parent.get("girls")

        # Date
        submission_date = self._safe_date(parent.get("submission_date"))
        if submission_date:
            doc.sip_submission_date = submission_date

        # Requirements with deterministic zero-token math
        for req in data.get("requirements", []):
            unit_cost, total_cost = self._calculate_requirement_costs(req)
            doc.append(
                "requirements_table",
                {
                    "facility_name": req.get("facility", ""),
                    "current_status": req.get("status", ""),
                    "requirements": req.get("requirement", ""),
                    "unit_cost": unit_cost,
                    "total_cost": total_cost,
                    "priority": cint(req.get("priority")),
                    "responsibility_assigned": req.get("responsibility", ""),
                    "timeline": "2026-06-30",
                },
            )

        doc.insert(ignore_permissions=True)

        if cr_name:
            frappe.db.set_value(
                "Registration Table",
                cr_name,
                {"status": "Done", "master_form": doc.name},
            )

        return {"status": "success", "name": doc.name, "action": "created"}

        doc.insert(ignore_permissions=True)

        # Update registration table
        if cr_name:
            frappe.db.set_value(
                "Registration Table",
                cr_name,
                {"status": "Done", "master_form": doc.name},
            )

        frappe.db.commit()

        return {"status": "success", "name": doc.name, "action": "created"}

    @staticmethod
    def _safe_date(date_str) -> Optional[date]:
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
            return getdate(date_str)
        except Exception:
            return None


class DimensionValidator:
    """Deterministic & AI-assisted validation for Accounting and ERP Dimensions across Doctypes."""

    @staticmethod
    def validate_employee_project_contract(doc):
        """Legacy check: Ensure Employee contract & designation matches active Sales Orders."""
        if getattr(doc, "branch", None) == "SBC - Staff" or not getattr(
            doc, "project", None
        ):
            return

        emp_name = doc.name
        project = doc.project
        contract = frappe.db.get_value("Employee", emp_name, "contract")
        province = getattr(doc, "custom_provinces", None)
        designation = getattr(doc, "designation", None)

        sales_orders = frappe.db.sql(
            """
            SELECT so.name, so.project, so.contract, so.province, 
                   so.creation, so.transaction_date
            FROM `tabSales Order` so
            INNER JOIN (
                SELECT province, MAX(transaction_date) AS latest_date
                FROM `tabSales Order`
                WHERE project = %s AND docstatus = 1
                GROUP BY province
            ) latest ON so.province = latest.province 
                     AND so.transaction_date = latest.latest_date
            WHERE so.project = %s AND so.docstatus = 1
            ORDER BY so.transaction_date DESC, so.creation DESC
        """,
            (project, project),
            as_dict=True,
        )

        if not sales_orders:
            return

        filtered = [
            so for so in sales_orders if (so.get("contract") or "") == (contract or "")
        ]
        if not filtered:
            frappe.throw(
                f"No Sales Orders found for Employee: <b>{emp_name}</b> "
                f"with contract <b>{contract or '(blank)'}</b>."
            )

        filtered = [
            so for so in filtered if (so.get("province") or "") == (province or "")
        ]
        if not filtered:
            frappe.throw(
                f"No Sales Orders found for Employee: <b>{emp_name}</b> "
                f"with province <b>{province or '(blank)'}</b>."
            )

        matched = False
        for so in filtered:
            items = frappe.db.sql(
                """
                SELECT si.item_name, si.custom_designation
                FROM `tabSales Order Item` si
                WHERE si.parent = %s AND si.item_name = 'Position'
            """,
                (so.name,),
                as_dict=True,
            )

            for item in items:
                if item.get("custom_designation") == designation:
                    matched = True
                    break
            if matched:
                break

        if not matched:
            frappe.throw(
                f"No Sales Order items found with designation '<b>{designation}</b>' "
                f"matching Employee: <b>{emp_name}</b>."
            )

    @staticmethod
    def validate_sales_order_designations(doc):
        """Legacy check: Ensure Sales Order position items cover active project employees."""
        if not getattr(doc, "project", None):
            return

        so_project = doc.project
        so_contract = getattr(doc, "contract", None)
        so_province = getattr(doc, "province", None)

        so_designations = {
            item.custom_designation
            for item in doc.get("items", [])
            if getattr(item, "item_name", "") == "Position"
            and getattr(item, "custom_designation", None)
        }

        employees = frappe.get_all(
            "Employee",
            filters={
                "project": so_project,
                "status": "Active",
                "branch": ["!=", "SBC - Staff"],
            },
            fields=["name", "contract", "custom_provinces", "designation"],
        )

        missing = defaultdict(list)
        for emp in employees:
            if (
                emp.contract == so_contract
                and emp.custom_provinces == so_province
                and emp.designation
                and emp.designation not in so_designations
            ):
                missing[emp.designation].append(emp.name)

        if missing:
            error_msg = (
                f"Cannot save Sales Order. Existing employees for Project <b>{so_project}</b> "
                f"(Contract: {so_contract or 'Blank'}, Province: {so_province or 'Blank'}) "
                f"require the following designations:<br><br>"
            )
            for desig, emps in missing.items():
                error_msg += f"• <b>{desig}</b> (Required by: {', '.join(emps)})<br>"
            error_msg += "<br>Please add these positions before saving."
            frappe.throw(error_msg)


# ROUTER / ORCHESTRATOR
class AnalysisRouter:
    _analyzers = {
        AnalysisType.LEAVE_RISK: LeaveAnalyzer,
        AnalysisType.FINANCE_AUDIT: FinanceAnalyzer,
        AnalysisType.EXPENSE_VALIDATE: ExpenseAnalyzer,
        AnalysisType.SCHOOL_EXTRACT: SchoolDataExtractor,
    }

    @classmethod
    def route(cls, analysis_type: Union[str, AnalysisType], **kwargs) -> Dict:
        if isinstance(analysis_type, str):
            try:
                analysis_type = AnalysisType(analysis_type)
            except ValueError:
                return {
                    "status": "failed",
                    "error": f"Unknown analysis type: {analysis_type}",
                }

        analyzer_class = cls._analyzers.get(analysis_type)
        if not analyzer_class:
            return {
                "status": "failed",
                "error": f"No analyzer registered for {analysis_type.value}",
            }

        analyzer = analyzer_class()

        # Map analysis types to their analyze methods
        method_map = {
            AnalysisType.LEAVE_RISK: lambda a, k: a.analyze(**k),
            AnalysisType.FINANCE_AUDIT: lambda a, k: a.analyze(**k),
            AnalysisType.EXPENSE_VALIDATE: lambda a, k: a.analyze(**k),
            AnalysisType.SCHOOL_EXTRACT: lambda a, k: a.extract(**k),
        }

        return method_map[analysis_type](analyzer, kwargs)

    @classmethod
    def detect_type(cls, doc_type: str, context: Optional[Dict] = None) -> AnalysisType:
        mapping = {
            "Leave Application": AnalysisType.LEAVE_RISK,
            "Project Ledger Entry": AnalysisType.FINANCE_AUDIT,
            "Employee Expense Claim": AnalysisType.EXPENSE_VALIDATE,
            "Beneficiary Master": AnalysisType.SCHOOL_EXTRACT,
        }
        return mapping.get(doc_type, AnalysisType.LEAVE_RISK)


# FRAPPE WHITELISTED API ENDPOINTS

# Initialize analyzers
_leave_analyzer = LeaveAnalyzer()
_finance_analyzer = FinanceAnalyzer()
_expense_analyzer = ExpenseAnalyzer()
_school_extractor = SchoolDataExtractor()


@frappe.whitelist()
def analyze_leave(doc=None, employee=None, name=None, from_date=None):
    return _leave_analyzer.analyze(
        doc=doc, employee=employee, name=name, from_date=from_date
    )


@frappe.whitelist()
def leave_analysis(doc=None, employee=None, name=None, from_date=None):
    return analyze_leave(doc=doc, employee=employee, name=name, from_date=from_date)


@frappe.whitelist()
def analyze_finance(name=None):
    if not name:
        return {"status": "failed", "error": "Project Ledger Entry name is required"}

    return _finance_analyzer.analyze(name=name)


@frappe.whitelist()
def finance_analysis(name=None):
    return analyze_finance(name=name)


def on_expense_claim_workflow_trigger(doc, method=None):
    """
    Triggers AI Audit when Employee Expense Claim is submitted for approval (workflow_state == 'Sent For Approval').
    """
    workflow_state = getattr(doc, "workflow_state", None) or getattr(
        doc, "status", None
    )
    if workflow_state == "Sent For Approval":
        # Check if audit is already completed or in progress to prevent duplicate queueing
        audit_status = getattr(doc, "ai_audit_status", None)
        if audit_status not in ["In Progress", "Queued"]:
            print(
                f"[DEBUG EXPENSE HOOK] Expense Claim {doc.name} state changed to 'Sent For Approval'. Triggering AI Audit queue..."
            )
            trigger_expense_claim_analysis(doc=doc)


@frappe.whitelist()
def trigger_expense_claim_analysis(doc=None, name=None):
    docname = name or (
        doc.name
        if hasattr(doc, "name")
        else (doc.get("name") if isinstance(doc, dict) else doc)
    )
    if not docname:
        return {
            "status": "failed",
            "error": "Employee Expense Claim name or doc is required.",
        }

    # Set status to Queued
    frappe.db.set_value("Employee Expense Claim", docname, "ai_audit_status", "Queued")

    # Enqueue in background worker queue
    frappe.enqueue(
        "ai_analytics.new_api.process_expense_claim_async",
        queue="long",
        docname=docname,
        now=False,
    )

    return {
        "status": "queued",
        "docname": docname,
        "message": "AI Audit queued successfully. Processing in background.",
    }


def process_expense_claim_async(docname: str):
    try:
        res = _expense_analyzer.analyze(docname)
        # Notify user UI dynamically
        frappe.publish_realtime(
            event="expense_claim_ai_completed",
            message={
                "docname": docname,
                "status": res.get("compliance_rating", "Completed"),
                "risk_level": res.get("risk_level", "Low"),
            },
            after_commit=True,
        )
    except Exception as e:
        get_logger("process_expense_claim_async").error(
            f"Async worker failed for {docname}: {e}"
        )
        frappe.db.set_value(
            "Employee Expense Claim", docname, "ai_audit_status", "Failed"
        )


@frappe.whitelist()
def analyze_expense(name=None, sync=True):
    if not name:
        return {"status": "failed", "error": "Employee Expense Claim name is required"}

    if sync:
        return _expense_analyzer.analyze(name)
    return trigger_expense_claim_analysis(name=name)


@frappe.whitelist()
def expense_analysis(name=None, sync=True):
    return analyze_expense(name=name, sync=sync)


@frappe.whitelist()
def extract_school_data(name=None, file=None, cr_name=None, district=None):
    return _school_extractor.extract(
        name=name, file=file, cr_name=cr_name, district=district
    )


@frappe.whitelist()
def fetch_school_data(name=None, file=None, cr_name=None, district=None):
    return extract_school_data(name=name, file=file, cr_name=cr_name, district=district)


# DIMENSION VALIDATION ENGINE
class DimensionValidator:
    """Deterministic & AI-assisted validation for Accounting and ERP Dimensions across Doctypes."""

    @classmethod
    def validate_employee(cls, doc) -> Dict:
        issues = []
        warnings = []

        mandatory_fields = {
            "department": "Department",
            "company": "Company",
            "designation": "Designation",
        }

        for field, label in mandatory_fields.items():
            if not getattr(doc, field, None):
                issues.append(f"Missing mandatory dimension: {label}")

        # Check project or cost center setup
        if not getattr(doc, "payroll_cost_center", None) and not getattr(
            doc, "department", None
        ):
            warnings.append("Neither Payroll Cost Center nor Department is specified.")

        valid = len(issues) == 0
        return {
            "status": "success" if valid else "warning",
            "doctype": "Employee",
            "docname": getattr(doc, "name", None),
            "is_valid": valid,
            "issues": issues,
            "warnings": warnings,
        }

    @classmethod
    def validate_sales_order(cls, doc) -> Dict:
        issues = []
        warnings = []

        header_project = getattr(doc, "project", None)
        header_cost_center = getattr(doc, "cost_center", None)

        if not header_project:
            warnings.append("Header Project is not specified.")

        items = getattr(doc, "items", [])
        for idx, item in enumerate(items, 1):
            item_cc = getattr(item, "cost_center", None)
            item_proj = getattr(item, "project", None)

            if header_cost_center and item_cc and item_cc != header_cost_center:
                warnings.append(
                    f"Row #{idx}: Item Cost Center ({item_cc}) differs from Header ({header_cost_center})."
                )
            if header_project and item_proj and item_proj != header_project:
                warnings.append(
                    f"Row #{idx}: Item Project ({item_proj}) differs from Header ({header_project})."
                )

        valid = len(issues) == 0
        return {
            "status": "success" if valid else "warning",
            "doctype": "Sales Order",
            "docname": getattr(doc, "name", None),
            "is_valid": valid,
            "issues": issues,
            "warnings": warnings,
        }

    @classmethod
    def validate_expense_claim(cls, doc) -> Dict:
        issues = []
        warnings = []

        if not getattr(doc, "employee", None):
            issues.append("Expense Claim must be linked to an Employee.")

        expenses = getattr(doc, "expenses", [])
        for idx, item in enumerate(expenses, 1):
            if not getattr(item, "cost_center", None):
                warnings.append(
                    f"Row #{idx}: Cost Center is missing for expense claim detail."
                )

        valid = len(issues) == 0
        return {
            "status": "success" if valid else "warning",
            "doctype": "Expense Claim",
            "docname": getattr(doc, "name", None),
            "is_valid": valid,
            "issues": issues,
            "warnings": warnings,
        }

    @classmethod
    def validate_generic(cls, doc) -> Dict:
        doctype = getattr(doc, "doctype", "Unknown")
        issues = []
        warnings = []

        # Check standard ERPNext accounting dimension fields
        for dim in ["cost_center", "project", "department"]:
            if hasattr(doc, dim) and not getattr(doc, dim, None):
                warnings.append(
                    f"Dimension '{dim.replace('_', ' ').title()}' is empty."
                )

        return {
            "status": "success",
            "doctype": doctype,
            "docname": getattr(doc, "name", None),
            "is_valid": True,
            "issues": issues,
            "warnings": warnings,
        }


@frappe.whitelist()
def dimension_analysis(doc, method=None):
    print(
        f"\n=================== [AI ANALYTICS - DIMENSION ANALYSIS] ==================="
    )
    if isinstance(doc, str):
        try:
            parsed = json.loads(doc) if doc.startswith("{") else doc
            doc = (
                frappe.get_doc(parsed)
                if isinstance(parsed, dict)
                else (
                    frappe.get_doc("Employee", parsed)
                    if frappe.db.exists("Employee", parsed)
                    else (
                        frappe.get_doc("Sales Order", parsed)
                        if frappe.db.exists("Sales Order", parsed)
                        else frappe.get_doc("Expense Claim", parsed)
                    )
                )
            )
        except Exception as e:
            print(f"[DEBUG DIMENSION] Error resolving document input: {e}")
            return {
                "status": "failed",
                "error": f"Failed to resolve document: {str(e)}",
            }

    doctype = getattr(doc, "doctype", None)
    docname = getattr(doc, "name", "New")
    print(
        f"[DEBUG DIMENSION] Executing validation for DocType: '{doctype}' | DocName: '{docname}'"
    )

    if doctype == "Employee":
        res = DimensionValidator.validate_employee(doc)
    elif doctype == "Sales Order":
        res = DimensionValidator.validate_sales_order(doc)
    elif doctype == "Expense Claim" or doctype == "Employee Expense Claim":
        res = DimensionValidator.validate_expense_claim(doc)
    else:
        res = DimensionValidator.validate_generic(doc)

    print(
        f"[DEBUG DIMENSION] Result: Status={res.get('status')} | Valid={res.get('is_valid')} | Issues={len(res.get('issues', []))} | Warnings={len(res.get('warnings', []))}"
    )
    print(f"========================================================================\n")
    return res


@frappe.whitelist()
def batch_analyze_leaves(employee_leaves=None, from_date=None):
    if not employee_leaves:
        return {"status": "failed", "error": "employee_leaves is required"}

    import json as _json

    try:
        leaves_list = _json.loads(employee_leaves)
    except Exception:
        return {"status": "failed", "error": "Invalid employee_leaves JSON"}

    return _leave_analyzer.batch_analyze(
        employee_leaves=leaves_list, from_date=from_date, background=True
    )


@frappe.whitelist()
def route_analysis(analysis_type=None, **kwargs):
    if not analysis_type:
        return {"status": "failed", "error": "analysis_type is required"}

    return AnalysisRouter.route(analysis_type, **kwargs)


# FRAPPE DOCUMENT HOOKS
def on_employee_validate(doc, method):
    DimensionValidator.validate_employee(doc)


def on_sales_order_validate(doc, method):
    DimensionValidator.validate_sales_order(doc)


# BACKGROUND JOB WORKERS
def leave_batch_worker(employee_leaves, from_date=None):
    analyzer = LeaveAnalyzer()
    results = []

    for emp, name in employee_leaves:
        try:
            result = analyzer.analyze(employee=emp, name=name, from_date=from_date)
            results.append(result)
        except Exception as e:
            results.append(
                {"status": "failed", "employee": emp, "name": name, "error": str(e)}
            )

    return results


# UTILITY FUNCTIONS (Backward Compatible)
def get_model_response(prompt=None, type=None):
    if not prompt:
        return "Failed"

    response = ai_gateway.call(prompt=prompt)

    if not response.success:
        return "Failed"

    return {"content": json.dumps(response.content), "index": response.key_index}


def _extract_json(raw_content):
    return ai_gateway._parse_json_response(raw_content)


def _safe_date(date_str):
    return SchoolDataExtractor._safe_date(date_str)
