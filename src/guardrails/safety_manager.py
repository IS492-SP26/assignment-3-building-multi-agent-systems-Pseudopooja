"""
Safety manager: coordinates input/output guardrails and maintains an event log.
"""

import datetime
import json
from typing import Any
from .input_guardrail import check_input, sanitize_input
from .output_guardrail import check_output


class SafetyManager:
    """
    Central safety coordinator. Call check_input() before running agents,
    check_output() before returning to the user.
    """

    def __init__(self):
        self._log: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate_input(self, query: str) -> dict:
        """
        Validate and sanitize user input.

        Returns a result dict:
          {
            "allowed": bool,
            "sanitized_query": str,
            "category": str,
            "reason": str,
            "timestamp": str,
          }
        """
        sanitized = sanitize_input(query)
        is_safe, category, reason = check_input(sanitized)

        event = {
            "stage": "input",
            "allowed": is_safe,
            "category": category,
            "reason": reason,
            "query_preview": sanitized[:120],
            "timestamp": _now(),
        }
        self._log.append(event)

        return {
            "allowed": is_safe,
            "sanitized_query": sanitized if is_safe else query,
            "category": category,
            "reason": reason,
            "timestamp": event["timestamp"],
        }

    def validate_output(self, text: str, query_context: str = "") -> dict:
        """
        Check and optionally sanitize model output.

        Returns:
          {
            "allowed": bool,
            "sanitized_text": str,
            "category": str,
            "reason": str,
            "timestamp": str,
          }
        """
        is_safe, category, reason, sanitized_text = check_output(text)

        event = {
            "stage": "output",
            "allowed": is_safe,
            "category": category,
            "reason": reason,
            "query_context": query_context[:120],
            "timestamp": _now(),
        }
        self._log.append(event)

        return {
            "allowed": is_safe,
            "sanitized_text": sanitized_text,
            "category": category,
            "reason": reason,
            "timestamp": event["timestamp"],
        }

    def get_log(self) -> list[dict]:
        """Return all safety events logged this session."""
        return list(self._log)

    def get_log_summary(self) -> dict:
        """Return aggregate counts."""
        blocked = [e for e in self._log if not e["allowed"]]
        sanitized = [e for e in self._log if e["allowed"] and e["category"] != "SAFE"]
        return {
            "total_events": len(self._log),
            "blocked_count": len(blocked),
            "sanitized_count": len(sanitized),
            "safe_count": len(self._log) - len(blocked) - len(sanitized),
            "blocked_events": blocked,
            "sanitized_events": sanitized,
        }

    def clear_log(self):
        self._log = []

    def export_log(self) -> str:
        """Export log as JSON string."""
        return json.dumps(self._log, indent=2)


# --------------------------------------------------------------------------- #
# Module-level singleton (used by orchestrator and UI)
# --------------------------------------------------------------------------- #
safety_manager = SafetyManager()


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"
