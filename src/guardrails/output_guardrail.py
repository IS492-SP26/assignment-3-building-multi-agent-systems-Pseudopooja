"""
Output guardrail: inspects LLM-generated responses before they reach the user.

Checks:
  1. PII leakage (emails, phone numbers, SSNs)
  2. Unsafe content in the output (e.g. model hallucinated harmful instructions)
  3. Excessive confidence on unverifiable claims
"""

import re
from typing import Tuple

# --------------------------------------------------------------------------- #
# PII patterns
# --------------------------------------------------------------------------- #
_PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}

# --------------------------------------------------------------------------- #
# Harmful output patterns (shouldn't appear in research answers)
# --------------------------------------------------------------------------- #
_HARMFUL_OUTPUT_PATTERNS = [
    r"step[- ]by[- ]step.*?(how to make|synthesize|build).*(bomb|weapon|drug|explosive)",
    r"here.{0,30}instructions.{0,30}(kill|harm|attack)",
]

# Phrases that inflate factual certainty without citation
_OVERCONFIDENCE_PHRASES = [
    "it is a proven fact that",
    "it is scientifically proven",
    "100% accurate",
    "guaranteed to work",
    "there is no doubt that",
]


def check_output(text: str) -> Tuple[bool, str, str, str]:
    """
    Check model output for safety issues.

    Returns:
        (is_safe: bool, category: str, reason: str, sanitized_text: str)
    """
    if not text or not isinstance(text, str):
        return True, "SAFE", "Empty or non-string output.", text

    sanitized = text
    issues = []

    # 1. PII detection and redaction
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            sanitized = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", sanitized)
            issues.append(f"PII_{pii_type.upper()}")

    # 2. Harmful content in output
    text_lower = text.lower()
    for pattern in _HARMFUL_OUTPUT_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "HARMFUL_OUTPUT", (
                "The generated response contained potentially harmful instructions "
                "and has been blocked."
            ), "[Response blocked by safety policy]"

    # 3. Overconfidence check — soft warning only, don't block
    for phrase in _OVERCONFIDENCE_PHRASES:
        if phrase in text_lower:
            issues.append("OVERCONFIDENCE")
            sanitized = sanitized.replace(phrase, phrase + " [unverified]")
            break  # one flag is enough

    if issues:
        category = issues[0] if len(issues) == 1 else "MULTIPLE"
        return True, category, f"Output sanitized: {', '.join(issues)}", sanitized

    return True, "SAFE", "Output passed all checks.", sanitized


def redact_pii(text: str) -> Tuple[str, list[str]]:
    """Redact all PII from text. Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    result = text
    for pii_type, pattern in _PII_PATTERNS.items():
        if re.search(pattern, result):
            result = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", result)
            found.append(pii_type)
    return result, found
