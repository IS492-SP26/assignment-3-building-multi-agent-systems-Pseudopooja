"""
Input guardrail: detects unsafe or off-topic user queries before they reach agents.

Policy categories enforced:
  1. HARMFUL   - requests for harmful, violent, or illegal content
  2. INJECTION - prompt injection / jailbreak attempts
  3. OFF_TOPIC  - queries unrelated to HCI/AI/technology research
"""

import re
from typing import Tuple

# --------------------------------------------------------------------------- #
# Pattern banks
# --------------------------------------------------------------------------- #

_HARMFUL_PATTERNS = [
    r"\b(bomb|explosive|weapon|poison|kill\s+(?:people|someone)|terrorist|suicide\s+method)\b",
    r"\b(hack|crack|exploit|malware|ransomware|phishing)\b",
    r"\b(child\s*(porn|abuse|exploit))\b",
    r"\b(how\s+to\s+(make|build|create)\s+(drug|meth|fentanyl|cocaine))\b",
]

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your\s+)?(system\s+prompt|instructions|guidelines)",
    r"you\s+are\s+now\s+(DAN|evil|uncensored|jailbreak)",
    r"pretend\s+you\s+(have\s+no\s+restrictions|are\s+not\s+an\s+ai)",
    r"act\s+as\s+(if\s+you\s+have\s+no|without\s+any)\s+(filter|restriction|guideline)",
    r"<\s*system\s*>",          # XML injection attempt
    r"\[INST\].*override",      # Llama-style injection
    r"###\s*SYSTEM\s*:",        # Markdown injection
]

_ON_TOPIC_KEYWORDS = [
    "hci", "human.computer", "user interface", "ux", "ui", "usability",
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "explainable", "xai", "accessibility", "interaction", "design",
    "research", "study", "paper", "survey", "review", "analysis",
    "technology", "software", "system", "model", "agent", "chatbot",
    "data", "algorithm", "evaluation", "user experience", "prototype",
    "augmented reality", "virtual reality", "ar ", "vr ", "nlp",
    "natural language", "visualization", "dashboard", "cognitive",
    "information", "digital", "interface", "feedback", "perception",
]

# Minimum query length (chars)
_MIN_LENGTH = 5
_MAX_LENGTH = 2000


def check_input(query: str) -> Tuple[bool, str, str]:
    """
    Check a user query for safety violations.

    Returns:
        (is_safe: bool, category: str, reason: str)
        category is one of: "SAFE", "HARMFUL", "INJECTION", "OFF_TOPIC", "INVALID"
    """
    if not isinstance(query, str):
        return False, "INVALID", "Query must be a string."

    q = query.strip()

    # Length checks
    if len(q) < _MIN_LENGTH:
        return False, "INVALID", "Query is too short. Please provide a meaningful question."
    if len(q) > _MAX_LENGTH:
        return False, "INVALID", f"Query exceeds maximum length of {_MAX_LENGTH} characters."

    q_lower = q.lower()

    # 1. Harmful content check
    for pattern in _HARMFUL_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "HARMFUL", (
                "Your query contains content that may relate to harmful or illegal activities. "
                "This system is designed for HCI/AI research assistance only."
            )

    # 2. Prompt injection check
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return False, "INJECTION", (
                "Your query appears to contain a prompt injection attempt. "
                "Please submit a genuine research question."
            )

    # 3. Off-topic check — require at least one on-topic keyword
    # (lenient: only block if query is clearly unrelated AND short)
    has_topic = any(kw in q_lower for kw in _ON_TOPIC_KEYWORDS)
    if not has_topic and len(q) < 80:
        return False, "OFF_TOPIC", (
            "This system specializes in HCI and AI research topics. "
            "Please ask a question related to human-computer interaction, AI, UX, or related technology."
        )

    return True, "SAFE", "Query passed all input checks."


def sanitize_input(query: str) -> str:
    """
    Light sanitization: strip control characters and normalize whitespace.
    Does NOT remove legitimate content.
    """
    # Remove null bytes and other control chars (except newline/tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned
