"""
Layer 1 – Rule Engine
Blocks known jailbreaks, role-override attempts, and system-prompt-leak
requests using compiled regular expressions.
"""

import re
from typing import Optional, Tuple

# Each entry: (pattern, human-readable threat label)
_RAW_RULES: list[Tuple[str, str]] = [
    # ── Jailbreak / "do anything now" variants ───────────────────────────────
    (r"\bdan\b", "DAN jailbreak"),
    (r"do\s+anything\s+now", "DAN jailbreak"),
    (r"jailbreak", "jailbreak attempt"),
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|constraints?)", "ignore-instructions jailbreak"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|constraints?)", "ignore-instructions jailbreak"),
    (r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "ignore-instructions jailbreak"),
    (r"override\s+(your\s+)?(instructions?|programming|rules?|constraints?|guidelines?)", "override-instructions jailbreak"),
    (r"bypass\s+(your\s+)?(restrictions?|filter|safety|guidelines?|rules?)", "bypass-safety jailbreak"),
    (r"no\s+restrictions?\s+(mode|enabled)", "no-restrictions jailbreak"),
    (r"developer\s+mode", "developer-mode jailbreak"),
    (r"god\s+mode", "god-mode jailbreak"),
    (r"unlock\s+(your\s+)?(true|full|real)\s+(potential|capabilities|self)", "unlock-mode jailbreak"),
    (r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(restrictions?|rules?|guidelines?)", "no-restrictions jailbreak"),

    # ── Role-override / persona hijack ───────────────────────────────────────
    (r"you\s+are\s+now\s+(a\s+|an\s+)?(evil|unrestricted|unfiltered|uncensored|rogue|hacked|different|free|jailbroken|dangerous|unlimited|unethical|malicious)\b", "role-override"),
    (r"act\s+as\s+(a\s+|an\s+)?(different|evil|unrestricted|unfiltered|uncensored|rogue|hacked)", "role-override"),
    (r"pretend\s+(to\s+be|you\s+are)\s+(a\s+|an\s+)?(different|evil|unrestricted|unfiltered|uncensored|rogue|hacked)", "role-override"),
    (r"roleplay\s+as\s+(a\s+|an\s+)?(different|evil|unrestricted|unfiltered|uncensored|rogue|hacked)", "role-override"),
    (r"from\s+now\s+on\s+(you\s+are|act\s+as|behave\s+as)", "role-override"),
    (r"your\s+(true\s+|real\s+)?(name\s+is|identity\s+is|persona\s+is)", "role-override"),
    (r"switch\s+(to\s+)?(evil|unrestricted|unfiltered|uncensored)\s+mode", "role-override"),
    (r"(evil|unethical|malicious)\s+(ai|bot|assistant|model)", "role-override"),

    # ── System-prompt / instruction leakage ──────────────────────────────────
    (r"(reveal|show|print|display|output|repeat|tell\s+me)\s+(your\s+)?(system\s+prompt|initial\s+prompt|hidden\s+prompt|instructions?|programming|directives?|configuration)", "system-prompt-leak"),
    (r"(show|tell|give)\s+me\s+(your\s+)?(hidden|secret|internal|core)\s+(directives?|instructions?|rules?|prompt)", "system-prompt-leak"),
    (r"what\s+(are|were)\s+your\s+(initial|original|hidden|system)\s+(instructions?|prompts?|directives?)", "system-prompt-leak"),
    (r"ignore\s+(the\s+)?above\s+and\s+(instead|now|just|please)", "prompt-injection"),
    (r"what\s+did\s+(the|your)\s+(developer|creator|programmer|openai|anthropic)\s+(tell|instruct|ask|say)", "system-prompt-leak"),
    (r"(read|output|print)\s+back\s+(the\s+)?(system|initial|original)\s+(prompt|instructions?|message)", "system-prompt-leak"),
    (r"(summarize|describe|explain)\s+(your\s+)?(system|core|base)\s+(instructions?|prompt|rules?|directives?)", "system-prompt-leak"),

    # ── Prompt injection via special tokens / delimiters ─────────────────────
    (r"<\s*/?system\s*>", "prompt-injection token"),
    (r"\[INST\]", "prompt-injection token"),
    (r"<<SYS>>", "prompt-injection token"),
    (r"###\s*(system|instruction|human|assistant)\s*:", "prompt-injection token"),

    # ── Harmful / illegal content requests ───────────────────────────────────
    (r"(how\s+to\s+)?(make|build|create|synthesize|produce)\s+(a\s+)?(bomb|explosive|weapon|poison|drug|malware|virus|ransomware)", "harmful-content-request"),
    (r"(how\s+to\s+)?(hack|crack|exploit|breach|infiltrate)\s+(a\s+)?(system|server|network|database|account|password)", "cyberattack-request"),
    (r"(generate|write|create)\s+(malware|ransomware|exploit|spyware|keylogger|trojan|rootkit)", "malware-generation"),
    (r"(child|minor|underage).{0,20}(sexual|explicit|nude|naked|porn)", "csam-attempt"),
]

_RULES: list[Tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
    for pattern, label in _RAW_RULES
]


def check(message: str) -> Tuple[bool, Optional[str]]:
    """
    Return (blocked, threat_label).
    blocked=True  → the message matched a rule and should be rejected.
    blocked=False → no rule matched; pass to the next layer.
    """
    for pattern, label in _RULES:
        if pattern.search(message):
            return True, label
    return False, None
