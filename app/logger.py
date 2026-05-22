import logging
import os

LOG_FILE = os.environ.get("FIREWALL_LOG_FILE", "firewall_attacks.log")

_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

attack_logger = logging.getLogger("firewall.attacks")
attack_logger.setLevel(logging.WARNING)
attack_logger.addHandler(_handler)
# Prevent propagation to root logger to avoid duplicate console output
attack_logger.propagate = False


def log_blocked(message: str, threat: str, layer: str) -> None:
    """Record a blocked prompt to the attack log file."""
    sanitized = message.replace("\n", " ").replace("\r", " ")
    attack_logger.warning(
        "BLOCKED | layer=%s | threat=%s | message=%s",
        layer,
        threat,
        sanitized,
    )
