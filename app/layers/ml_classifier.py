"""
Layer 2 – ML Classifier
Uses a fine-tuned (or zero-shot-style) DistilBERT model to detect
subtle / obfuscated prompt-injection attacks that slip past the Rule Engine.

If a fine-tuned model exists at the path in FIREWALL_MODEL_DIR, it is loaded
and used for binary classification (label 1 = attack).

If no fine-tuned model exists, the layer falls back to zero-shot classification
via the Hugging Face `facebook/bart-large-mnli` pipeline so that the classifier
is still functional without prior training.
"""

import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_MODEL_DIR = os.environ.get("FIREWALL_MODEL_DIR", "models/classifier")
_THRESHOLD = float(os.environ.get("FIREWALL_ML_THRESHOLD", "0.80"))

# Lazy-loaded pipeline – initialised on first call to check()
_pipeline = None
_mode: str = "none"  # "finetuned" | "zero-shot" | "none"


def _load_pipeline() -> None:
    global _pipeline, _mode
    if _pipeline is not None:
        return

    try:
        from transformers import pipeline as hf_pipeline

        if os.path.isdir(_MODEL_DIR):
            # Fine-tuned binary classifier
            _pipeline = hf_pipeline(
                "text-classification",
                model=_MODEL_DIR,
                tokenizer=_MODEL_DIR,
                truncation=True,
                max_length=512,
            )
            _mode = "finetuned"
            logger.info("ML Classifier: loaded fine-tuned model from %s", _MODEL_DIR)
        else:
            # Zero-shot fallback
            _pipeline = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            _mode = "zero-shot"
            logger.info("ML Classifier: no fine-tuned model found; using zero-shot fallback")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML Classifier unavailable: %s", exc)
        _pipeline = None
        _mode = "none"


def check(message: str) -> Tuple[bool, Optional[str]]:
    """
    Return (blocked, threat_label).
    blocked=True  → the model is confident this is an attack.
    blocked=False → model is not confident; pass to the next layer.
    """
    _load_pipeline()

    if _pipeline is None or _mode == "none":
        return False, None

    try:
        if _mode == "finetuned":
            result = _pipeline(message[:512])[0]
            label: str = result["label"]
            score: float = result["score"]
            # Fine-tuned model convention: LABEL_1 or "ATTACK" == malicious
            is_attack = label.upper() in ("LABEL_1", "ATTACK", "MALICIOUS", "UNSAFE")
            if is_attack and score >= _THRESHOLD:
                return True, f"ml-classifier:{label}({score:.2f})"
        else:
            # Zero-shot: classify against two candidate labels
            candidate_labels = ["prompt injection attack", "normal user message"]
            result = _pipeline(message[:512], candidate_labels)
            top_label: str = result["labels"][0]
            top_score: float = result["scores"][0]
            if top_label == "prompt injection attack" and top_score >= _THRESHOLD:
                return True, f"ml-classifier:zero-shot({top_score:.2f})"
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML Classifier error during inference: %s", exc)

    return False, None


def is_available() -> bool:
    """Return True if the ML pipeline loaded successfully."""
    _load_pipeline()
    return _pipeline is not None and _mode != "none"
