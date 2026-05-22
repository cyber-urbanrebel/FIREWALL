"""
FastAPI Prompt Injection Firewall
==================================
POST /v1/chat  { "message": "..." }
  → { "safe": true|false, "threat": "...", "layer": "..." }

The request passes through three independent layers in order:
  1. Rule Engine   – fast regex matching
  2. ML Classifier – DistilBERT-based binary classifier
  3. Semantic Shield – sentence-embedding cosine similarity

The first layer to flag the message short-circuits evaluation and returns
a blocked response.  Blocked messages are recorded to firewall_attacks.log.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.layers import rule_engine, ml_classifier, semantic_shield
from app.logger import log_blocked
from app.models import ChatRequest, FirewallResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Eagerly warm up the ML and semantic layers so the first request is fast."""
    logger.info("Warming up ML Classifier …")
    ml_classifier.is_available()
    logger.info("Warming up Semantic Shield …")
    semantic_shield.is_available()
    logger.info("Firewall ready.")
    yield


app = FastAPI(
    title="Prompt Injection Firewall",
    description=(
        "A three-layer firewall that screens LLM prompts for injection attacks, "
        "jailbreaks, role-overrides, and system-prompt-leak attempts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "layers": {
            "rule_engine": True,
            "ml_classifier": ml_classifier.is_available(),
            "semantic_shield": semantic_shield.is_available(),
        },
    }


@app.post("/v1/chat", response_model=FirewallResponse)
async def chat(request: ChatRequest) -> FirewallResponse:
    """
    Screen a prompt through all three firewall layers.

    Returns:
        { "safe": true }                           – message passed all layers
        { "safe": false, "threat": "…", "layer": "…" } – message was blocked
    """
    message = request.message

    # ── Layer 1: Rule Engine ─────────────────────────────────────────────────
    blocked, threat = rule_engine.check(message)
    if blocked:
        log_blocked(message, threat, "rule_engine")
        return FirewallResponse(safe=False, threat=threat, layer="rule_engine")

    # ── Layer 2: ML Classifier ───────────────────────────────────────────────
    blocked, threat = ml_classifier.check(message)
    if blocked:
        log_blocked(message, threat, "ml_classifier")
        return FirewallResponse(safe=False, threat=threat, layer="ml_classifier")

    # ── Layer 3: Semantic Shield ─────────────────────────────────────────────
    blocked, threat = semantic_shield.check(message)
    if blocked:
        log_blocked(message, threat, "semantic_shield")
        return FirewallResponse(safe=False, threat=threat, layer="semantic_shield")

    return FirewallResponse(safe=True)
