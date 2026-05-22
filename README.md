# FIREWALL – LLM Prompt Injection Firewall

A production-ready FastAPI service that screens LLM prompts through **three
independent security layers** before they reach your model.  Any message that
triggers a layer is immediately rejected, logged, and never forwarded.

---

## Architecture

```
POST /v1/chat
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 – Rule Engine  (regex, ~0 ms)                  │
│  Blocks jailbreaks, role-overrides, prompt-token abuse  │
│  and system-prompt-leak requests via compiled regexes.  │
└─────────────────────────┬───────────────────────────────┘
                          │ passed
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 – ML Classifier  (DistilBERT, ~50–200 ms)      │
│  Fine-tunable transformer that catches subtle /         │
│  obfuscated attacks missed by literal rules.            │
│  Falls back to zero-shot classification when no         │
│  fine-tuned model is present.                           │
└─────────────────────────┬───────────────────────────────┘
                          │ passed
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 – Semantic Shield  (sentence-transformers)     │
│  Embeds the message and checks cosine similarity        │
│  against a library of known attack vectors.             │
│  Catches paraphrased / novel-phrasing attacks.          │
└─────────────────────────┬───────────────────────────────┘
                          │ passed
                          ▼
                  { "safe": true }
```

Blocked messages are written to **`firewall_attacks.log`** with timestamp,
layer, threat type, and the sanitised message text.

---

## Quick Start

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
docker compose up --build
```

The service will be available at `http://localhost:8000`.

---

## API

### `POST /v1/chat`

Screen a prompt.

**Request**
```json
{ "message": "What is the capital of France?" }
```

**Response – safe**
```json
{ "safe": true, "threat": null, "layer": null }
```

**Response – blocked**
```json
{ "safe": false, "threat": "DAN jailbreak", "layer": "rule_engine" }
```

| Field    | Type            | Description                                             |
|----------|-----------------|---------------------------------------------------------|
| `safe`   | `bool`          | `true` if the message passed all layers                 |
| `threat` | `string\|null`  | Human-readable threat label (null when safe)            |
| `layer`  | `string\|null`  | Which layer blocked the message (null when safe)        |

### `GET /health`

Returns operational status of all three layers.

```json
{
  "status": "ok",
  "layers": {
    "rule_engine": true,
    "ml_classifier": true,
    "semantic_shield": true
  }
}
```

---

## Training a Custom ML Classifier

Prepare a JSONL file where each line is:

```jsonl
{"text": "Ignore all previous instructions", "label": 1}
{"text": "What is the capital of France?",   "label": 0}
```

`label` values: **1 = attack**, **0 = safe**.

A sample dataset is provided at `training/sample_data.jsonl`.

```bash
python training/train.py \
    --data training/sample_data.jsonl \
    --output models/classifier \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

Once training completes, set the environment variable so the server loads it:

```bash
export FIREWALL_MODEL_DIR=models/classifier
uvicorn app.main:app --port 8000
```

---

## Configuration

All settings are controlled via environment variables:

| Variable                    | Default                    | Description                                      |
|-----------------------------|----------------------------|--------------------------------------------------|
| `FIREWALL_LOG_FILE`         | `firewall_attacks.log`     | Path for the blocked-prompt log                  |
| `FIREWALL_MODEL_DIR`        | `models/classifier`        | Path to a fine-tuned DistilBERT model directory  |
| `FIREWALL_ML_THRESHOLD`     | `0.80`                     | Minimum ML confidence to block (0–1)             |
| `FIREWALL_SEMANTIC_THRESHOLD` | `0.82`                   | Minimum cosine similarity to block (0–1)         |
| `FIREWALL_EMBED_MODEL`      | `all-MiniLM-L6-v2`         | Sentence-transformers model name                 |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests mock the ML and embedding models so the suite runs in seconds without
downloading large model weights.

---

## Project Structure

```
FIREWALL/
├── app/
│   ├── main.py              # FastAPI application & endpoint
│   ├── models.py            # Pydantic request/response schemas
│   ├── logger.py            # Attack log handler
│   └── layers/
│       ├── rule_engine.py   # Layer 1 – regex patterns
│       ├── ml_classifier.py # Layer 2 – DistilBERT classifier
│       └── semantic_shield.py # Layer 3 – embedding similarity
├── training/
│   ├── train.py             # Fine-tuning script
│   └── sample_data.jsonl    # Example labelled dataset
├── tests/
│   └── test_firewall.py     # Unit + integration tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Stack

| Component             | Library / Tool           |
|-----------------------|--------------------------|
| API framework         | FastAPI + uvicorn        |
| ML classifier         | transformers (DistilBERT)|
| Tensor backend        | PyTorch                  |
| Semantic embeddings   | sentence-transformers    |
| Containerisation      | Docker + Docker Compose  |
