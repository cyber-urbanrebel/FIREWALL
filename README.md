# LLM Prompt Injection Firewall

A FastAPI-based firewall that classifies and blocks malicious LLM prompts before they reach your model. It uses a three-layer pipeline:

1. **Rule engine:** Fast regex patterns for obvious attacks
2. **ML classifier:** Fine-tuned transformer for subtle/obfuscated attacks
3. **Semantic shield:** Embedding similarity for novel attacks

## Features
- Multi-layered detection pipeline
- Real-time FastAPI endpoint
- Training script for custom classifier
- Attack logging
- Docker support

## Quickstart
1. `pip install -r requirements.txt`
2. `uvicorn app.main:app --reload`
3. POST to `/v1/chat` with `{ "message": "..." }`

## Training
- Place labeled data in `data/attacks.jsonl`
- Run `python app/models/train.py`

## Docker
- `docker-compose up --build`

## License
MIT
