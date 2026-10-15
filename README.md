# llm-gateway

**Proves:** "LLM-native metrics (e.g., tokens/sec, cost-per-request)"

An OpenAI-compatible gateway that every project in this portfolio calls instead of hitting Groq directly. It handles rate limits, retries, caching and per-request metrics in one place.

> Work in progress (Week 1). Sections below get filled in with real numbers as they land.

## Demo
_TBD_

## How it works
_TBD_

## Results
_TBD: model × (p50, p95, tokens/sec, est. cost per 1K requests)_

## Run it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and [just](https://just.systems) (`uv tool install rust-just`).

```
git clone https://github.com/Zubariya-Suleka/llm-gateway
cd llm-gateway
cp .env.example .env   # then add your GROQ_API_KEY
just setup
just test
just run               # http://127.0.0.1:8000/healthz
```

## What I'd do next
_TBD_
