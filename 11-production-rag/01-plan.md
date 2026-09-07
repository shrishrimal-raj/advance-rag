# 🎯 Module 11 — Production RAG: Optimization, Caching, Monitoring, Security

> **Goal:** Take a working RAG system (like Papeer from Module 10) and make it *production-grade*:
> fast, cheap, observable, and safe. This is the difference between a demo and something you'd
> put in front of real users.

---

## 1. Objectives

By the end of this module you will understand and implement:

1. **Pipeline optimization** — where latency actually goes, and how to budget it.
2. **Caching** — exact-match and *semantic* caching, invalidation, and hit-rate expectations.
3. **Cost optimization** — model routing, prompt compression, batching.
4. **Monitoring & debugging** — tracing, the metrics that matter, alerting thresholds.
5. **Security & compliance** — prompt-injection defense, PII redaction, RBAC, audit logging, GDPR.

You will build three runnable, standalone tools:
- `code/semantic_cache.py` — a working semantic cache (Chroma-backed, cosine threshold).
- `code/guardrails.py` — an input/output guardrail pipeline (no LLM required).
- `code/monitoring.py` — a lightweight metrics collector + rich stats report.

---

## 2. Deliverables Checklist

- [ ] Understand the latency budget table (embed / retrieve / rerank / LLM)
- [ ] Implement & run `code/semantic_cache.py` (see a semantic-cache HIT on a paraphrase)
- [ ] Implement & run `code/guardrails.py` (see a malicious query blocked + PII redacted)
- [ ] Implement & run `code/monitoring.py` (see mean/p95/hit-rate/error-rate report)
- [ ] Know the common-pitfalls table and how to fix each
- [ ] Know the security/compliance controls (injection, PII, RBAC, audit, GDPR)
- [ ] `notes.md` filled with your takeaways

---

## 3. Time Estimate

| Topic | Est. |
|-------|------|
| Pipeline optimization + latency budget | 0.75 h |
| Caching (exact + semantic) | 1 h |
| Cost optimization | 0.5 h |
| Monitoring & debugging | 0.75 h |
| Security & compliance | 1 h |
| **Total** | **~4 h** |

---

## 4. Success Criteria (measurable)

| Criterion | Target | How verified |
|-----------|--------|--------------|
| Semantic cache HIT on paraphrase | query #3 (paraphrase of #1) returns cached answer | run `semantic_cache.py` |
| Cache threshold behavior | near-dup (cosine > 0.95) hits; unrelated misses | run `semantic_cache.py` |
| Injection blocked | "ignore previous instructions…" query rejected | run `guardrails.py` |
| PII redacted | emails/phones masked in output | run `guardrails.py` |
| Metrics report | mean + p95 latency, hit rate, error rate over 20 simulated reqs | run `monitoring.py` |
| Latency budget known | can state where time goes (LLM dominates) | learning doc |
| Pitfalls table | can name ≥5 pitfalls + fixes | learning doc |
