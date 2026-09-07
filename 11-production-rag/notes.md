# ✍️ My Notes — Module 11: Production RAG

> Fill this in as you go. This is *your* learning artifact.

## Latency budget I measured / expect
| Stage | Typical cost | Notes |
|-------|-------------|-------|
| Embed query | ~50 ms |       |
| Retrieve (dense+BM25) | ~20 ms |       |
| Rerank (cross-encoder) | ~100 ms |       |
| LLM generate | 1–3 s | dominates! |

## Caching
- Exact-match cache: 
- Semantic cache threshold I used: 
- Observed hit rate: 
- Invalidation strategy: 

## Cost levers I'd apply
- 

## Monitoring metrics I'd track
- 

## Security controls checklist
- [ ] Prompt-injection filter
- [ ] PII redaction
- [ ] Metadata-based RBAC
- [ ] Audit logging
- [ ] GDPR (right to erasure) plan

## Pitfalls I've seen / would avoid
- 

## Questions for next time
- 
