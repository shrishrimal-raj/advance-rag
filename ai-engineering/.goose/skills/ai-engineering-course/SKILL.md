---
name: ai-engineering-course
description: Use when working in the ai-engineering/ SDE->AI Engineer course - adding modules, writing multi-approach code, portfolio projects, or verifying runs. Encodes the course conventions, hard hardware constraints, and known API pitfalls.
---

# AI Engineering Course Skill

Guidance for building and verifying code in the `ai-engineering/` 10-week course.

## When to use
- Adding or editing a weekly module (`NN-slug/`).
- Writing the four code approaches (main / approach_2 / approach_3 / benchmark).
- Building or testing a portfolio project under `portfolio/`.
- Verifying that scripts compile and run.

## Hard constraints
- NEVER run/pull Ollama locally (8 GB laptop). LLM = Yolo-Auto cloud via root `.env`. Local MiniLM OK.
- Minimize full-pipeline LLM runs (1-2/module). Heavy workloads = code+docs only.
- No model downloads >100 MB.

## Code conventions
1. UTF-8 guard at top of every runnable script (win32 reconfigure stdout/stderr).
2. `sys.path.append(...parents[1])` then `from shared.config import get_llm, get_embeddings` (use `parents[2]` from `code/<subdir>/`).
3. Graceful degradation: no LLM/key -> clear hint + `sys.exit(0)`, never crash.

## API pitfalls
- `rank_bm25.BM25Okapi`; `langchain_classic` fallback; `LLMChainExtractor.from_llm(llm)`; ParentDocument byte_store check; SelfQuery neutralize `{}`; `InMemoryStore` from `langchain_core.stores`; `JSONLoader(file_path=..., jq_schema='.', text_content=False)`.

## Verification checklist
- [ ] `python -m py_compile` all new files
- [ ] Run all offline (no-LLM) scripts end-to-end
- [ ] <=1-2 LLM smoke tests
- [ ] `02-learning.md` has >=5 mermaid diagrams

## Layout
- Week: `NN-slug/{01-plan.md,02-learning.md,03-implementation.md,code/...}`
- Portfolio: `portfolio/<project>/{README.md,docs/,tests/,Dockerfile,docker-compose.yml,.github/workflows/ci.yml}`
