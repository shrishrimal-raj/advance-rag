export const meta = {
  name: 'advance_rag_batch_b',
  description: 'Modules 07-11 v2 expansion: multi-approach code + Noob-to-Expert diagrams',
  phases: [{ title: 'wave1' }, { title: 'wave2' }]
};

const CONV = args.conventions;

const SCHEMA = {
  type: 'object',
  properties: {
    files_created: { type: 'array', items: { type: 'string' } },
    all_compiled: { type: 'boolean' },
    fully_ran: { type: 'array', items: { type: 'string' } },
    diagrams_added: { type: 'boolean' },
    issues: { type: 'string' }
  },
  required: ['files_created', 'all_compiled', 'fully_ran', 'diagrams_added', 'issues']
};

function brief(mod, folder, filesSpec, extra) {
  return CONV + '\n\nYOUR MODULE: ' + mod + ' (folder: ' + folder + '/)\n\nCREATE THESE FILES (exact paths):\n' + filesSpec + '\n\n' + (extra || '') + '\nIf any target file already exists, read it first and complete/improve it instead of blind overwrite.';
}

phase('wave1');
log('wave1: modules 07-09');
const w1 = await parallel([
  () => agent(brief('07', '07-advanced-rag-patterns', [
    '- code/rag_fusion/approach_2_from_scratch.py — Framework-free RAG Fusion: LLM expands the user query into 3 variants, each searched in raw chromadb (dense), rank lists fused with numpy RRF (k=60), top-k sent to LLM for the answer. NOTE: script lives in code/<subdir>/ so use parents[2] for the shared import. Graceful degradation.',
    '- code/hyde/approach_2_from_scratch.py — Framework-free HyDE: LLM writes a hypothetical answer document, embed it, retrieve from raw chromadb, then answer the real question using retrieved context. Show hypothetical doc + retrieved docs + final answer. Use parents[2]. Graceful degradation.',
    '- code/pattern_comparison.py — Side-by-side pattern shootout at code/ level (parents[1]): naive RAG vs hybrid (BM25+dense RRF) vs RAG-Fusion vs HyDE on 3 fixed questions over a small indexed corpus. Table: latency ms, context keyword-coverage proxy, answer length. Rich output + verdict on when each pattern wins. LLM-dependent: graceful degradation.'
  ].join('\n')), { label: 'mod07', schema: SCHEMA }),
  () => agent(brief('08', '08-agentic-rag-langgraph', [
    '- code/plan_and_execute_rag.py — Plan-and-Execute with LangGraph: planner node decomposes the question into steps, executor node runs each step (RAG search tool or direct reasoning), synthesizer merges step results into the final answer. Explicit StateGraph with typed state, print the plan + per-step trace. Graceful degradation.',
    '- code/multi_tool_agent.py — Multi-tool ReAct agent: bind 3 tools (vector_search, metadata_filter_search, calculator) to the LLM, let it choose per step; print full tool-call trace showing different tools selected for different sub-questions. Graceful degradation.',
    '- code/streaming_agent.py — Token streaming from an agentic RAG: use astream_events (or astream) on a small LangGraph RAG agent, print tokens as they arrive with a stage banner (retrieval -> generation). If the provider does not support streaming, detect and fall back to non-streamed answer with a note. Graceful degradation.'
  ].join('\n')), { label: 'mod08', schema: SCHEMA }),
  () => agent(brief('09', '09-rag-evaluation-ragas', [
    '- code/approach_2_custom_metrics.py — Evaluation WITHOUT RAGAS: implement (a) faithfulness proxy = claim-level support via embedding similarity between generated claims and context, (b) context precision/recall heuristics via semantic overlap between retrieved context and reference answer, (c) answer relevancy = embedding similarity(question, answer). Build a small hand-crafted eval set (5-10 Q/reference-answer/context triples derived from data/samples docs) and print a metrics table. No LLM needed if contexts are precomputed — keep it runnable offline.',
    '- code/approach_3_llm_judge_from_scratch.py — DIY LLM-as-judge: structured rubric prompt scoring 1-5 on faithfulness, relevance, completeness; strict JSON output parsing with retry-on-malformed; aggregate mean scores per criterion. Evaluate 3 sample Q/A/context triples. Graceful degradation.',
    '- code/ab_pipeline_comparison.py — A/B pipeline comparison: pipeline A = naive single dense retriever; pipeline B = hybrid BM25+dense RRF + rerank. Same 5 questions, same generator. Score both with the custom metrics from approach_2_custom_metrics (import its functions). Table A vs B per metric + verdict. Graceful degradation.'
  ].join('\n')), { label: 'mod09', schema: SCHEMA })
]);

phase('wave2');
log('wave2: modules 10-11');
const w2 = await parallel([
  () => agent(brief('10', '10-capstone-project', [
    '- code/tests/test_ingestion.py — pytest suite for the existing papeer package (code/papeer/ingestion.py): ingesting data/samples docs yields non-empty chunks with metadata; re-ingestion is idempotent (no duplicate ids). Keep tests fast (<60s), no LLM calls.',
    '- code/tests/test_retrieval.py — pytest suite for code/papeer/retrieval.py: known questions return relevant top-k (assert expected keywords present in concatenated top-3), metadata filtering narrows results correctly. No LLM calls.',
    '- code/deploy/Dockerfile — python:3.12-slim + uv, copy project, uv sync --frozen --no-dev, expose 8501, CMD streamlit run code/ui/streamlit_app.py. Non-root user, healthcheck.',
    '- code/deploy/docker-compose.yml — app service (build context repo root), chroma persist dir as named volume, env_file .env, restart unless-stopped, healthcheck hitting streamlit /_stcore/health.',
    '- code/deploy/DEPLOYMENT.md — Production runbook: prerequisites, build/run commands, env var table, scaling notes, backup/restore of the vector store, monitoring hooks (LangSmith), rollback procedure.',
    '- code/ui/streamlit_app.py — Live UI over the papeer package: query box, top-k results with scores + source metadata, generated answer panel, sidebar controls (k, score threshold, provider display). Must import cleanly and show a friendly hint page when no LLM provider is reachable (never crash on startup).'
  ].join('\n'), 'EXTRA: run uv sync ONCE first (pytest + streamlit were just added to pyproject.toml). Then verify tests actually pass: uv run pytest code/tests -v — all green required. The streamlit app must at least pass py_compile plus an import check via uv run python.'), { label: 'mod10', schema: SCHEMA }),
  () => agent(brief('11', '11-production-rag', [
    '- code/cost_optimization.py — Token/cost accounting: for 3 sample queries measure tokens per stage (retrieved context chars->tokens estimate, prompt, completion) using response usage when available else char/4 heuristic; cost table for OpenAI gpt-4o-mini vs Yolo-Auto qwen3.8-27b pricing placeholders; quantify savings levers (smaller k, smaller chunks, caching) as % estimates. Rich tables. Graceful degradation (heuristic mode works offline).',
    '- code/caching_strategies.py — Exact cache (SHA-256 of normalized query) vs semantic cache (embedding cosine > threshold): replay a 10-query workload with repeats + paraphrases, report hit rate, avg latency saved, and false-positive risk demo (near-miss query that should NOT hit). Local embeddings only — runnable offline.',
    '- code/debugging_playbook.py — Failure triage demo: scripted simulation of 5 production failures (empty retrieval, low-confidence scores, hallucinated answer, LLM timeout, embedding dim mismatch) — for each: symptoms, diagnostic steps, root cause, fix. Interactive menu (rich Prompt) with a --demo flag that runs all five non-interactively.',
    '- code/security_compliance.py — PII guardrails: regex detectors for email/phone/SSN/credit-card, redaction function applied pre-indexing, access-control demo via metadata ACL filtering (public/internal/restricted), compliance checklist output (GDPR/HIPAA-relevant notes). Runnable offline.'
  ].join('\n')), { label: 'mod11', schema: SCHEMA })
]);

return { wave1: w1, wave2: w2 };