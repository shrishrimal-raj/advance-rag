# 📝 Module 09 — My Notes: RAG Evaluation with RAGAS

> Fill this in as you work through the module. It's your personal record of what you learned, what you measured, and what you'd do next.

## 1. The big idea (in my own words)
- Why evaluation separates toy RAG from production RAG:
  - 
- Retrieval vs generation — why I evaluate them separately:
  - 

## 2. The four core metrics — my one-line definitions
| Metric | Stage | What a LOW score means | My fix to try |
|--------|-------|------------------------|---------------|
| Faithfulness | Generation | | |
| Answer Relevancy | Generation | | |
| Context Precision | Retrieval | | |
| Context Recall | Retrieval | |

## 3. Golden dataset
- How many examples did I build? 
- Which source files are covered? 
- One "hard" example I added and why it's hard:
  - 

## 4. Baseline scorecard (fill after running `evaluate_pipeline.py`)
| Metric | Mean score | Floor I'd set | Regression margin |
|--------|-----------|---------------|-------------------|
| Faithfulness | | | |
| Answer Relevancy | | | |
| Context Precision | | | |
| Context Recall | | | |

- Overall bottleneck stage (retrieval or generation): 
- The single change I'd make first, and which metric I expect it to move:

## 5. "Break it on purpose" experiment
- What I changed: 
- Which metric dropped (as expected?): 
- Did any *other* metric move unexpectedly? What did that teach me?

## 6. LLM-as-judge caveats I'll respect
- Bias I'm most worried about for my setup: 
- Mitigation I'll apply (e.g., different judge model than generator): 
- How I'll spot-audit the judge's verdicts: 

## 7. CI plan
- Fast smoke subset size (every commit): 
- Full eval cadence (nightly / on model+prompt changes): 
- Pass/fail rule I'll encode: 

## 8. Questions / confusions to revisit
- 
- 

## 9. Next steps
- [ ] Expand golden set with real user questions
- [ ] Add a reranker and re-measure Context Precision
- [ ] Persist baseline + plot trend over time
- [ ] Wire eval into CI with a regression gate
