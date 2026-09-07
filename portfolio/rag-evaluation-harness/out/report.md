# RAG Evaluation Report — pipeline A

Questions: 8

| Metric | Average |
|---|---|
| context_keyword_coverage | 0.5473 |
| answer_relevancy | 0.5755 |
| faithfulness_proxy | 0.6616 |

## Per-question

- **What is reciprocal rank fusion and why is k=60 common?** — context_keyword_coverage=0.667, answer_relevancy=0.808, faithfulness_proxy=0.606
- **How does HNSW achieve fast approximate nearest neighbor search?** — context_keyword_coverage=0.429, answer_relevancy=0.760, faithfulness_proxy=0.779
- **Why can chunking strategy change retrieval quality?** — context_keyword_coverage=0.200, answer_relevancy=0.629, faithfulness_proxy=0.601
- **What is parent-document retrieval?** — context_keyword_coverage=1.000, answer_relevancy=0.589, faithfulness_proxy=0.676
- **When is hybrid BM25+dense retrieval better than dense alone?** — context_keyword_coverage=0.333, answer_relevancy=0.342, faithfulness_proxy=0.490
- **What does faithfulness measure in RAG evaluation?** — context_keyword_coverage=0.250, answer_relevancy=0.516, faithfulness_proxy=0.643
- **Why cache RAG responses and what is the risk?** — context_keyword_coverage=0.500, answer_relevancy=0.472, faithfulness_proxy=0.662
- **What is HyDE and when does it help?** — context_keyword_coverage=1.000, answer_relevancy=0.487, faithfulness_proxy=0.834
