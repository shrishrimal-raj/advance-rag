# 🎯 Module 1 Plan — RAG Fundamentals & Architecture

## Learning Objectives
By the end of this module you will be able to:
1. Explain **why** RAG exists: hallucinations, knowledge cutoffs, private/proprietary data.
2. Describe the core RAG data flow: **Knowledge Base → Retriever → Generator**.
3. Distinguish **RAG vs Fine-Tuning vs Prompt Engineering** and know when to use each.
4. Build a working **naive RAG pipeline** from scratch with LangChain + ChromaDB.
5. Identify the gap between naive RAG and production RAG (the rest of this course).

## Prerequisites
- Basic Python (functions, imports, virtual environments)
- `uv` installed and `uv sync` run at project root
- Optional but recommended: [Ollama](https://ollama.com) running locally (`ollama serve` + `ollama pull llama3.1`) — needed for the LLM generation step. Without it, the retrieval part still runs and the script exits gracefully.

## Deliverables Checklist
- [ ] Read `02-learning.md` (theory + architecture diagrams)
- [ ] Run `code/main.py` end-to-end: `uv run python 01-rag-fundamentals/code/main.py`
- [ ] Change the example questions and observe how retrieved chunks change
- [ ] Break something on purpose (e.g., set chunk_size=30) and observe the effect
- [ ] Write your takeaways in `notes.md`

## Time Estimate
| Activity | Time |
|----------|------|
| Theory reading | 45 min |
| Running & experimenting with code | 60 min |
| Notes + reflection | 15 min |
| **Total** | **~2 hours** |

## Success Criteria ("You're done when...")
- ✅ You can draw the RAG data flow from memory (KB → Retriever → Generator)
- ✅ You can explain in one sentence why RAG reduces hallucinations
- ✅ You can state 3 situations where fine-tuning beats RAG and 3 where RAG beats fine-tuning
- ✅ `main.py` runs and prints: retrieved chunks → generated answer with citations
- ✅ You can list at least 4 ways this naive pipeline falls short of production grade
