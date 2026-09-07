# 🎯 Module 2 Plan — Document Processing & Chunking

## Learning Objectives
1. Explain what document loaders are and how they normalize files into `Document` objects.
2. Use the right loader for each format: Text, CSV, JSON, PDF, Markdown, Web, RecursiveWeb.
3. Compare text-splitting strategies: Character, RecursiveCharacter, Markdown-header, length-based, semantic.
4. Apply chunking best practices: size, overlap, semantic integrity, metadata strategy.
5. Attach and filter on metadata to power downstream retrieval filters.

## Prerequisites
- Module 1 completed (you know the ingestion side of the RAG data flow)
- `uv sync` done; sample data present in `data/samples/`

## Deliverables Checklist
- [ ] Read `02-learning.md`
- [ ] Run `code/main.py`: `uv run python 02-document-processing-chunking/code/main.py`
- [ ] Add your own document to `data/samples/` and watch it get ingested
- [ ] Compare the 4 splitter outputs for the same document — note which preserves structure best
- [ ] Write takeaways in `notes.md`

## Time Estimate
| Activity | Time |
|----------|------|
| Theory reading | 45 min |
| Running & comparing splitters | 60 min |
| Ingesting your own doc + notes | 30 min |
| **Total** | **~2.5 hours** |

## Success Criteria ("You're done when...")
- ✅ You can name a suitable loader for any file format you're likely to meet at work
- ✅ You can justify a chunk_size/overlap choice for a given document type
- ✅ You understand why "one splitter fits all" is wrong
- ✅ You can explain how metadata attached at ingestion time enables filtered retrieval later
- ✅ The comparison script runs and prints chunk stats for all 4 strategies
