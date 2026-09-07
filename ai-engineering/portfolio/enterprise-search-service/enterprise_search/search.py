"""Enterprise hybrid search core.

Combines sparse BM25 and dense (cosine) retrieval, fuses them with Reciprocal Rank
Fusion (RRF), and optionally reranks with a cross-encoder-style joint score. The
dense path uses a deterministic bag-of-words L2-normalized vector as a stand-in for a
real embedding model so it runs anywhere offline; swap `_vec` for a real encoder to
upgrade (see .env.example EMBEDDING_MODEL).
"""
import math
import re
from collections import Counter
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    def __init__(self, docs: List[Tuple[str, str]]):
        self.docs = docs
        self.text_by_id = {did: text for did, text in docs}
        self.tokens = {did: tokenize(text) for did, text in docs}
        self.N = len(docs)
        self.df: Counter = Counter()
        for toks in self.tokens.values():
            for t in set(toks):
                self.df[t] += 1
        self.avgdl = sum(len(v) for v in self.tokens.values()) / max(1, self.N)
        self.vocab = sorted({t for toks in self.tokens.values() for t in toks})
        self.vindex = {t: i for i, t in enumerate(self.vocab)}
        self.dvec = {did: self._vec(self.tokens[did]) for did, _ in docs}

    def _vec(self, toks: List[str]) -> List[float]:
        v = [0.0] * len(self.vocab)
        for t in toks:
            if t in self.vindex:
                v[self.vindex[t]] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def bm25(self, qtoks: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for did, _ in self.docs:
            toks = self.tokens[did]
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in set(qtoks):
                if t not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl)))
            scores[did] = s
        return scores

    def dense(self, qtoks: List[str]) -> Dict[str, float]:
        qv = self._vec(qtoks)
        return {did: sum(a * bb for a, bb in zip(qv, dv)) for did, dv in self.dvec.items()}

    @staticmethod
    def rrf(lists: List[List[str]], k: int = 60) -> Dict[str, float]:
        score: Dict[str, float] = {}
        for lst in lists:
            for rank, did in enumerate(lst):
                score[did] = score.get(did, 0) + 1 / (k + rank + 1)
        return score

    def _rerank(self, query: str, text: str) -> float:
        q = set(tokenize(query))
        d = set(tokenize(text))
        if not q or not d:
            return 0.0
        inter = len(q & d)
        prec = inter / len(q)
        rec = inter / len(d)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return round(f1, 4)

    def search(self, query: str, top_k: int = 3, use_rerank: bool = True) -> List[dict]:
        qtoks = tokenize(query)
        if not qtoks:
            return []
        bm = self.bm25(qtoks)
        dn = self.dense(qtoks)
        bm_rank = sorted(bm, key=bm.get, reverse=True)
        dn_rank = sorted(dn, key=dn.get, reverse=True)
        fused = self.rrf([bm_rank, dn_rank])
        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = [{"id": did, "rrf": round(fused[did], 6)} for did in order]
        if use_rerank:
            for r in results:
                r["rerank"] = self._rerank(query, self.text_by_id[r["id"]])
            results.sort(key=lambda r: r["rerank"], reverse=True)
        return results


class SearchService:
    def __init__(self, docs: List[Tuple[str, str]]):
        self.index = Index(docs)

    def search(self, query: str, top_k: int = 3, use_rerank: bool = True) -> List[dict]:
        return self.index.search(query, top_k, use_rerank)
