"""The Multimodal RAG Engine core.

Routes **text / image / audio** inputs through pluggable modality processors, fuses their
evidence into a single retrieval context, and tracks **live dashboard metrics**. Offline
and testable via injected processors; swap in real CLIP / Whisper / vision clients in
production without touching the routing or fusion logic.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

MODALITIES = ("text", "image", "audio")


@dataclass
class Evidence:
    modality: str
    content: str
    score: float = 1.0


@dataclass
class DashboardMetrics:
    processed: int = 0
    by_modality: Dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def record(self, modality: str, ok: bool):
        self.processed += 1
        self.by_modality[modality] = self.by_modality.get(modality, 0) + 1
        if not ok:
            self.errors += 1

    def snapshot(self) -> dict:
        return {"processed": self.processed, "by_modality": dict(self.by_modality),
                "errors": self.errors,
                "success_rate": round(1 - self.errors / self.processed, 4) if self.processed else 1.0}


def detect_modality(item: dict) -> str:
    """Classify an input item by its declared type; unknown types fall back to text."""
    t = item.get("type", "text")
    return t if t in MODALITIES else "text"


class MultimodalRAGEngine:
    def __init__(self, processors: Dict[str, Callable[[dict], str]]):
        self.processors = processors  # modality -> fn(item) -> extracted content
        self.metrics = DashboardMetrics()

    def process(self, item: dict) -> Evidence:
        modality = detect_modality(item)
        proc = self.processors.get(modality)
        if proc is None:
            self.metrics.record(modality, False)
            return Evidence(modality=modality, content="", score=0.0)
        try:
            content = proc(item)
            self.metrics.record(modality, True)
            return Evidence(modality=modality, content=content, score=1.0)
        except Exception:  # noqa: BLE001 - a bad media file must not kill the query
            self.metrics.record(modality, False)
            return Evidence(modality=modality, content="", score=0.0)

    def retrieve(self, items: List[dict]) -> List[Evidence]:
        return [self.process(it) for it in items]

    def fuse(self, items: List[dict]) -> str:
        """Combine all modalities' evidence into one retrieval context (best-scored first)."""
        evs = [e for e in self.retrieve(items) if e.content]
        if not evs:
            return ""
        parts = [f"[{e.modality}] {e.content}" for e in sorted(evs, key=lambda x: -x.score)]
        return " | ".join(parts)

    def dashboard(self) -> dict:
        return self.metrics.snapshot()
