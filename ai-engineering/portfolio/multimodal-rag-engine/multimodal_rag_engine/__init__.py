"""The Multimodal RAG Engine: text/image/audio routing, evidence fusion, live dashboards."""
from .core import DashboardMetrics, Evidence, MultimodalRAGEngine, detect_modality

__all__ = ["DashboardMetrics", "Evidence", "MultimodalRAGEngine", "detect_modality"]
