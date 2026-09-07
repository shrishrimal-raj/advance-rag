import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal_rag_engine.core import MultimodalRAGEngine, detect_modality  # noqa: E402


def _procs():
    return {
        "text": lambda it: it.get("text", "").strip(),
        "image": lambda it: f"image:{it.get('src', '')}",
        "audio": lambda it: f"transcript:{it.get('wav', '')}",
    }


def test_detect_modality_defaults_text():
    assert detect_modality({"type": "image"}) == "image"
    assert detect_modality({}) == "text"
    assert detect_modality({"type": "bogus"}) == "text"


def test_process_routes_by_modality():
    eng = MultimodalRAGEngine(_procs())
    e = eng.process({"type": "image", "src": "a.png"})
    assert e.modality == "image" and e.content == "image:a.png"


def test_fuse_combines_modalities():
    eng = MultimodalRAGEngine(_procs())
    ctx = eng.fuse([{"type": "text", "text": "hello"}, {"type": "audio", "wav": "x.wav"}])
    assert "[text] hello" in ctx and "[audio] transcript:x.wav" in ctx


def test_metrics_track_success_and_error():
    def bad_audio(it):
        raise RuntimeError("decode fail")

    eng = MultimodalRAGEngine({"text": lambda it: it.get("text", ""), "audio": bad_audio})
    eng.fuse([{"type": "text", "text": "ok"}, {"type": "audio", "wav": "y.wav"}])
    snap = eng.dashboard()
    assert snap["processed"] == 2 and snap["errors"] == 1
    assert snap["by_modality"] == {"text": 1, "audio": 1}
    assert snap["success_rate"] == 0.5


def test_unknown_processor_counts_error():
    eng = MultimodalRAGEngine({"text": lambda it: "t"})
    eng.process({"type": "image", "src": "z.png"})
    assert eng.dashboard()["errors"] == 1
