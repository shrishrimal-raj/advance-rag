"""Module 2 — Approach C: Language-Aware Code Chunking with LanguageTextSplitter.

Plain character splitters shred code mid-function, producing chunks that are
syntactically broken and semantically useless. LanguageTextSplitter uses a
language grammar (Python, JavaScript, ...) to prefer cutting at function/class
boundaries.

This script feeds Python and JavaScript snippets through the splitter and shows
before/after in syntax-highlighted panels.

Run from project root:
    uv run python 02-document-processing-chunking/code/approach_3_code_splitter.py
"""
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import re

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

PYTHON_SNIPPET = '''\
"""Order processing service for an e-commerce backend."""
from dataclasses import dataclass


@dataclass
class Order:
    order_id: str
    customer_id: str
    total_cents: int
    status: str = "pending"


def validate_order(order: Order) -> bool:
    """An order must have a positive total and a known customer."""
    if order.total_cents <= 0:
        raise ValueError(f"order {order.order_id} has non-positive total")
    if not order.customer_id:
        raise ValueError(f"order {order.order_id} is missing customer_id")
    return True


def apply_discount(order: Order, percent: float) -> Order:
    """Apply a percentage discount and clamp to [0, 100]."""
    clamped = max(0.0, min(100.0, percent))
    order.total_cents = int(order.total_cents * (1 - clamped / 100))
    return order


class OrderService:
    """In-memory order book with lifecycle transitions."""

    def __init__(self):
        self._orders: dict[str, Order] = {}

    def create(self, order: Order) -> None:
        validate_order(order)
        self._orders[order.order_id] = order

    def ship(self, order_id: str) -> Order:
        order = self._orders[order_id]
        if order.status != "pending":
            raise RuntimeError(f"cannot ship order in status {order.status}")
        order.status = "shipped"
        return order

    def refund(self, order_id: str) -> Order:
        order = self._orders[order_id]
        if order.status != "shipped":
            raise RuntimeError(f"only shipped orders can be refunded")
        order.status = "refunded"
        order.total_cents = 0
        return order

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for o in self._orders.values():
            by_status[o.status] = by_status.get(o.status, 0) + 1
        return {"total": len(self._orders), "by_status": by_status}
'''

JS_SNIPPET = '''\
// Event-driven rate limiter with sliding window + exponential backoff.
class RateLimiter {
  constructor({ windowMs = 60000, maxRequests = 100 } = {}) {
    this.windowMs = windowMs;
    this.maxRequests = maxRequests;
    this.events = [];
  }

  record(timestamp = Date.now()) {
    this.events.push(timestamp);
    this._prune(timestamp);
  }

  _prune(now) {
    const cutoff = now - this.windowMs;
    while (this.events.length && this.events[0] < cutoff) {
      this.events.shift();
    }
  }

  remaining(timestamp = Date.now()) {
    this._prune(timestamp);
    return Math.max(0, this.maxRequests - this.events.length);
  }

  backoff(attempt, baseMs = 100, factor = 2, capMs = 5000) {
    const jitter = Math.random() * baseMs;
    return Math.min(capMs, baseMs * Math.pow(factor, attempt)) + jitter;
  }
}

function withRetry(fn, limiter, maxAttempts = 5) {
  let attempt = 0;
  return (async () => {
    while (true) {
      try {
        if (limiter.remaining() === 0) {
          await sleep(limiter.backoff(attempt));
        }
        limiter.record();
        return await fn();
      } catch (err) {
        attempt += 1;
        if (attempt >= maxAttempts) throw err;
        await sleep(limiter.backoff(attempt));
      }
    }
  })();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
'''


def make_splitter(language: str, chunk_size: int):
    """Version-safe language-aware splitter.

    Newer langchain_text_splitters expose dedicated classes
    (PythonCodeTextSplitter / JSFrameworkTextSplitter); older ones expose the
    unified LanguageTextSplitter. Try both."""
    try:
        from langchain_text_splitters import LanguageTextSplitter
        return LanguageTextSplitter(language=language, chunk_size=chunk_size, chunk_overlap=0)
    except ImportError:
        pass
    from langchain_text_splitters import PythonCodeTextSplitter, JSFrameworkTextSplitter
    cls = {"python": PythonCodeTextSplitter, "javascript": JSFrameworkTextSplitter}[language]
    return cls(chunk_size=chunk_size, chunk_overlap=0)


def show_case(title: str, language: str, code: str, chunk_size: int):
    console.rule(f"[bold]{title}[/bold]")
    console.print(Panel(
        Syntax(code, language, word_wrap=True),
        title=f"[dim]BEFORE — full source ({len(code)} chars)[/dim]",
        border_style="dim", expand=False))

    splitter = make_splitter(language, chunk_size)
    chunks = splitter.split_text(code)

    t = Table(title=f"{type(splitter).__name__}(language={language!r}, chunk_size={chunk_size})")
    t.add_column("#", justify="right", style="cyan")
    t.add_column("chars", justify="right")
    t.add_column("starts at")
    for i, c in enumerate(chunks):
        first_line = next((ln.strip() for ln in c.splitlines() if ln.strip()), "")
        t.add_row(str(i), str(len(c)), first_line[:60])
    console.print(t)

    for i, c in enumerate(chunks):
        console.print(Panel(
            Syntax(c, language, word_wrap=True),
            title=f"[cyan]AFTER — chunk {i}[/cyan] [dim]({len(c)} chars)[/dim]",
            border_style="cyan", expand=False))

    # Prove boundaries respect structure: every chunk should start at a
    # def/class/function/method boundary (or top-level comment/docstring).
    boundary_re = re.compile(
        r"^\s*(def\s|class\s|function\s|const\s|//|\"\"\"|/\*|@|[A-Za-z_]\w*\s*\()")
    ok = all(boundary_re.match(ln) for c in chunks
             for ln in [next((l for l in c.splitlines() if l.strip()), "")])
    console.print(
        f"[green]✓ All {len(chunks)} chunks start at a structural boundary[/green]"
        if ok else
        "[yellow]⚠ Some chunk starts mid-structure — lower chunk_size or accept it[/yellow]")
    console.print()


def main():
    console.print(Panel("[bold]Module 2 — Approach C: LanguageTextSplitter (code-aware chunking)[/bold]",
                        border_style="blue"))
    console.print(
        "[dim]Why: character splitters cut mid-function → broken, un-retrievable code chunks.\n"
        "LanguageTextSplitter knows Python/JS grammar and prefers def/class/function seams.[/dim]\n"
    )
    show_case("Case 1 — Python (order service)", "python", PYTHON_SNIPPET, chunk_size=400)
    show_case("Case 2 — JavaScript (rate limiter)", "javascript", JS_SNIPPET, chunk_size=400)

    console.rule("[bold red]Takeaway[/bold red]")
    console.print(
        "• Code repos: always use LanguageTextSplitter (or tree-sitter based splitters).\n"
        "• chunk_size ≈ one function/class; overlap stays 0 (duplicating code is pure noise).\n"
        "• Attach metadata per chunk: file path, symbol name, line range → precise citations."
    )


if __name__ == "__main__":
    main()
