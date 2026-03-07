from __future__ import annotations

import logging
import sys
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class ProgressTracker(Protocol):
    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None: ...
    def on_step(self, problem: int, repeat: int, total_problems: int, total_repeats: int,
                reward: float, tokens: int, latency_ms: float) -> None: ...
    def on_done(self, correct: int, total: int, elapsed: float, total_tokens: int) -> None: ...


class ConsoleProgress:
    """TTY-aware progress tracker: live bar on terminal, periodic logs otherwise."""

    def __init__(self, log_interval: float = 30.0) -> None:
        self._t0 = 0.0
        self._tokens = 0
        self._steps = 0
        self._line_len = 0
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._log_interval = log_interval
        self._last_log = 0.0

    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None:
        self._t0 = time.monotonic()
        self._last_log = self._t0
        total = total_problems * total_repeats
        msg = f"{benchmark} | {total_problems} problems x {total_repeats} repeats = {total} steps"
        if self._is_tty:
            sys.stderr.write(f"\n{'='*60}\n  {msg}\n{'='*60}\n\n")
            sys.stderr.flush()
        else:
            logger.info(msg)

    def on_step(self, problem: int, repeat: int, total_problems: int, total_repeats: int,
                reward: float, tokens: int, latency_ms: float) -> None:
        self._tokens += tokens
        self._steps += 1
        elapsed = time.monotonic() - self._t0
        total = total_problems * total_repeats
        pct = 100 * self._steps / total if total else 0

        if self._is_tty:
            self._write_bar(pct, total, problem, repeat, reward, elapsed)
        else:
            now = time.monotonic()
            if now - self._last_log >= self._log_interval or self._steps == total:
                rate = self._steps / elapsed if elapsed > 0 else 0
                logger.info(
                    "progress: %d/%d (%.1f%%) | %.1f/s | %d tok",
                    self._steps, total, pct, rate, self._tokens,
                )
                self._last_log = now

    def _write_bar(self, pct: float, total: int, problem: int, repeat: int,
                   reward: float, elapsed: float) -> None:
        rate = self._steps / elapsed if elapsed > 0 else 0
        tok_rate = self._tokens / elapsed if elapsed > 0 else 0
        w = 30
        filled = int(w * self._steps / total) if total else 0
        bar = "█" * filled + "░" * (w - filled)
        line = (
            f"\r  [{bar}] {pct:5.1f}% | "
            f"{self._steps}/{total} | "
            f"p{problem+1} r{repeat+1} | "
            f"rw={reward:.1f} | "
            f"{rate:.1f}/s | "
            f"{self._tokens:,} tok ({tok_rate:.0f}/s)"
        )
        pad = max(0, self._line_len - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._line_len = len(line)

    def on_done(self, correct: int, total: int, elapsed: float, total_tokens: int) -> None:
        acc = correct / total if total else 0
        msg = f"Done {elapsed:.1f}s | {acc:.1%} ({correct}/{total}) | {total_tokens:,} tok"
        if self._is_tty:
            sys.stderr.write(f"\n\n{'='*60}\n  {msg}\n{'='*60}\n\n")
            sys.stderr.flush()
        else:
            logger.info(msg)


class NoOpProgress:
    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None: pass
    def on_step(self, problem: int, repeat: int, total_problems: int, total_repeats: int,
                reward: float, tokens: int, latency_ms: float) -> None: pass
    def on_done(self, correct: int, total: int, elapsed: float, total_tokens: int) -> None: pass
