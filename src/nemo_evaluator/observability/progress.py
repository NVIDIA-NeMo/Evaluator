# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import sys
import time
from typing import Protocol

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as e.g. '1h23m', '45m12s', or '12s'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


class ProgressTracker(Protocol):
    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None: ...
    def on_step(
        self,
        problem: int,
        repeat: int,
        total_problems: int,
        total_repeats: int,
        reward: float,
        tokens: int,
        latency_ms: float,
    ) -> None: ...
    def on_phase(self, problem: int, repeat: int, total_problems: int, total_repeats: int, phase: str) -> None: ...
    def on_done(
        self, correct: int, total: int, elapsed: float, total_tokens: int, mean_reward: float | None = None
    ) -> None: ...


class ConsoleProgress:
    """TTY-aware progress tracker with 4-state bar.

    Per-step phases (dark→light as progress advances):
      ░ not started   ▒ solving   ▓ verifying/judging   █ done

    Text summary:  {done}D {judging}J {solving}S {waiting}W
    """

    _PHASE_ORDER = {"solving": 1, "verifying": 2, "judging": 2}

    def __init__(self, log_interval: float = 60.0) -> None:
        self._t0 = 0.0
        self._tokens = 0
        self._steps = 0
        self._correct = 0
        self._total = 0
        self._line_len = 0
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._log_interval = log_interval
        self._last_log = 0.0
        self._phases: dict[tuple[int, int], int] = {}

    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None:
        self._t0 = time.monotonic()
        self._last_log = self._t0
        self._total = total_problems * total_repeats
        msg = f"{benchmark} | {total_problems} problems x {total_repeats} repeats = {self._total} steps"
        if self._is_tty:
            sys.stderr.write(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}\n\n")
            sys.stderr.flush()
        else:
            logger.info(msg)

    def on_step(
        self,
        problem: int,
        repeat: int,
        total_problems: int,
        total_repeats: int,
        reward: float,
        tokens: int,
        latency_ms: float,
    ) -> None:
        self._tokens += tokens
        self._steps += 1
        if reward > 0:
            self._correct += 1
        self._phases.pop((problem, repeat), None)
        elapsed = time.monotonic() - self._t0

        if self._is_tty:
            self._redraw(elapsed)
        else:
            total = total_problems * total_repeats
            pct = 100 * self._steps / total if total else 0
            now = time.monotonic()
            if now - self._last_log >= self._log_interval or self._steps == total:
                rate = self._steps / elapsed if elapsed > 0 else 0
                eta = (total - self._steps) / rate if rate > 0 else 0
                eta_str = _fmt_duration(eta) if eta > 0 else "—"
                elapsed_str = _fmt_duration(elapsed)
                acc = self._correct / self._steps if self._steps else 0
                logger.info(
                    "progress: %d/%d (%.1f%%) | pass@1=%.1f%% (%d/%d) | %.1f/s | %s elapsed | ETA %s | %d tok",
                    self._steps,
                    total,
                    pct,
                    100 * acc,
                    self._correct,
                    self._steps,
                    rate,
                    elapsed_str,
                    eta_str,
                    self._tokens,
                )
                self._last_log = now

    def on_phase(self, problem: int, repeat: int, total_problems: int, total_repeats: int, phase: str) -> None:
        self._phases[(problem, repeat)] = self._PHASE_ORDER.get(phase, 1)
        if not self._is_tty:
            return
        elapsed = time.monotonic() - self._t0
        self._redraw(elapsed)

    def _redraw(self, elapsed: float) -> None:
        total = self._total
        if total == 0:
            return

        n_done = self._steps
        n_solving = sum(1 for v in self._phases.values() if v == 1)
        n_judging = sum(1 for v in self._phases.values() if v == 2)
        n_wait = total - n_done - n_solving - n_judging

        w = 30
        w_done = round(w * n_done / total)
        w_judge = round(w * (n_done + n_judging) / total) - w_done
        w_solve = round(w * (n_done + n_judging + n_solving) / total) - w_done - w_judge
        w_wait = w - w_done - w_judge - w_solve
        bar = "█" * w_done + "▓" * w_judge + "▒" * w_solve + "░" * w_wait

        pct = 100 * n_done / total
        rate = n_done / elapsed if elapsed > 0 else 0
        tok_rate = self._tokens / elapsed if elapsed > 0 else 0

        parts = [f"{n_done} done"]
        if n_judging:
            parts.append(f"{n_judging} judge")
        if n_solving:
            parts.append(f"{n_solving} solve")
        if n_wait:
            parts.append(f"{n_wait} wait")
        counts = ", ".join(parts)

        line = f"\r  [{bar}] {pct:5.1f}% | {counts} | {rate:.1f}/s | {self._tokens:,} tok ({tok_rate:.0f}/s)"
        pad = max(0, self._line_len - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._line_len = len(line)

    def on_done(
        self, correct: int, total: int, elapsed: float, total_tokens: int, mean_reward: float | None = None
    ) -> None:
        mr = f" | mean_reward={mean_reward:.4f}" if mean_reward is not None else ""
        acc = correct / total if total else 0
        msg = f"Done {elapsed:.1f}s | pass@1={acc:.1%} ({correct}/{total}){mr} | {total_tokens:,} tok"
        if self._is_tty:
            sys.stderr.write(f"\n\n{'=' * 60}\n  {msg}\n{'=' * 60}\n\n")
            sys.stderr.flush()
        else:
            logger.info(msg)


class NoOpProgress:
    def on_start(self, benchmark: str, total_problems: int, total_repeats: int) -> None:
        pass

    def on_step(
        self,
        problem: int,
        repeat: int,
        total_problems: int,
        total_repeats: int,
        reward: float,
        tokens: int,
        latency_ms: float,
    ) -> None:
        pass

    def on_phase(self, problem: int, repeat: int, total_problems: int, total_repeats: int, phase: str) -> None:
        pass

    def on_done(
        self, correct: int, total: int, elapsed: float, total_tokens: int, mean_reward: float | None = None
    ) -> None:
        pass
