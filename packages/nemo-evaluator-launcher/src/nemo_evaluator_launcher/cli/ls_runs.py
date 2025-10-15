# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import datetime as _dt
import sys
from dataclasses import dataclass
from typing import Optional

from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class Cmd:
    """List invocations (runs) from the exec DB as a table."""

    limit: Optional[int] = field(default=None, alias=["--limit"], help="Max rows")
    executor: Optional[str] = field(
        default=None,
        alias=["--executor"],
        help="Filter by executor",
    )
    # TODO(agronskiy): think about if we can propagate a `--status` filter into here.
    since: Optional[str] = field(
        default=None,
        alias=["--since"],
        help="Filter by either ISO date/time (e.g., 2025-08-20 or 2025-08-20T12:00:00) or "
        "an interval into the past, e.g. `1d` or `3h`; formally `{N}[d|h]`.",
    )

    def execute(self) -> None:
        # Import heavy dependencies only when needed
        from nemo_evaluator_launcher.api.functional import (
            get_invocation_benchmarks,
            list_all_invocations_summary,
        )

        rows = list_all_invocations_summary()

        if self.executor:
            rows = [
                r
                for r in rows
                if (r.get("executor") or "").lower() == self.executor.lower()
            ]

        if self.since:
            try:
                # Check if it's a relative time format like "1d" or "3h"
                if self.since.lower().endswith("d") and len(self.since) > 1:
                    days = int(self.since[:-1])
                    if days < 0:
                        raise ValueError("Days should be non-negative")
                    since_ts = (
                        _dt.datetime.now() - _dt.timedelta(days=days)
                    ).timestamp()
                elif self.since.lower().endswith("h") and len(self.since) > 1:
                    hours = int(self.since[:-1])
                    if hours < 0:
                        raise ValueError("Hours should be non-negative")
                    since_ts = (
                        _dt.datetime.now() - _dt.timedelta(hours=hours)
                    ).timestamp()
                elif "T" in self.since:
                    since_ts = _dt.datetime.fromisoformat(self.since).timestamp()
                else:
                    since_ts = _dt.datetime.fromisoformat(
                        self.since + "T00:00:00"
                    ).timestamp()
                rows = [r for r in rows if (r.get("earliest_job_ts") or 0) >= since_ts]
            except Exception:
                logger.fatal(
                    f"Invalid --since value: {self.since}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or N[d|h] for N days|hours."
                )
                sys.exit(2)

        if self.limit is not None and self.limit >= 0:
            rows = rows[: self.limit]

        header = [
            "invocation_id",
            "earliest_job_ts",
            "num_jobs",
            "executor",
            "benchmarks",
        ]
        table_rows = []
        for r in rows:
            ts = r.get("earliest_job_ts", 0) or 0
            try:
                ts_iso = (
                    _dt.datetime.fromtimestamp(ts).replace(microsecond=0).isoformat()
                )
            except Exception:
                ts_iso = ""
            inv = r.get("invocation_id", "")
            try:
                bmarks = get_invocation_benchmarks(inv)
                bmarks_cell = ",".join(bmarks) if bmarks else "unknown"
            except Exception:
                bmarks_cell = "unknown"
            table_rows.append(
                [
                    str(inv),
                    ts_iso,
                    str(r.get("num_jobs", 0)),
                    str(r.get("executor", "")),
                    bmarks_cell,
                ]
            )

        widths = [len(h) for h in header]
        for tr in table_rows:
            for i, cell in enumerate(tr):
                if len(cell) > widths[i]:
                    widths[i] = len(cell)
        fmt = "  ".join([f"{{:<{w}}}" for w in widths])
        print(fmt.format(*header))
        print("  ".join(["-" * w for w in widths]))
        for tr in table_rows:
            print(fmt.format(*tr))
