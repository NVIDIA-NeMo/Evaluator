import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator_launcher.api.functional import RunConfig
from nemo_evaluator_launcher.common.execdb import generate_invocation_id
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.watcher.configs import WatchConfig

WATCH_STATE_DIR = Path.home() / ".nemo-evaluator" / "watch-state"


class SubmittedCheckpoint(BaseModel):
    """Record of a submitted checkpoint evaluation."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: str = Field(description="Path to the checkpoint directory.")
    invocation_id: str = Field(description="Invocation ID of the submitted evaluation.")
    timestamp: str = Field(description="Timestamp of the submission.")
    watch_config: WatchConfig = Field(description="Watch configuration used.")
    eval_config: RunConfig = Field(description="Evaluation configuration used.")


class WatchStateDB:
    """Append-only JSONL log of submitted checkpoint evaluations.

    Each call to :meth:`append` writes one JSON line to ``WATCH_STATE_FILE`` and
    updates the in-memory index, mirroring the pattern used by
    :class:`~nemo_evaluator_launcher.common.execdb.ExecutionDB`.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            session_id = generate_invocation_id()
            path = WATCH_STATE_DIR / f"watch-state.{session_id}.v1.jsonl"
        self._path = path
        self._submitted: dict[str, SubmittedCheckpoint] = {}  # keyed by invocation_id
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        entry = SubmittedCheckpoint(**record)
                        self._submitted[entry.invocation_id] = entry
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse watch state line", error=str(e))
        except OSError as e:
            logger.warning(
                f"Failed to load watch state from {self._path}, starting fresh",
                error=str(e),
            )

    def submitted_paths(self) -> set[str]:
        return {r.checkpoint for r in self._submitted.values()}

    def append(self, record: SubmittedCheckpoint) -> None:
        """Append a submitted checkpoint to the log and update in-memory state."""
        self._submitted[record.invocation_id] = record
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record.model_dump()) + "\n")
        except OSError as e:
            logger.error(
                "Failed to write to watch state", path=str(self._path), error=str(e)
            )
            raise
