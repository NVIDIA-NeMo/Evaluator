from __future__ import annotations

import importlib
import io
import logging
import os
import random
import re
import shlex
import subprocess
import tarfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Generator, Iterable

from .base import NemoEvaluatorSandbox, NemoSandboxCommand, NemoSandboxSession

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EcsFargateConfig:
    region: str | None
    cluster: str
    task_definition: str
    container_name: str
    subnets: list[str]
    security_groups: list[str]
    assign_public_ip: bool = False

    # Image selection
    image_template: str | None = None  # supports {task_id}
    # If true and image_template is provided, register a per-task task definition and deregister on cleanup.
    register_task_definition_per_task: bool = True

    # Used only when we need to auto-register a task definition from scratch.
    cpu: str = "8192"
    memory: str = "32768"
    execution_role_arn: str | None = None
    task_role_arn: str | None = None
    log_group: str | None = None
    log_stream_prefix: str = "terminalbench"

    # Hard TTL for the sandbox task. Container main process will be `sleep <max_task_lifetime_sec>`.
    max_task_lifetime_sec: int = 180 * 60

    # Retries for ecs.run_task placement/capacity failures.
    run_task_max_retries: int = 30

    # File staging (required for ECS sandbox here)
    s3_bucket: str | None = None
    s3_prefix: str = "terminal-bench"

    # Minimum timeout for each `aws ecs execute-command` subprocess call.
    ecs_exec_timeout_sec: int = 180


class AwsCliMissingError(RuntimeError):
    pass


class EcsExecError(RuntimeError):
    pass


def _require_aws_sdks():
    """
    Lazily import boto3/botocore only when ECS sandbox is actually used.

    This avoids requiring AWS dependencies for non-ECS runs.
    """
    try:
        boto3 = importlib.import_module("boto3")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ECS Fargate sandbox requires AWS SDK dependencies (boto3/botocore).\n"
            "Install them (e.g. `pip install boto3`) or avoid using the ECS backend."
        ) from e

    Config = getattr(botocore_config, "Config")
    ClientError = getattr(botocore_exceptions, "ClientError")
    NoCredentialsError = getattr(botocore_exceptions, "NoCredentialsError")
    PartialCredentialsError = getattr(botocore_exceptions, "PartialCredentialsError")
    return boto3, Config, ClientError, NoCredentialsError, PartialCredentialsError


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _aws_credentials_preflight(region: str | None) -> str:
    """
    Validate that AWS credentials are present and usable.

    Returns:
        str: account id (for logging).
    """
    boto3, _Config, ClientError, NoCredentialsError, PartialCredentialsError = (
        _require_aws_sdks()
    )
    try:
        sts = boto3.client("sts", region_name=region)
        ident = sts.get_caller_identity()
        return str(ident.get("Account", "unknown"))
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise RuntimeError(
            "AWS credentials not found or incomplete. For ECS sandbox you must provide "
            "valid AWS credentials to BOTH boto3 and the AWS CLI.\n\n"
            "Common fixes:\n"
            "- export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (/ AWS_SESSION_TOKEN)\n"
            "- or configure a profile and set AWS_PROFILE\n"
            "- or run in an environment with an attached IAM role\n"
        ) from e
    except ClientError as e:
        raise RuntimeError(
            "AWS credential check failed (sts:GetCallerIdentity). This usually means the credentials "
            "are invalid/expired or the environment cannot reach AWS STS.\n\n"
            f"AWS error: {e}\n"
        ) from e


class _EcsExecContainer:
    """
    Minimal docker.Container-like shim used by some agents (installed agents).
    """

    def __init__(self, sandbox: "EcsFargateSandbox"):
        self._s = sandbox
        self.attrs = {"Config": {"User": ""}}

    def exec_run(self, cmd: list[str], user: str = ""):
        _ = user  # ECS Exec user switching is handled by remote shell / sudo if needed.
        out = self._s._exec_capture(cmd=cmd, timeout_sec=180.0, check=True)
        return type("ExecResult", (), {"exit_code": 0, "output": out.encode()})


class EcsFargateTmuxSession(NemoSandboxSession):
    _TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
    _LONG_TEXT_THRESHOLD = 2000

    def __init__(self, *, session_name: str, sandbox: "EcsFargateSandbox"):
        self._session_name = session_name
        self._sandbox = sandbox
        self.container = _EcsExecContainer(sandbox)
        self._previous_buffer: str | None = None
        self._logger = logging.getLogger(f"{__name__}.EcsFargateTmuxSession")

    def start(self) -> None:
        if self._has_session():
            return
        self._sandbox._exec_capture(
            cmd=[
                "bash",
                "-lc",
                (
                    f"tmux new-session -x 160 -y 40 -d -s {shlex.quote(self._session_name)} "
                    f"\\; set-option -t {shlex.quote(self._session_name)} history-limit 50000"
                ),
            ],
            timeout_sec=60.0,
        )

    def _has_session(self) -> bool:
        payload = self._sandbox._exec_capture(
            cmd=[
                "bash",
                "-lc",
                (
                    f"tmux has-session -t {shlex.quote(self._session_name)} 2>/dev/null "
                    "&& echo __NEMO_YES__ || echo __NEMO_NO__"
                ),
            ],
            timeout_sec=90.0,
            check=True,
        )
        return "__NEMO_YES__" in payload

    def stop(self) -> None:
        return

    def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]

        special_keys = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
        for k in keys:
            if (
                isinstance(k, str)
                and (len(k) > self._LONG_TEXT_THRESHOLD)
                and (k not in special_keys)
            ):
                self._logger.info(
                    "Large keystroke payload detected (%s chars); using tmux paste-buffer",
                    len(k),
                )
                self._sandbox._tmux_paste_large_text(
                    session_name=self._session_name,
                    text=k,
                    timeout_sec=max(300.0, float(max_timeout_sec) + 60.0),
                )

        keys = [
            k
            for k in keys
            if not (
                isinstance(k, str)
                and (len(k) > self._LONG_TEXT_THRESHOLD)
                and (k not in special_keys)
            )
        ]

        if (
            block
            and keys
            and keys[-1] in ("Enter", "C-m", "KPEnter", "C-j", "^M", "^J")
        ):
            keys = keys[:-1] + [self._TMUX_COMPLETION_COMMAND, "Enter"]

        if keys:
            self._sandbox._exec_capture(
                cmd=["tmux", "send-keys", "-t", self._session_name, *keys],
                timeout_sec=60.0,
            )

        if block:
            self._sandbox._exec_capture(
                cmd=["timeout", f"{max_timeout_sec}s", "tmux", "wait", "done"],
                timeout_sec=max_timeout_sec + 30.0,
            )
        elif min_timeout_sec > 0:
            time.sleep(min_timeout_sec)

    def send_command(self, command: NemoSandboxCommand) -> None:
        keys = [command.command, "Enter"] if command.append_enter else [command.command]
        self.send_keys(
            keys=keys,
            block=command.block,
            min_timeout_sec=command.min_timeout_sec,
            max_timeout_sec=command.max_timeout_sec,
        )

    def capture_pane(self, capture_entire: bool = False) -> str:
        cmd = ["tmux", "capture-pane", "-p"]
        if capture_entire:
            cmd.extend(["-S", "-"])
        cmd.extend(["-t", self._session_name])
        return self._sandbox._exec_capture(cmd=cmd, timeout_sec=60.0)

    def is_session_alive(self) -> bool:
        try:
            return self._has_session()
        except Exception:
            return False

    def get_asciinema_timestamp(self) -> float:
        return 0.0

    def _get_visible_screen(self) -> str:
        return self.capture_pane(capture_entire=False)

    def _find_new_content(self, current_buffer: str) -> str | None:
        if self._previous_buffer is None:
            return None

        pb = self._previous_buffer.strip()
        if pb and pb in current_buffer:
            idx = current_buffer.index(pb)
            if "\n" in pb:
                idx = pb.rfind("\n")
            return current_buffer[idx:]
        return None

    def get_incremental_output(self) -> str:
        current_buffer = self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"

        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer

        if new_content is not None:
            if new_content.strip():
                return f"New Terminal Output:\n{new_content}"
            return f"Current Terminal Screen:\n{self._get_visible_screen()}"

        return f"Current Terminal Screen:\n{self._get_visible_screen()}"

    def copy_to_sandbox(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None:
        self._sandbox.copy_to_sandbox(
            paths=paths,
            container_dir=container_dir,
            container_filename=container_filename,
        )


class EcsFargateSandbox(NemoEvaluatorSandbox):
    """
    Sandbox backed by ECS Fargate + ECS Exec.

    No inbound connectivity is required. File transfer is done by uploading a tar to S3
    and downloading it from inside the container using python stdlib.
    """

    def __init__(
        self,
        *,
        cfg: EcsFargateConfig,
        task_arn: str,
        run_id: str,
        task_id: str,
        trial_name: str,
    ):
        if _which("aws") is None:
            raise AwsCliMissingError(
                "AWS CLI ('aws') not found. ECS sandbox requires AWS CLI + session-manager-plugin."
            )
        if _which("session-manager-plugin") is None:
            raise AwsCliMissingError(
                "session-manager-plugin not found. ECS Exec requires session-manager-plugin "
                "to be installed on the harness host (it is invoked by the AWS CLI)."
            )
        self._cfg = cfg
        self._task_arn = task_arn
        self._run_id = run_id
        self._task_id = task_id
        self._trial_name = trial_name
        self._sessions: dict[str, EcsFargateTmuxSession] = {}
        self.container = _EcsExecContainer(self)
        self._logger = logging.getLogger(f"{__name__}.EcsFargateSandbox")

    @classmethod
    def spin_up(
        cls,
        *,
        cfg: EcsFargateConfig,
        task_id: str,
        trial_name: str,
        run_id: str,
        pre_upload_paths: Iterable[Path] | None = None,
        upload_dest_dir: str | None = None,
    ) -> ContextManager["EcsFargateSandbox"]:
        return _spin_up_ecs_fargate_sandbox(
            cfg=cfg,
            task_id=task_id,
            trial_name=trial_name,
            run_id=run_id,
            pre_upload_paths=pre_upload_paths,
            upload_dest_dir=upload_dest_dir,
        )

    def _aws_ecs_execute(
        self, *, command: str, timeout_sec: float
    ) -> subprocess.CompletedProcess:
        args = [
            "aws",
            *(["--region", self._cfg.region] if self._cfg.region else []),
            "ecs",
            "execute-command",
            "--cluster",
            self._cfg.cluster,
            "--task",
            self._task_arn,
            "--container",
            self._cfg.container_name,
            "--interactive",
            "--command",
            command,
        ]
        env = os.environ.copy()
        env.setdefault("AWS_RETRY_MODE", "standard")
        env.setdefault("AWS_MAX_ATTEMPTS", "12")
        effective_timeout = max(
            float(timeout_sec), float(self._cfg.ecs_exec_timeout_sec)
        )
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=False,
            timeout=effective_timeout,
            env=env,
        )

    def _aws_ecs_execute_with_retry(
        self, *, command: str, timeout_sec: float
    ) -> subprocess.CompletedProcess:
        effective_timeout = max(
            float(timeout_sec), float(self._cfg.ecs_exec_timeout_sec)
        )
        start = time.time()
        throttle_sleep = 1.0
        last_cp: subprocess.CompletedProcess | None = None

        def _as_text(x) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x)

        while True:
            try:
                cp = self._aws_ecs_execute(
                    command=command, timeout_sec=effective_timeout
                )
            except subprocess.TimeoutExpired as te:
                self._logger.warning(
                    "ECS Exec timed out after %ss; will retry for a bit",
                    int(effective_timeout),
                )
                cp = subprocess.CompletedProcess(
                    args=te.cmd,
                    returncode=124,
                    stdout=_as_text(getattr(te, "stdout", "")),
                    stderr=_as_text(getattr(te, "stderr", ""))
                    or f"TimeoutExpired: command timed out after {effective_timeout} seconds",
                )

            last_cp = cp
            combined = (
                _as_text(getattr(cp, "stdout", ""))
                + "\n"
                + _as_text(getattr(cp, "stderr", ""))
            ).strip()

            is_throttled = (
                "ThrottlingException" in combined
                or "TooManyRequestsException" in combined
                or "Rate exceeded" in combined
            )
            is_exec_not_ready = (
                "TargetNotConnectedException" in combined
                or "execute command agent isn’t running" in combined
                or "execute command agent isn't running" in combined
                or "execute command was not enabled" in combined
                or "TimeoutExpired" in combined
            )

            if is_throttled and (time.time() - start) < 600:
                sleep_sec = min(30.0, throttle_sleep) + random.random()
                self._logger.warning(
                    "ECS Exec throttled; backing off for %.1fs", sleep_sec
                )
                time.sleep(sleep_sec)
                throttle_sleep = min(30.0, throttle_sleep * 1.7)
                continue

            if is_exec_not_ready and (time.time() - start) < 180:
                time.sleep(3.0)
                continue

            return last_cp

    def _parse_exec_markers(
        self, *, cp: subprocess.CompletedProcess, check: bool
    ) -> str:
        def _as_text(x) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x)

        combined_lines = (
            _as_text(getattr(cp, "stdout", ""))
            + "\n"
            + _as_text(getattr(cp, "stderr", ""))
        ).splitlines()

        filtered_lines: list[str] = []
        for line in combined_lines:
            if line.startswith("The Session Manager plugin was installed successfully"):
                continue
            if line.startswith("Starting session with SessionId:"):
                continue
            if line.startswith("Exiting session with sessionId:"):
                continue
            filtered_lines.append(line)

        text_out = "\n".join(filtered_lines).strip("\n")

        begin_idx = None
        for i, line in enumerate(filtered_lines):
            if line.strip() == "__TB_BEGIN__":
                begin_idx = i
                break

        rc = None
        rc_line_idx = None
        for i in range(len(filtered_lines) - 1, -1, -1):
            line = filtered_lines[i].strip()
            if line.startswith("__TB_RC__="):
                rc_line_idx = i
                try:
                    rc = int(line.split("=", 1)[1])
                except Exception:
                    rc = None
                break

        if (
            begin_idx is not None
            and rc_line_idx is not None
            and rc_line_idx > begin_idx
        ):
            payload_lines = filtered_lines[begin_idx + 1 : rc_line_idx]
            payload = "\n".join(payload_lines).strip("\n")
        else:
            payload = text_out

        if not check:
            return payload

        effective_rc = rc if rc is not None else cp.returncode
        if effective_rc != 0:
            err = text_out.strip()
            if (
                "TargetNotConnectedException" in err
                or "execute command was not enabled" in err
                or "execute command agent" in err
            ):
                raise EcsExecError(
                    "ECS Exec failed. The AWS CLI reports that execute-command is not enabled or the "
                    "exec target is not connected / exec agent is not running.\n\n"
                    "This sandbox runs tasks with enableExecuteCommand=True, so the most common causes are:\n"
                    "- The task is in private subnets without NAT/VPC endpoints to SSM/SSMMessages/EC2Messages\n"
                    "- The task IAM role (taskRoleArn) is missing SSM/SSM Messages permissions required for ECS Exec\n"
                    "- The exec agent is still initializing (retries happen, but it can still time out)\n\n"
                    f"rc={effective_rc}\n"
                    f"OUTPUT:\n{text_out}"
                )
            raise EcsExecError(
                f"ECS Exec failed: rc={effective_rc}\nOUTPUT:\n{text_out}"
            )

        return payload

    def _exec_capture(
        self, *, cmd: list[str], timeout_sec: float, check: bool = True
    ) -> str:
        shell = " ".join(shlex.quote(x) for x in cmd)
        wrapped = (
            "printf '__TB_BEGIN__\\n'; "
            f"{shell}; "
            "rc=$?; "
            "printf '\\n__TB_RC__=%s\\n' \"$rc\""
        )
        command = f"sh -lc {shlex.quote(wrapped)}"

        def _as_text(x) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return str(x)

        if len(command) > 6000:
            return self._exec_capture_via_s3_script(
                shell=shell, timeout_sec=timeout_sec, check=check
            )

        cp = self._aws_ecs_execute_with_retry(command=command, timeout_sec=timeout_sec)
        out = (
            _as_text(getattr(cp, "stdout", ""))
            + "\n"
            + _as_text(getattr(cp, "stderr", ""))
        ).lower()
        if "command too long" in out:
            return self._exec_capture_via_s3_script(
                shell=shell, timeout_sec=timeout_sec, check=check
            )

        return self._parse_exec_markers(cp=cp, check=check)

    def _exec_capture_via_s3_script(
        self, *, shell: str, timeout_sec: float, check: bool
    ) -> str:
        boto3, _Config, _ClientError, _NoCredentialsError, _PartialCredentialsError = (
            _require_aws_sdks()
        )
        if not self._cfg.s3_bucket:
            raise RuntimeError(
                "ECS Exec command exceeded length limits and S3 staging is not configured.\n"
                "Set s3_bucket to enable long-command fallback."
            )

        script = (
            "#!/bin/sh\n"
            "printf '__TB_BEGIN__\\n'\n"
            f"{shell}\n"
            "rc=$?\n"
            "printf '\\n__TB_RC__=%s\\n' \"$rc\"\n"
        )

        s3 = boto3.client("s3", region_name=self._cfg.region)
        key = (
            f"{self._cfg.s3_prefix}/{self._run_id}/{self._task_id}/{self._trial_name}/"
            f"exec/{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.sh"
        )
        s3.put_object(Bucket=self._cfg.s3_bucket, Key=key, Body=script.encode("utf-8"))
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._cfg.s3_bucket, "Key": key},
            ExpiresIn=3600,
        )

        py = (
            "import os,urllib.request\n"
            "url=os.environ['TB_URL']\n"
            "dst=os.environ['TB_DST']\n"
            "with urllib.request.urlopen(url, timeout=180) as r:\n"
            "  data=r.read()\n"
            "open(dst,'wb').write(data)\n"
            "print('ok')\n"
        )
        remote = f"/tmp/nemo_exec_{int(time.time() * 1000)}.sh"
        download_and_run = (
            f"PY=python3; command -v python3 >/dev/null 2>&1 || PY=python; "
            f"TB_URL={shlex.quote(url)} TB_DST={shlex.quote(remote)} "
            f"$PY -c {shlex.quote(py)} >/dev/null 2>&1; "
            f"chmod +x {shlex.quote(remote)} >/dev/null 2>&1 || true; "
            f"sh {shlex.quote(remote)}; "
            f"rm -f {shlex.quote(remote)} >/dev/null 2>&1 || true"
        )
        cp = self._aws_ecs_execute_with_retry(
            command=f"sh -lc {shlex.quote(download_and_run)}",
            timeout_sec=timeout_sec,
        )
        return self._parse_exec_markers(cp=cp, check=check)

    def _s3_stage_text(self, *, text: str, suffix: str) -> str:
        boto3, _Config, _ClientError, _NoCredentialsError, _PartialCredentialsError = (
            _require_aws_sdks()
        )
        if not self._cfg.s3_bucket:
            raise RuntimeError(
                "S3 staging is required for large-text tmux paste fallback. Set s3_bucket."
            )
        s3 = boto3.client("s3", region_name=self._cfg.region)
        key = (
            f"{self._cfg.s3_prefix}/{self._run_id}/{self._task_id}/{self._trial_name}/"
            f"{suffix}/{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.txt"
        )
        s3.put_object(Bucket=self._cfg.s3_bucket, Key=key, Body=text.encode("utf-8"))
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._cfg.s3_bucket, "Key": key},
            ExpiresIn=3600,
        )

    def _tmux_paste_large_text(
        self, *, session_name: str, text: str, timeout_sec: float
    ) -> None:
        url = self._s3_stage_text(text=text, suffix="tmux-paste")
        remote = f"/tmp/nemo_paste_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.txt"

        py = (
            "import os,urllib.request\n"
            "url=os.environ['TB_URL']\n"
            "dst=os.environ['TB_DST']\n"
            "with urllib.request.urlopen(url, timeout=180) as r:\n"
            "  data=r.read()\n"
            "open(dst,'wb').write(data)\n"
            "print('ok')\n"
        )

        self._exec_capture(
            cmd=[
                "bash",
                "-lc",
                (
                    f"PY=python3; command -v python3 >/dev/null 2>&1 || PY=python; "
                    f"TB_URL={shlex.quote(url)} TB_DST={shlex.quote(remote)} "
                    f"$PY -c {shlex.quote(py)} >/dev/null"
                ),
            ],
            timeout_sec=min(600.0, float(timeout_sec)),
        )

        buf_name = f"nemo_paste_{uuid.uuid4().hex[:8]}"
        self._exec_capture(
            cmd=["tmux", "load-buffer", "-b", buf_name, remote], timeout_sec=60.0
        )
        self._exec_capture(
            cmd=["tmux", "paste-buffer", "-b", buf_name, "-t", session_name],
            timeout_sec=60.0,
        )

        self._exec_capture(
            cmd=["tmux", "delete-buffer", "-b", buf_name], timeout_sec=60.0, check=False
        )
        self._exec_capture(cmd=["rm", "-f", remote], timeout_sec=60.0, check=False)

    def create_session(
        self,
        session_name: str,
        is_active_stream: bool = False,
        as_configured_user: bool = True,
    ) -> EcsFargateTmuxSession:
        _ = is_active_stream
        _ = as_configured_user
        if session_name in self._sessions:
            raise ValueError(f"Session {session_name} already exists")
        session = EcsFargateTmuxSession(session_name=session_name, sandbox=self)
        session.start()
        self._sessions[session_name] = session
        return session

    def copy_to_sandbox(
        self,
        *,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None:
        boto3, _Config, _ClientError, _NoCredentialsError, _PartialCredentialsError = (
            _require_aws_sdks()
        )
        if container_dir is None:
            raise ValueError("container_dir is required")
        if isinstance(paths, Path):
            paths = [paths]

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            if container_filename is not None:
                if len(paths) != 1 or not paths[0].is_file():
                    raise ValueError(
                        "container_filename requires exactly one file path"
                    )
                tar.add(paths[0], arcname=container_filename)
            else:
                for p in paths:
                    if p.is_file():
                        tar.add(p, arcname=p.name)
                    elif p.is_dir():
                        for item in p.rglob("*"):
                            tar.add(item, arcname=item.relative_to(p))
        buf.seek(0)
        tar_bytes = buf.read()

        if not self._cfg.s3_bucket:
            raise RuntimeError(
                "ECS sandbox requires S3 staging for bundle upload, but no s3_bucket was provided."
            )

        s3 = boto3.client("s3", region_name=self._cfg.region)
        key = (
            f"{self._cfg.s3_prefix}/{self._run_id}/{self._task_id}/{self._trial_name}/"
            f"{int(time.time() * 1000)}.tar"
        )
        s3.put_object(Bucket=self._cfg.s3_bucket, Key=key, Body=tar_bytes)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._cfg.s3_bucket, "Key": key},
            ExpiresIn=3600,
        )

        py = (
            "import os,tarfile,urllib.request,io\n"
            "url=os.environ['TB_URL']\n"
            "dest=os.environ['TB_DEST']\n"
            "with urllib.request.urlopen(url, timeout=180) as r:\n"
            "  data=r.read()\n"
            "os.makedirs(dest, exist_ok=True)\n"
            "tarfile.open(fileobj=io.BytesIO(data), mode='r:*').extractall(dest)\n"
            "print('ok')\n"
        )
        self._exec_capture(
            cmd=[
                "bash",
                "-lc",
                (
                    f"PY=python3; command -v python3 >/dev/null 2>&1 || PY=python; "
                    f"TB_URL={shlex.quote(url)} TB_DEST={shlex.quote(container_dir)} "
                    f"$PY -c {shlex.quote(py)}"
                ),
            ],
            timeout_sec=300.0,
        )

    def stop(self) -> None:
        self._sessions.clear()


@contextmanager
def _spin_up_ecs_fargate_sandbox(
    *,
    cfg: EcsFargateConfig,
    task_id: str,
    trial_name: str,
    run_id: str,
    pre_upload_paths: Iterable[Path] | None = None,
    upload_dest_dir: str | None = None,
) -> Generator[EcsFargateSandbox, None, None]:
    boto3, Config, ClientError, _NoCredentialsError, _PartialCredentialsError = (
        _require_aws_sdks()
    )
    account_id = _aws_credentials_preflight(cfg.region)
    ecs = boto3.client(
        "ecs",
        region_name=cfg.region,
        config=Config(retries={"max_attempts": 12, "mode": "standard"}),
    )

    bootstrap = (
        "if command -v python >/dev/null 2>&1; then :; "
        "elif command -v python3 >/dev/null 2>&1; then "
        "  P=$(command -v python3); "
        '  ln -sf "$P" /usr/local/bin/python 2>/dev/null || true; '
        '  ln -sf "$P" /usr/bin/python 2>/dev/null || true; '
        "fi"
    )
    keepalive_command = [
        "sh",
        "-lc",
        f"{bootstrap}; sleep {int(cfg.max_task_lifetime_sec)}",
    ]

    overrides: dict = {"containerOverrides": [{"name": cfg.container_name}]}
    overrides["containerOverrides"][0]["command"] = list(keepalive_command)
    overrides["containerOverrides"][0]["environment"] = [
        {"name": "TEST_DIR", "value": "/tests"}
    ]

    task_definition_to_run = cfg.task_definition
    registered_task_definition_arn: str | None = None

    if cfg.image_template and cfg.register_task_definition_per_task:
        image = cfg.image_template.format(task_id=task_id)

        if cfg.log_group:
            try:
                logs = boto3.client("logs", region_name=cfg.region)
                logs.create_log_group(logGroupName=cfg.log_group)
            except Exception:
                pass

        try:
            base = ecs.describe_task_definition(taskDefinition=cfg.task_definition)[
                "taskDefinition"
            ]
        except ClientError:
            base = None

        raw_family = f"tb-{run_id}-{task_id}-{trial_name}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        family = re.sub(r"[^A-Za-z0-9_-]", "_", raw_family)
        if not family or not re.match(r"^[A-Za-z0-9]", family):
            family = f"tb_{family}"
        family = family[:255]
        log.info(
            "Registering per-task ECS task definition family=%s (raw=%s)",
            family,
            raw_family,
        )

        if base is not None:
            container_defs = base.get("containerDefinitions") or []
            found = False
            for cd in container_defs:
                if cd.get("name") == cfg.container_name:
                    cd["image"] = image
                    cd["command"] = list(keepalive_command)
                    if cfg.log_group:
                        cd["logConfiguration"] = {
                            "logDriver": "awslogs",
                            "options": {
                                "awslogs-group": cfg.log_group,
                                "awslogs-region": cfg.region or "",
                                "awslogs-stream-prefix": cfg.log_stream_prefix,
                            },
                        }
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "Base task definition does not contain the configured container_name.\n\n"
                    f"container_name: {cfg.container_name}\n"
                    f"task_definition: {cfg.task_definition}\n"
                    f"available_containers: {[c.get('name') for c in container_defs]}\n"
                )

            register_payload: dict = {
                "family": family,
                "networkMode": base.get("networkMode", "awsvpc"),
                "requiresCompatibilities": base.get(
                    "requiresCompatibilities", ["FARGATE"]
                ),
                "cpu": base.get("cpu"),
                "memory": base.get("memory"),
                "containerDefinitions": container_defs,
            }
            for k in (
                "taskRoleArn",
                "executionRoleArn",
                "ephemeralStorage",
                "runtimePlatform",
                "volumes",
                "placementConstraints",
                "proxyConfiguration",
                "pidMode",
                "ipcMode",
                "inferenceAccelerators",
            ):
                if k in base and base[k] is not None:
                    register_payload[k] = base[k]
        else:
            execution_role_arn = cfg.execution_role_arn or ""
            if not execution_role_arn:
                raise RuntimeError(
                    "Unable to describe base task definition and no execution role ARN provided.\n\n"
                    "To auto-register per-task Fargate task definitions, provide an execution role ARN.\n"
                    f"task_definition (missing): {cfg.task_definition}\n"
                )

            container_def: dict = {
                "name": cfg.container_name,
                "image": image,
                "essential": True,
                "command": list(keepalive_command),
            }
            if cfg.log_group:
                container_def["logConfiguration"] = {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": cfg.log_group,
                        "awslogs-region": cfg.region or "",
                        "awslogs-stream-prefix": cfg.log_stream_prefix,
                    },
                }

            register_payload = {
                "family": family,
                "networkMode": "awsvpc",
                "requiresCompatibilities": ["FARGATE"],
                "cpu": str(cfg.cpu),
                "memory": str(cfg.memory),
                "executionRoleArn": execution_role_arn,
                "containerDefinitions": [container_def],
            }
            if cfg.task_role_arn:
                register_payload["taskRoleArn"] = cfg.task_role_arn

        reg = None
        last_register_error: Exception | None = None
        for attempt in range(1, 16):
            try:
                reg = ecs.register_task_definition(**register_payload)
                break
            except ClientError as e:
                last_register_error = e
                code = (e.response.get("Error") or {}).get("Code", "")
                msg = str(e)
                retryable = (
                    "Too many concurrent attempts to create a new revision" in msg
                    or "Rate exceeded" in msg
                    or code in {"ThrottlingException", "TooManyRequestsException"}
                )
                if not retryable or attempt >= 15:
                    raise RuntimeError(
                        "Failed to register per-task ECS task definition.\n\n"
                        f"family: {family}\n"
                        f"AWS error: {e}\n"
                    ) from e
                sleep_sec = (
                    min(30.0, 0.75 * (2 ** min(6, attempt - 1))) + random.random()
                )
                log.warning(
                    "ECS RegisterTaskDefinition concurrency limit hit; retrying %s/15 in %.1fs "
                    "(family=%s, code=%s)",
                    attempt,
                    sleep_sec,
                    family,
                    code,
                )
                time.sleep(sleep_sec)

        if reg is None:
            raise RuntimeError(
                "Failed to register per-task ECS task definition after retries.\n\n"
                f"family: {family}\n"
                f"last_error: {last_register_error}\n"
            )

        registered_task_definition_arn = reg["taskDefinition"]["taskDefinitionArn"]
        task_definition_to_run = registered_task_definition_arn

    last_failures = None
    last_client_error: Exception | None = None
    for attempt in range(1, max(1, int(cfg.run_task_max_retries)) + 1):
        try:
            resp = ecs.run_task(
                cluster=cfg.cluster,
                taskDefinition=task_definition_to_run,
                launchType="FARGATE",
                platformVersion="LATEST",
                enableExecuteCommand=True,
                overrides=overrides,
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": cfg.subnets,
                        "securityGroups": cfg.security_groups,
                        "assignPublicIp": "ENABLED"
                        if cfg.assign_public_ip
                        else "DISABLED",
                    }
                },
            )
            failures = resp.get("failures") or []
            if failures:
                last_failures = failures
                reasons = " | ".join(str(f.get("reason", "")) for f in failures)
                retryable = "Capacity is unavailable" in reasons
                if not retryable or attempt >= int(cfg.run_task_max_retries):
                    raise RuntimeError(
                        f"ECS run_task failures for task_id={task_id}: {failures}"
                    )
                sleep_sec = min(60.0, (2.0 ** min(6, attempt - 1))) + random.random()
                log.warning(
                    "ECS capacity unavailable for task_id=%s; retrying %s/%s in %.1fs",
                    task_id,
                    attempt,
                    cfg.run_task_max_retries,
                    sleep_sec,
                )
                time.sleep(sleep_sec)
                continue

            tasks = resp.get("tasks") or []
            if not tasks:
                raise RuntimeError("ECS run_task returned no tasks")
            task_arn = tasks[0]["taskArn"]
            break
        except ClientError as e:
            last_client_error = e
            msg = str(e)
            retryable = "Capacity is unavailable" in msg
            if not retryable or attempt >= int(cfg.run_task_max_retries):
                raise RuntimeError(
                    "Failed to run ECS task. This is usually caused by missing IAM permissions "
                    "(ecs:RunTask / iam:PassRole), an invalid cluster/task definition, or invalid "
                    "subnets/security groups.\n\n"
                    f"AWS account: {account_id}\n"
                    f"cluster: {cfg.cluster}\n"
                    f"task_definition: {task_definition_to_run}\n"
                    f"container_name: {cfg.container_name}\n"
                    f"task_id: {task_id}\n"
                    f"AWS error: {e}\n"
                ) from e
            sleep_sec = min(60.0, (2.0 ** min(6, attempt - 1))) + random.random()
            log.warning(
                "ECS run_task capacity error for task_id=%s; retrying %s/%s in %.1fs",
                task_id,
                attempt,
                cfg.run_task_max_retries,
                sleep_sec,
            )
            time.sleep(sleep_sec)
            continue
    else:
        raise RuntimeError(
            "Failed to run ECS task after retries.\n\n"
            f"task_id: {task_id}\n"
            f"cluster: {cfg.cluster}\n"
            f"task_definition: {task_definition_to_run}\n"
            f"last_failures: {last_failures}\n"
            f"last_client_error: {last_client_error}\n"
        )

    log.info(
        "Started ECS task: %s (account=%s, task_id=%s)", task_arn, account_id, task_id
    )

    start = time.time()
    while True:
        d = ecs.describe_tasks(cluster=cfg.cluster, tasks=[task_arn])
        t = (d.get("tasks") or [None])[0]
        if t is None:
            raise RuntimeError("ECS task disappeared")
        status = t.get("lastStatus")
        if status == "RUNNING":
            break
        if status == "STOPPED":
            raise RuntimeError(f"ECS task stopped early: {t.get('stoppedReason')}")
        if time.time() - start > 300:
            raise TimeoutError("Timed out waiting for ECS task to be RUNNING")
        time.sleep(2.0)

    sandbox = EcsFargateSandbox(
        cfg=cfg,
        task_arn=task_arn,
        run_id=run_id,
        task_id=task_id,
        trial_name=trial_name,
    )

    try:
        if pre_upload_paths and upload_dest_dir:
            sandbox.copy_to_sandbox(
                paths=list(pre_upload_paths), container_dir=upload_dest_dir
            )
        yield sandbox
    finally:
        try:
            ecs.stop_task(
                cluster=cfg.cluster, task=task_arn, reason="nemo sandbox done"
            )
        except Exception:
            log.warning("Failed to stop ECS task", exc_info=True)

        if registered_task_definition_arn is not None:
            try:
                ecs.deregister_task_definition(
                    taskDefinition=registered_task_definition_arn
                )
            except Exception:
                log.warning(
                    "Failed to deregister temporary ECS task definition", exc_info=True
                )
