"""SLURM executor: generate sbatch scripts, submit, and manage jobs."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from nemo_evaluator.executors import Executor, ProcessState

logger = logging.getLogger(__name__)

_META_FILE = "slurm_job.json"
_RUNNING_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
_RESUMABLE_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}


def _jobs_store() -> Path:
    """Canonical local jobs store, respecting XDG_DATA_HOME."""
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "nel" / "jobs"


def _get_job_ids(meta: dict) -> list[str]:
    """Return all job IDs from RunMeta.details or an aggregate slurm_job.json."""
    ids = meta.get("job_ids", [])
    if ids:
        return list(ids)
    jid = meta.get("job_id", "")
    return [jid] if jid else []


def _is_sharded(meta: dict) -> bool:
    if meta.get("is_sharded"):
        return True
    return len(_get_job_ids(meta)) > 1


def _resolve_latest_job_id(
    hostname: str,
    remote_dir: str,
    fallback_id: str,
    username: str | None = None,
) -> str:
    """Follow .nel_job_chain on the remote host to get the latest job ID."""
    if not remote_dir or not hostname:
        return fallback_id
    try:
        from nemo_evaluator.executors.ssh import ssh_run

        chain = ssh_run(
            hostname,
            f"tail -1 {shlex.quote(remote_dir + '/.nel_job_chain')} 2>/dev/null",
            username=username,
            timeout=10.0,
        ).strip()
        return chain if chain else fallback_id
    except Exception:
        return fallback_id


def _resolve_shard_latest_ids(
    hostname: str,
    parent_dir: str,
    job_ids: list[str],
    username: str | None = None,
) -> list[str]:
    """Batch-resolve latest job IDs for all shards in one SSH call."""
    n = len(job_ids)
    if n == 0:
        return []
    if n == 1:
        return [_resolve_latest_job_id(hostname, parent_dir, job_ids[0], username)]

    cmds = []
    for i, fb in enumerate(job_ids):
        chain_path = shlex.quote(f"{parent_dir}/shard_{i}/.nel_job_chain")
        cmds.append(f'c=$(tail -1 {chain_path} 2>/dev/null); echo "${{c:-{fb}}}"')
    try:
        from nemo_evaluator.executors.ssh import ssh_run

        output = ssh_run(hostname, "; ".join(cmds), username=username, timeout=15.0)
        lines = output.strip().splitlines()
        if len(lines) == n:
            return [ln.strip() or fb for ln, fb in zip(lines, job_ids)]
    except Exception:
        pass
    return list(job_ids)


def _update_aggregate_meta(remote_dir: str, new_job_ids: list[str]) -> None:
    """Find and update the sharded aggregate meta by remote_dir match."""
    jobs_dir = _jobs_store()
    if not jobs_dir.is_dir():
        return
    for meta_path in jobs_dir.glob(f"sharded_*/{_META_FILE}"):
        try:
            agg = json.loads(meta_path.read_text())
            if agg.get("remote_dir") == remote_dir:
                agg["job_ids"] = new_job_ids
                meta_path.write_text(json.dumps(agg, indent=2))
                return
        except (json.JSONDecodeError, OSError):
            continue


def _fetch_eval_progress(
    hostname: str,
    rdir: str,
    shard_count: int | None = None,
    username: str | None = None,
) -> dict[str, tuple[int, int | None]]:
    """Count verified samples vs expected for each shard via one SSH call."""
    import re as _re

    from nemo_evaluator.executors.ssh import ssh_run

    qdir = shlex.quote(rdir)
    if shard_count:
        glob = f"{qdir}/shard_*"
    else:
        glob = qdir
    cmd = (
        f"wc -l {glob}/*/*/verified_log.jsonl 2>/dev/null;"
        f"echo '---';"
        f"grep -m1 'problems=' {glob}/logs/slurm-*.log 2>/dev/null | head -1"
    )
    output = ssh_run(hostname, cmd, username=username, timeout=15.0)
    sections = output.split("---")
    wc_out = sections[0] if sections else ""
    log_out = sections[1].strip() if len(sections) > 1 else ""

    result: dict[str, tuple[int, int | None]] = {}
    for line in wc_out.strip().splitlines():
        line = line.strip()
        if not line or "total" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            count = int(parts[0])
        except ValueError:
            continue
        path = parts[1]
        if shard_count:
            m = _re.search(r"shard_(\d+)", path)
            if m:
                result[f"shard_{m.group(1)}"] = (count, None)
        else:
            result["root"] = (count, None)

    per_shard_expected: int | None = None
    if log_out:
        m_p = _re.search(r"problems=(\d+)", log_out)
        m_r = _re.search(r"repeats=(\d+)", log_out)
        if m_p:
            per_shard_expected = int(m_p.group(1)) * (int(m_r.group(1)) if m_r else 1)

    if per_shard_expected and per_shard_expected > 0:
        for key in result:
            done, _ = result[key]
            result[key] = (done, per_shard_expected)

    return result


def _progress_label(done: int, expected: int | None) -> str:
    if expected and expected > 0:
        pct = round(100 * done / expected)
        return f"{done}/{expected} ({pct}%)"
    return f"{done} verified"


class SlurmExecutor(Executor):
    name = "slurm"

    def run(self, config, *, dry_run=False, resume=False, background=False, **_kwargs) -> None:
        import click

        from nemo_evaluator.orchestration.slurm_gen import stamp_output_dir, write_sbatch

        parent_dir = config.output.dir
        stamp_output_dir(config)
        resolved_dir = config.output.dir  # may now include timestamped subdir

        is_remote = bool(config.cluster.hostname)
        local_staging = None
        if is_remote:
            local_staging = Path(tempfile.mkdtemp(prefix="nel-slurm-"))
            script_paths, extra_paths = write_sbatch(config, output_dir=local_staging, dry_run=dry_run)
        else:
            script_paths, extra_paths = write_sbatch(config, dry_run=dry_run)

        is_sharded = len(script_paths) > 1 or (
            len(script_paths) == 1 and script_paths[0].parent.name.startswith("shard_")
        )
        for sp in script_paths:
            click.echo(f"Generated: {sp}")
        for sp in extra_paths:
            if sp.name == ".secrets.env":
                click.echo(f"  secrets: {sp}  (chmod 600)")
            else:
                click.echo(f"  sidecar: {sp}")

        if dry_run:
            if is_remote:
                target = config.cluster.hostname
                click.echo(f"\nFiles staged in: {local_staging}")
                click.echo(f"Remote target:   {target}:{resolved_dir}")
                click.echo("\nTo submit manually:")
                all_files = [str(p) for p in script_paths] + [str(p) for p in extra_paths]
                click.echo(f"  scp -r {' '.join(all_files)} {target}:{resolved_dir}/")
                for sp in script_paths:
                    shard_dir = sp.parent.name if is_sharded else ""
                    remote_script = (
                        f"{resolved_dir}/{shard_dir}/{sp.name}" if shard_dir else f"{resolved_dir}/{sp.name}"
                    )
                    click.echo(f"  ssh {target} sbatch {remote_script}")
            else:
                for sp in script_paths:
                    click.echo(f"\nTo submit:  sbatch {sp}")
            if is_sharded:
                click.echo(f"\nAfter all shards complete:  nel eval merge {resolved_dir}")
            return

        if is_remote:
            from nemo_evaluator.executors.ssh import submit_eval

            job_ids: list[str] = []
            submission_error: Exception | None = None
            try:
                for script_path in script_paths:
                    shard_extras = [p for p in extra_paths if p.parent == script_path.parent]
                    shard_remote = f"{resolved_dir}/{script_path.parent.name}" if is_sharded else resolved_dir
                    meta = submit_eval(
                        script_path=script_path,
                        hostname=config.cluster.hostname,
                        remote_dir=shard_remote,
                        username=config.cluster.username,
                        extra_files=shard_extras,
                    )
                    meta["parent_dir"] = parent_dir
                    meta["submitted_at"] = datetime.now(timezone.utc).isoformat()

                    meta_dir = _jobs_store() / meta["job_id"]
                    meta_dir.mkdir(parents=True, exist_ok=True)
                    meta_path = meta_dir / _META_FILE
                    meta_path.write_text(json.dumps(meta, indent=2))
                    job_ids.append(meta["job_id"])

                    click.echo(f"SLURM job submitted: {meta['job_id']}  ({script_path.name})")
            except Exception as exc:
                submission_error = exc
            finally:
                if local_staging:
                    shutil.rmtree(local_staging, ignore_errors=True)

            from nemo_evaluator.run_store import (
                RunMeta,
                config_summary,
                generate_run_id,
            )

            if not job_ids:
                raise click.ClickException(f"No shards submitted: {submission_error}")

            run_id = generate_run_id(config)
            run_meta = RunMeta(
                run_id=run_id,
                executor="slurm",
                output_dir=resolved_dir,
                started_at=datetime.now(timezone.utc).isoformat(),
                config_summary=config_summary(config),
                details={
                    "job_id": job_ids[0],
                    "job_ids": job_ids,
                    "hostname": config.cluster.hostname,
                    "remote_dir": resolved_dir,
                    "username": config.cluster.username or "",
                    "parent_dir": parent_dir,
                    "is_sharded": is_sharded,
                },
            )
            run_meta.save()

            if is_sharded and job_ids:
                agg_meta = {
                    "job_id": job_ids[0],
                    "job_ids": job_ids,
                    "hostname": config.cluster.hostname,
                    "remote_dir": resolved_dir,
                    "username": config.cluster.username or "",
                    "parent_dir": parent_dir,
                    "is_sharded": True,
                    "num_shards": len(job_ids),
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                }
                agg_dir = _jobs_store() / f"sharded_{job_ids[0]}"
                agg_dir.mkdir(parents=True, exist_ok=True)
                (agg_dir / _META_FILE).write_text(json.dumps(agg_meta, indent=2))

            host = config.cluster.hostname
            click.echo(f"\nrun_id: {run_id}  |  {len(job_ids)} job(s)")
            click.echo(f"Remote dir: {host}:{resolved_dir}")
            click.echo(f"Metadata:   {_jobs_store()}")
            click.echo(f"\nTail logs:  nel eval logs -r {run_id} -f")
            click.echo(f"Status:     nel eval status -r {run_id}")
            if is_sharded:
                click.echo(f"Merge:      nel eval merge {resolved_dir}")
            if submission_error:
                click.echo(
                    f"\nWARNING: Only {len(job_ids)}/{len(script_paths)} shards submitted. Error: {submission_error}",
                    err=True,
                )
            return

        import subprocess

        for script_path in script_paths:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise click.ClickException(f"sbatch failed for {script_path}: {result.stderr}")
            click.echo(result.stdout.strip())
        if is_sharded:
            click.echo(f"\nAfter all shards complete:  nel eval merge {resolved_dir}")

    def status(self, output_dir: str | Path) -> ProcessState:
        meta = _read_meta(output_dir)
        if meta is None:
            return ProcessState("slurm", False, {"error": f"No {_META_FILE} found"})

        job_ids = _get_job_ids(meta)
        hostname = meta["hostname"]
        username = meta.get("username") or None

        if not _is_sharded(meta):
            from nemo_evaluator.executors.ssh import check_job_status

            rdir = meta.get("remote_dir", str(output_dir))
            latest_id = _resolve_latest_job_id(hostname, rdir, job_ids[0] if job_ids else "", username)
            info = check_job_status(hostname=hostname, job_id=latest_id, username=username)
            running = info.get("state", "") in _RUNNING_STATES
            try:
                progress = _fetch_eval_progress(hostname, rdir, username=username)
                if "root" in progress:
                    done, expected = progress["root"]
                    info["progress"] = _progress_label(done, expected)
            except Exception as exc:
                logger.debug("progress fetch failed: %s", exc)
            return ProcessState("slurm", running, info)

        rdir = meta.get("remote_dir", str(output_dir))
        latest_ids = _resolve_shard_latest_ids(hostname, rdir, job_ids, username)

        from nemo_evaluator.executors.ssh import batch_check_job_status

        all_status = batch_check_job_status(hostname, latest_ids, username)

        details: dict[str, object] = {}
        any_running = False
        states: list[str] = []
        for i, (orig_id, latest_id) in enumerate(zip(job_ids, latest_ids)):
            info = all_status.get(latest_id, {"job_id": latest_id, "state": "UNKNOWN"})
            state = info.get("state", "UNKNOWN")
            states.append(state)
            elapsed = info.get("time", "")
            attempt = f", chain {latest_id}" if latest_id != orig_id else ""
            label = f"{state} (job {latest_id}{attempt})"
            if elapsed and state in _RUNNING_STATES:
                label += f" {elapsed} elapsed"
            details[f"shard_{i}"] = label
            if state in _RUNNING_STATES:
                any_running = True

        counts = Counter(states)
        completed = counts.get("COMPLETED", 0)
        summary_parts = [f"{cnt} {st}" for st, cnt in sorted(counts.items())]
        shards_summary = f"{completed}/{len(job_ids)} COMPLETED, {', '.join(summary_parts)}"

        try:
            progress = _fetch_eval_progress(hostname, rdir, shard_count=len(job_ids), username=username)
            if progress:
                for shard_key, (done, expected) in progress.items():
                    if shard_key in details:
                        details[shard_key] = f"{details[shard_key]}, progress: {_progress_label(done, expected)}"
                total_done = sum(d for d, _ in progress.values())
                exp_vals = [e for _, e in progress.values() if e is not None]
                total_exp = sum(exp_vals) if exp_vals else None
                shards_summary += f", progress: {_progress_label(total_done, total_exp)}"
        except Exception as exc:
            logger.debug("progress fetch failed: %s", exc)

        details = {"shards": shards_summary, **details}

        try:
            from nemo_evaluator.executors.ssh import ssh_run

            qdir = shlex.quote(rdir)
            merge_out = ssh_run(
                hostname,
                f"[[ -f {qdir}/.merge_lock/.done ]] && echo DONE "
                f"|| ([[ -d {qdir}/.merge_lock ]] && echo IN_PROGRESS || echo PENDING)",
                username=username,
                timeout=10.0,
            ).strip()
            details["merge"] = merge_out
        except Exception:
            pass

        return ProcessState("slurm", any_running, details)

    def stop(self, output_dir: str | Path, *, shard_idx: int | None = None) -> bool:
        meta = _read_meta(output_dir)
        if meta is None:
            logger.warning("No %s found in %s", _META_FILE, output_dir)
            return False

        job_ids = _get_job_ids(meta)
        hostname = meta["hostname"]
        username = meta.get("username") or None

        if not _is_sharded(meta):
            from nemo_evaluator.executors.ssh import cancel_job

            try:
                cancel_job(hostname=hostname, job_id=job_ids[0], username=username)
                return True
            except Exception as e:
                logger.error("Failed to cancel SLURM job %s: %s", job_ids[0], e)
                return False

        rdir = meta.get("remote_dir", str(output_dir))

        if shard_idx is not None:
            if shard_idx < 0 or shard_idx >= len(job_ids):
                logger.error("Shard index %d out of range (0-%d)", shard_idx, len(job_ids) - 1)
                return False
            latest = _resolve_latest_job_id(hostname, f"{rdir}/shard_{shard_idx}", job_ids[shard_idx], username)
            from nemo_evaluator.executors.ssh import cancel_job

            try:
                cancel_job(hostname=hostname, job_id=latest, username=username)
                return True
            except Exception as e:
                logger.error("Failed to cancel shard %d (job %s): %s", shard_idx, latest, e)
                return False

        latest_ids = _resolve_shard_latest_ids(hostname, rdir, job_ids, username)
        from nemo_evaluator.executors.ssh import ssh_run

        ids_str = " ".join(latest_ids)
        try:
            ssh_run(hostname, f"scancel {ids_str}", username=username, timeout=15.0)
            return True
        except Exception as e:
            logger.error("Failed to cancel sharded jobs: %s", e)
            return False

    def logs(
        self,
        output_dir: str | Path,
        *,
        follow: bool = False,
        tail: int | None = None,
        shard_idx: int | None = None,
    ) -> str | None:
        meta = _read_meta(output_dir)
        if meta is None:
            return None
        from nemo_evaluator.executors.ssh import ssh_run

        hostname = meta["hostname"]
        username = meta.get("username") or None
        rdir = meta.get("remote_dir", str(output_dir))
        job_ids = _get_job_ids(meta)

        if not _is_sharded(meta):
            latest_id = _resolve_latest_job_id(hostname, rdir, job_ids[0] if job_ids else "", username)
            log_file = f"{rdir}/logs/slurm-{latest_id}.log"
            if tail:
                return ssh_run(hostname, f"tail -n {tail} {log_file}", username=username, timeout=30.0)
            return ssh_run(hostname, f"cat {log_file}", username=username, timeout=60.0)

        if shard_idx is not None:
            if shard_idx < 0 or shard_idx >= len(job_ids):
                return f"Shard index {shard_idx} out of range (0-{len(job_ids) - 1})."
            shard_rdir = f"{rdir}/shard_{shard_idx}"
            latest_id = _resolve_latest_job_id(hostname, shard_rdir, job_ids[shard_idx], username)
            log_file = f"{shard_rdir}/logs/slurm-{latest_id}.log"
            if tail:
                return ssh_run(hostname, f"tail -n {tail} {log_file}", username=username, timeout=30.0)
            return ssh_run(hostname, f"cat {log_file}", username=username, timeout=60.0)

        tail_n = tail or 20
        cmds = []
        for i, fb in enumerate(job_ids):
            sdir = shlex.quote(f"{rdir}/shard_{i}")
            cmds.append(
                f'echo "=== shard_{i} (job {fb}) ==="; '
                f'_l=$(tail -1 {sdir}/.nel_job_chain 2>/dev/null); _l="${{_l:-{fb}}}"; '
                f'tail -n {tail_n} {sdir}/logs/slurm-$_l.log 2>/dev/null || echo "(no log yet)"'
            )
        script = '; echo ""; '.join(cmds)
        try:
            return ssh_run(hostname, script, username=username, timeout=60.0)
        except Exception as e:
            return f"Failed to fetch logs: {e}"

    def resume_run(self, run_meta, **kwargs) -> None:
        import re
        import shlex as shlex_mod

        import click

        details = run_meta.details
        hostname = details.get("hostname", "")
        username = details.get("username") or None
        rdir = details.get("remote_dir", run_meta.output_dir)
        job_ids = _get_job_ids(details)
        continue_attempts = kwargs.get("continue_attempts", False)
        shard_idx = kwargs.get("shard_idx")

        from nemo_evaluator.executors.ssh import ssh_run

        if not _is_sharded(details):
            script = f"{rdir}/nel_eval.sbatch"
            if not continue_attempts:
                ssh_run(
                    hostname,
                    f"rm -f {shlex_mod.quote(rdir)}/.nel_infra_retries {shlex_mod.quote(rdir)}/.nel_accumulated_walltime",
                    username=username,
                    timeout=10.0,
                )
            output = ssh_run(hostname, f"sbatch {shlex_mod.quote(script)}", username=username, timeout=30.0)
            click.echo(f"Resubmitted: {output.strip()}")
            m = re.search(r"(\d+)", output)
            if m:
                new_id = m.group(1)
                run_meta.update_details(job_id=new_id, job_ids=[new_id])
                new_meta_dir = _jobs_store() / new_id
                new_meta_dir.mkdir(parents=True, exist_ok=True)
                (new_meta_dir / _META_FILE).write_text(json.dumps({**details, "job_id": new_id}, indent=2))
                click.echo(f"New job ID: {new_id}")
                click.echo(f"Tail logs:  nel eval logs -r {run_meta.run_id} -f")
                click.echo(f"Status:     nel eval status -r {run_meta.run_id}")
            return

        from nemo_evaluator.executors.ssh import batch_check_job_status

        latest_ids = _resolve_shard_latest_ids(hostname, rdir, job_ids, username)

        if shard_idx is not None:
            if shard_idx < 0 or shard_idx >= len(job_ids):
                raise click.ClickException(f"Shard index {shard_idx} out of range (0-{len(job_ids) - 1}).")
            shards_to_resume = [shard_idx]
        else:
            all_status = batch_check_job_status(hostname, latest_ids, username)
            shards_to_resume = []
            for i, latest_id in enumerate(latest_ids):
                info = all_status.get(latest_id, {})
                state = info.get("state", "UNKNOWN")
                if state in _RESUMABLE_STATES or state == "UNKNOWN":
                    shards_to_resume.append(i)
                elif state == "COMPLETED":
                    click.echo(f"  shard_{i}: COMPLETED — skipping")
                elif state in _RUNNING_STATES:
                    click.echo(f"  shard_{i}: {state} — still active, skipping")
                elif state.startswith("CANCELLED"):
                    click.echo(f"  shard_{i}: CANCELLED — skipping (use --shard {i} to force)")
                else:
                    shards_to_resume.append(i)

        if not shards_to_resume:
            click.echo("No shards need resuming.")
            return

        new_job_ids = list(job_ids)
        failed_shards: list[int] = []
        for i in shards_to_resume:
            shard_dir = f"{rdir}/shard_{i}"
            script = f"{shard_dir}/nel_eval.sbatch"
            try:
                if not continue_attempts:
                    ssh_run(
                        hostname,
                        f"rm -f {shlex_mod.quote(shard_dir)}/.nel_infra_retries "
                        f"{shlex_mod.quote(shard_dir)}/.nel_accumulated_walltime",
                        username=username,
                        timeout=10.0,
                    )
                output = ssh_run(hostname, f"sbatch {shlex_mod.quote(script)}", username=username, timeout=30.0)
                click.echo(f"  shard_{i}: {output.strip()}")
                m = re.search(r"(\d+)", output)
                if m:
                    new_job_ids[i] = m.group(1)
            except Exception as e:
                click.echo(f"  shard_{i}: FAILED — {e}", err=True)
                failed_shards.append(i)

        if new_job_ids != list(job_ids):
            run_meta.update_details(job_ids=new_job_ids)
            _update_aggregate_meta(rdir, new_job_ids)

        submitted = len(shards_to_resume) - len(failed_shards)
        msg = f"\nResubmitted {submitted}/{len(job_ids)} shards."
        if failed_shards:
            msg += f"  ({len(failed_shards)} failed: {failed_shards})"
        click.echo(msg)
        click.echo(f"Status:     nel eval status -r {run_meta.run_id}")
        click.echo(f"Tail logs:  nel eval logs -r {run_meta.run_id} -f")

    @staticmethod
    def detect(output_dir: str | Path) -> bool:
        return _read_meta(output_dir) is not None


def _resolve_latest_job_id_from_meta(meta: dict) -> str:
    """Follow .nel_job_chain on the remote host to get the latest job ID.

    Backward-compatible wrapper — delegates to _resolve_latest_job_id.
    """
    return _resolve_latest_job_id(
        hostname=meta.get("hostname", ""),
        remote_dir=meta.get("remote_dir", ""),
        fallback_id=meta.get("job_id", ""),
        username=meta.get("username") or None,
    )


def resolve_job(
    job_id: str | None = None,
    host: str | None = None,
    output_dir: str | Path | None = None,
) -> dict | None:
    """Resolve job metadata from any combination of identifiers.

    Resolution order:
    1. Bare numeric job_id → look up in local jobs store
    2. output_dir → look for slurm_job.json inside it
    3. output_dir → search jobs store by remote_dir match
    4. output_dir → scan shard_*/slurm_job.json and aggregate (recovery)
    """
    jobs_dir = _jobs_store()

    if job_id:
        p = jobs_dir / str(job_id) / _META_FILE
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        if host:
            return {"job_id": job_id, "hostname": host, "remote_dir": ""}

    if output_dir:
        p = Path(output_dir) / _META_FILE
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        if jobs_dir.is_dir():
            out_str = str(output_dir)
            for meta_path in jobs_dir.glob(f"*/{_META_FILE}"):
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("remote_dir") == out_str:
                        return meta
                except (json.JSONDecodeError, OSError):
                    continue

        # Fallback: scan shard_*/slurm_job.json for recovery when parent meta
        # is missing (e.g. interrupted submission, manual deletion).
        out_path = Path(output_dir)
        shard_metas = []
        for shard_dir in sorted(out_path.glob("shard_*")):
            sp = shard_dir / _META_FILE
            if sp.exists():
                try:
                    shard_metas.append(json.loads(sp.read_text()))
                except (json.JSONDecodeError, OSError):
                    continue
        if shard_metas:
            return {
                "job_id": shard_metas[0]["job_id"],
                "job_ids": [m["job_id"] for m in shard_metas],
                "hostname": shard_metas[0].get("hostname", ""),
                "remote_dir": str(output_dir),
                "username": shard_metas[0].get("username", ""),
                "is_sharded": True,
                "num_shards": len(shard_metas),
            }

    return None


def _read_meta(output_dir: str | Path) -> dict | None:
    """Back-compat wrapper around resolve_job()."""
    return resolve_job(output_dir=output_dir)
