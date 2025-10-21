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
"""Export evaluation artifacts to local filesystem."""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from nemo_evaluator_launcher.common.execdb import JobData
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.exporters.base import BaseExporter, ExportResult
from nemo_evaluator_launcher.exporters.registry import register_exporter
from nemo_evaluator_launcher.exporters.utils import (
    download_gitlab_artifacts,
    extract_accuracy_metrics,
    extract_exporter_config,
    get_benchmark_info,
    get_container_from_mapping,
    get_model_name,
    get_relevant_artifacts,
    get_task_name,
    ssh_cleanup_masters,
    ssh_download_artifacts,
    ssh_setup_masters,
    validate_artifacts,
)


@register_exporter("local")
class LocalExporter(BaseExporter):
    """Export all artifacts to local/remote filesystem with optional JSON/CSV summaries.

    Config keys:
      output_dir (str): Output directory for exported results (default: "./nemo-evaluator-launcher-results")
      copy_logs (bool): Whether to copy logs with artifacts (default: False)
      only_required (bool): Copy only required+optional artifacts (default: True)
      format (str or None): Summary format, one of None, "json", or "csv" (default: None; no summary, only original artifacts)
      log_metrics (list[str]): Filters for metric names; includes full metric name or substring pattern
      output_filename (str): Overrides default processed_results.json/csv filename
    """

    def supports_executor(self, executor_type: str) -> bool:
        return True  # Local export compatible with all executors

    def export_job(self, job_data: JobData) -> ExportResult:
        """Export job artifacts to local directory."""
        # Merge auto-export + CLI config
        cfg = extract_exporter_config(job_data, "local", self.config)
        skip_validation = bool(cfg.get("skip_validation", False))

        output_dir = Path(cfg.get("output_dir", "./nemo-evaluator-launcher-results"))
        job_export_dir = output_dir / job_data.invocation_id / job_data.job_id
        job_export_dir.mkdir(parents=True, exist_ok=True)

        try:
            paths = self.get_job_paths(job_data)
            exported_files: List[str] = []

            # Stage artifacts per storage type
            if paths["storage_type"] == "local_filesystem":
                exported_files = self._copy_local_artifacts(paths, job_export_dir, cfg)
            elif paths["storage_type"] == "remote_local":
                # Same as local_filesystem (we're on the remote machine, accessing locally)
                exported_files = self._copy_local_artifacts(paths, job_export_dir, cfg)
            elif paths["storage_type"] == "remote_ssh":
                cp = ssh_setup_masters({job_data.job_id: job_data})
                try:
                    exported_files = ssh_download_artifacts(
                        paths, job_export_dir, cfg, cp
                    )
                finally:
                    ssh_cleanup_masters(cp)
            else:
                raise NotImplementedError(
                    f"Export not implemented for storage type: {paths['storage_type']}"
                )

            # Validate artifacts
            artifacts_dir = job_export_dir / "artifacts"
            validation = (
                validate_artifacts(artifacts_dir)
                if not skip_validation
                else {
                    "can_export": True,
                    "missing_required": [],
                    "missing_optional": [],
                    "message": "Validation skipped",
                }
            )

            # Save metadata
            self._save_job_metadata(job_data, job_export_dir)
            exported_files.append(str(job_export_dir / "job_metadata.json"))

            if not validation["can_export"]:
                return ExportResult(
                    success=False,
                    dest=str(job_export_dir),
                    message=validation["message"],
                    metadata=validation,
                )

            if validation["missing_optional"]:
                logger.info(
                    f"Exporting without optional artifacts: {', '.join(validation['missing_optional'])}",
                    job_id=job_data.job_id,
                )

            # Optional summary (JSON/CSV) at invocation level
            msg = f"Exported {len(exported_files)} files. {validation['message']}"
            meta: Dict[str, Any] = {"files_count": len(exported_files)}
            fmt = cfg.get("format")
            if fmt in ["json", "csv"]:
                try:
                    summary_path = self._write_summary(job_data, cfg)
                    meta["summary_path"] = str(summary_path)
                    msg += f". Summary: {summary_path.name}"
                except Exception as e:
                    logger.warning(f"Failed to create {fmt} summary: {e}")
                    msg += " (summary failed)"

            meta["output_dir"] = str(job_export_dir.resolve())

            return ExportResult(
                success=True, dest=str(job_export_dir), message=msg, metadata=meta
            )

        except Exception as e:
            logger.error(f"Failed to export job {job_data.job_id}: {e}")
            return ExportResult(
                success=False,
                dest=str(job_export_dir),
                message=f"Export failed: {str(e)}",
                metadata={},
            )

    def export_invocation(self, invocation_id: str) -> Dict[str, Any]:
        """Export all jobs in an invocation (with connection reuse)."""
        jobs = self.db.get_jobs(invocation_id)
        if not jobs:
            return {
                "success": False,
                "error": f"No jobs found for invocation {invocation_id}",
            }

        control_paths = ssh_setup_masters(jobs)
        try:
            results = {}
            for job_id, job_data in jobs.items():
                result = self.export_job(job_data)
                results[job_id] = result.__dict__
            return {"success": True, "invocation_id": invocation_id, "jobs": results}
        finally:
            ssh_cleanup_masters(control_paths)

    def export_multiple_invocations(self, invocation_ids: List[str]) -> Dict[str, Any]:
        db_jobs: Dict[str, JobData] = {}
        results: Dict[str, Any] = {}
        for inv in invocation_ids:
            jobs = self.db.get_jobs(inv)
            if jobs:
                db_jobs.update(jobs)
                results[inv] = {"success": True, "job_count": len(jobs)}
            else:
                results[inv] = {"success": False, "error": f"No jobs found for {inv}"}
        if not db_jobs:
            return {
                "success": False,
                "error": "No jobs to export",
                "invocations": results,
            }

        # Reuse SSH masters across all jobs/hosts and stage artifacts locally
        cp = ssh_setup_masters(db_jobs)
        try:
            first = next(iter(db_jobs.values()))
            cfg = extract_exporter_config(first, "local", self.config)
            fmt = cfg.get("format")
            output_dir = Path(
                cfg.get("output_dir", "./nemo-evaluator-launcher-results")
            )
            filename = cfg.get("output_filename", f"processed_results.{fmt}")
            out_path = output_dir / filename  # consolidated file at output_dir

            # Stage artifacts for all jobs into <output_dir>/<inv>/<job>/
            for jd in db_jobs.values():
                try:
                    self.export_job(jd)
                except Exception:
                    pass  # keep going; remaining jobs may still contribute

            # Build consolidated summary from staged artifacts
            all_metrics, jobs_list = {}, []
            for jd in db_jobs.values():
                artifacts_dir = output_dir / jd.invocation_id / jd.job_id / "artifacts"
                metrics = extract_accuracy_metrics(
                    jd,
                    lambda _: {
                        "artifacts_dir": artifacts_dir,
                        "storage_type": "local_filesystem",
                    },
                    cfg.get("log_metrics", []),
                )
                all_metrics[jd.job_id] = metrics
                jobs_list.append(jd)

            if fmt == "json":
                if out_path.exists():
                    data = json.loads(out_path.read_text(encoding="utf-8"))
                else:
                    data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "benchmarks": {},
                    }
                for jd in jobs_list:
                    bench, model, entry = self._build_entry(
                        jd, all_metrics.get(jd.job_id, {}), cfg
                    )
                    m = (
                        data.setdefault("benchmarks", {})
                        .setdefault(bench, {})
                        .setdefault("models", {})
                    )
                    lst = m.setdefault(model, [])
                    idx = next(
                        (
                            i
                            for i, e in enumerate(lst)
                            if e.get("invocation_id") == jd.invocation_id
                            and e.get("job_id") == jd.job_id
                        ),
                        None,
                    )
                    if idx is None:
                        lst.append(entry)
                    else:
                        lst[idx] = entry
                data["export_timestamp"] = datetime.now().isoformat()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            elif fmt == "csv":
                for jd in jobs_list:
                    self._csv_upsert(out_path, jd, all_metrics.get(jd.job_id, {}), cfg)

            return {
                "success": True,
                "invocations": results,
                "metadata": {
                    "total_invocations": len(invocation_ids),
                    "total_jobs": len(db_jobs),
                    "summary_path": str(out_path.resolve()),
                },
            }
        finally:
            ssh_cleanup_masters(cp)

    # Artifact staging helpers
    def _copy_local_artifacts(
        self, paths: Dict[str, Any], export_dir: Path, cfg: Dict[str, Any]
    ) -> List[str]:
        exported_files: List[str] = []
        copy_logs = bool(cfg.get("copy_logs", False))
        copy_artifacts = bool(cfg.get("copy_artifacts", True))
        only_required = bool(cfg.get("only_required", True))

        # separate logic for artifacts and logs
        # artifacts/
        if copy_artifacts and paths["artifacts_dir"].exists():
            if only_required:
                names = [
                    a
                    for a in get_relevant_artifacts()
                    if (paths["artifacts_dir"] / a).exists()
                ]
                (export_dir / "artifacts").mkdir(parents=True, exist_ok=True)
                for name in names:
                    src = paths["artifacts_dir"] / name
                    dst = export_dir / "artifacts" / name
                    shutil.copy2(src, dst)
                    exported_files.append(str(dst))
            else:
                # Restore recursive copy (test_copy_all_tree expects nested files)
                shutil.copytree(
                    paths["artifacts_dir"], export_dir / "artifacts", dirs_exist_ok=True
                )
                exported_files.extend(
                    [
                        str(f)
                        for f in (export_dir / "artifacts").rglob("*")
                        if f.is_file()
                    ]
                )

        # logs/
        # If only_required is False → always copy logs; otherwise respect copy_logs
        if ((not only_required) or copy_logs) and paths["logs_dir"].exists():
            shutil.copytree(paths["logs_dir"], export_dir / "logs", dirs_exist_ok=True)
            exported_files.extend(
                [str(f) for f in (export_dir / "logs").rglob("*") if f.is_file()]
            )

        return exported_files

    def _download_gitlab_remote_artifacts(
        self, paths: Dict[str, Any], export_dir: Path
    ) -> List[str]:
        artifacts = download_gitlab_artifacts(paths, export_dir, extract_specific=True)
        return [str(p) for p in artifacts.values()]

    def _save_job_metadata(self, job_data: JobData, export_dir: Path):
        metadata = {
            "invocation_id": job_data.invocation_id,
            "job_id": job_data.job_id,
            "executor": job_data.executor,
            "timestamp": job_data.timestamp,
        }
        (export_dir / "job_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

    # Summary JSON/CSV helpers
    def _write_summary(self, job_data: JobData, cfg: Dict[str, Any]) -> Path:
        """Read per-job artifacts, extract metrics, and update invocation-level summary."""
        output_dir = Path(cfg.get("output_dir", "./nemo-evaluator-launcher-results"))
        artifacts_dir = (
            output_dir / job_data.invocation_id / job_data.job_id / "artifacts"
        )
        fmt = cfg.get("format")
        filename = cfg.get("output_filename", f"processed_results.{fmt}")
        out_path = output_dir / job_data.invocation_id / filename

        # Extract metrics
        metrics = extract_accuracy_metrics(
            job_data,
            lambda jd: {
                "artifacts_dir": artifacts_dir,
                "storage_type": "local_filesystem",
            },
            cfg.get("log_metrics", []),
        )

        if fmt == "json":
            self._json_upsert(out_path, job_data, metrics, cfg)
        elif fmt == "csv":
            self._csv_upsert(out_path, job_data, metrics, cfg)
        return out_path.resolve()

    def _build_entry(
        self, job_data: JobData, metrics: Dict[str, float], cfg: Dict[str, Any]
    ) -> tuple[str, str, dict]:
        bench = get_benchmark_info(job_data)
        benchmark_name = bench["benchmark"]
        model_name = get_model_name(job_data, cfg)
        entry = {
            "invocation_id": job_data.invocation_id,
            "job_id": job_data.job_id,
            "harness": bench.get("harness", "unknown"),
            "container": get_container_from_mapping(job_data),
            "scores": metrics,
            "timestamp": datetime.now().isoformat(),
            "executor": job_data.executor,
        }
        return benchmark_name, model_name, entry

    def _json_upsert(
        self,
        out_path: Path,
        job_data: JobData,
        metrics: Dict[str, float],
        cfg: Dict[str, Any],
    ) -> None:
        if out_path.exists():
            data = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            data = {"export_timestamp": datetime.now().isoformat(), "benchmarks": {}}

        benchmark_name, model_name, entry = self._build_entry(job_data, metrics, cfg)
        bench = data.setdefault("benchmarks", {}).setdefault(benchmark_name, {})
        models = bench.setdefault("models", {})

        # Switch to list semantics
        lst = models.setdefault(model_name, [])
        # Upsert by unique combination
        idx = next(
            (
                i
                for i, e in enumerate(lst)
                if e.get("invocation_id") == job_data.invocation_id
                and e.get("job_id") == job_data.job_id
            ),
            None,
        )
        if idx is None:
            lst.append(entry)  # append
        else:
            lst[idx] = entry  # override

        data["export_timestamp"] = datetime.now().isoformat()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _csv_upsert(
        self,
        out_path: Path,
        job_data: JobData,
        metrics: Dict[str, float],
        cfg: Dict[str, Any],
    ) -> None:
        base_cols = [
            "Model Name",
            "Harness",
            "Task Name",
            "Executor",
            "Container",
            "Invocation ID",
            "Job ID",
        ]
        rows, headers = [], []
        if out_path.exists():
            with out_path.open("r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                headers = next(r, [])
                rows = list(r)
        else:
            headers = base_cols.copy()

        # Build metric names by stripping <benchmark>_ prefix from keys
        benchmark, model_name, entry = self._build_entry(job_data, metrics, cfg)

        # clean headers using bare benchmark
        task_prefix = benchmark  # no harness prefix
        clean_metrics = []
        for full_key in metrics.keys():
            if full_key.startswith(f"{task_prefix}_"):
                clean_metrics.append(full_key[len(task_prefix) + 1 :])
            else:
                clean_metrics.append(full_key)

        # Extend headers if new metrics appear
        metric_cols_existing = [h for h in headers if h not in base_cols]
        new_metric_cols = [
            m for m in sorted(set(clean_metrics)) if m not in metric_cols_existing
        ]
        if new_metric_cols:
            headers = headers + new_metric_cols
            for row in rows:
                row.extend([""] * len(new_metric_cols))

        # Build row for this job (upsert keyed by invocation_id + job_id)
        bench = get_benchmark_info(job_data)
        task_name = get_task_name(job_data)
        row = [
            model_name,
            bench.get("harness", "unknown"),
            task_name,
            job_data.executor,
            get_container_from_mapping(job_data),
            job_data.invocation_id,
            job_data.job_id,
        ]
        # Fill metric columns from <benchmark>_<...>
        for col in headers[len(base_cols) :]:
            full_key = f"{task_prefix}_{col}"
            val = metrics.get(full_key, "")
            try:
                row.append("" if val == "" else float(val))
            except Exception:
                row.append(val)

        # Upsert row
        idx_by_key = {(r[5], r[6]): i for i, r in enumerate(rows) if len(r) >= 7}
        key = (job_data.invocation_id, job_data.job_id)
        if key in idx_by_key:
            rows[idx_by_key[key]] = row
        else:
            rows.append(row)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
