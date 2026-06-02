# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SSH-based remote SLURM submission helpers.

The functions under test shell out to ``ssh`` / ``scp`` so we patch
``_ssh`` and ``_scp`` and assert on the calls they would have made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nemo_evaluator.executors import ssh as ssh_mod


def _record_calls():
    ssh_calls: list[tuple] = []
    scp_calls: list[tuple] = []

    def fake_ssh(target, cmd, *args, **kwargs):
        ssh_calls.append((target, cmd))
        return ""

    def fake_scp(local_path, remote_dest, target, *args, **kwargs):
        scp_calls.append((local_path, remote_dest, target))
        return ""

    return ssh_calls, scp_calls, fake_ssh, fake_scp


def test_copy_to_remote_local_base_preserves_subdir_structure(tmp_path: Path):
    """When local_base is set, files in subdirs are copied to matching
    remote subdirs (not flattened).  Critical for legacy container://
    benchmarks that drop a ``run_config.yaml`` under ``{staging}/{safe_name}/``.
    """
    sbatch = tmp_path / "nel_eval.sbatch"
    sbatch.write_text("#!/bin/bash\n")
    secrets = tmp_path / ".secrets.env"
    secrets.write_text("FOO=bar\n")
    sub = tmp_path / "container_xyz"
    sub.mkdir()
    sidecar = sub / "run_config.yaml"
    sidecar.write_text("config: {}\n")

    ssh_calls, scp_calls, fake_ssh, fake_scp = _record_calls()
    with patch.object(ssh_mod, "_ssh", side_effect=fake_ssh), patch.object(ssh_mod, "_scp", side_effect=fake_scp):
        ssh_mod.copy_to_remote(
            hostname="loginhost",
            local_paths=[sbatch, secrets, sidecar],
            remote_dir="/srv/run/abc",
            local_base=tmp_path,
        )

    # subdir mkdir was issued before scp.
    mkdir_cmds = [c[1] for c in ssh_calls if "mkdir -p" in c[1]]
    assert any("/srv/run/abc/container_xyz" in c for c in mkdir_cmds), mkdir_cmds

    scp_dests = [c[1] for c in scp_calls]
    assert "loginhost:/srv/run/abc/nel_eval.sbatch" in scp_dests
    assert "loginhost:/srv/run/abc/.secrets.env" in scp_dests
    assert "loginhost:/srv/run/abc/container_xyz/run_config.yaml" in scp_dests


def test_copy_to_remote_without_local_base_flattens(tmp_path: Path):
    """Default behavior (no local_base): everything lands directly under
    remote_dir.  Preserves backwards compat for callers that pass siblings
    only and don't care about subdir structure."""
    a = tmp_path / "a.txt"
    a.write_text("a")
    b = tmp_path / "b.txt"
    b.write_text("b")

    _, scp_calls, fake_ssh, fake_scp = _record_calls()
    with patch.object(ssh_mod, "_ssh", side_effect=fake_ssh), patch.object(ssh_mod, "_scp", side_effect=fake_scp):
        ssh_mod.copy_to_remote(
            hostname="loginhost",
            local_paths=[a, b],
            remote_dir="/srv/run/abc",
        )

    scp_dests = [c[1] for c in scp_calls]
    # No filename appended — old behavior puts files into the dir.
    assert all(d == "loginhost:/srv/run/abc/" for d in scp_dests), scp_dests


def test_submit_eval_propagates_local_base(tmp_path: Path):
    """submit_eval forwards ``local_base`` to copy_to_remote so nested
    sidecars don't get flattened on the way up."""
    sbatch = tmp_path / "nel_eval.sbatch"
    sbatch.write_text("#!/bin/bash\n")
    sub = tmp_path / "bench_x"
    sub.mkdir()
    sidecar = sub / "run_config.yaml"
    sidecar.write_text("k: v\n")

    captured: dict = {}

    def fake_copy(hostname, local_paths, remote_dir, username=None, local_base=None):
        captured["hostname"] = hostname
        captured["local_paths"] = list(local_paths)
        captured["remote_dir"] = remote_dir
        captured["local_base"] = local_base

    def fake_submit(hostname, remote_script, username=None):
        return "12345"

    with (
        patch.object(ssh_mod, "copy_to_remote", side_effect=fake_copy),
        patch.object(ssh_mod, "submit_sbatch", side_effect=fake_submit),
    ):
        meta = ssh_mod.submit_eval(
            script_path=sbatch,
            hostname="loginhost",
            remote_dir="/srv/run/abc",
            extra_files=[sidecar],
            local_base=tmp_path,
        )

    assert captured["local_base"] == tmp_path
    assert sbatch in captured["local_paths"]
    assert sidecar in captured["local_paths"]
    assert meta["job_id"] == "12345"
