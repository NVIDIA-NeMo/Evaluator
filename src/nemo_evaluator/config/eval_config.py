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
"""Top-level EvalConfig and YAML parser with env-var expansion."""

from __future__ import annotations

import os
import re
import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .benchmarks import BenchmarkConfig
from .clusters import ClusterConfig, LocalCluster, SlurmCluster, _parse_walltime
from .output import OutputConfig
from .sandboxes import SandboxConfig
from .scoring import (
    EnsembleJudgeMetric,
    JudgeMetric,
    PairwiseJudgeMetric,
    RewardModelMetric,
)
from .services import (
    _MODEL_SERVICE_TYPES,
    GymResourceService,
    NatAgentService,
    ServiceConfig,
)
from .solvers import (
    AgentSolverConfig,
    ContainerSolverConfig,
    GymDelegationSolverConfig,
    HarborSolverConfig,
    NatSolverConfig,
    OpenClawSolverConfig,
    SimpleSolver,
    ToolCallingSolverConfig,
)


def _get_solver_service(solver: Any) -> str | None:
    """Extract the primary model service name from a solver config."""
    return getattr(solver, "service", None)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: dict[str, ServiceConfig]
    sandboxes: dict[str, SandboxConfig] = Field(default_factory=dict)
    benchmarks: list[BenchmarkConfig] = Field(min_length=1)
    cluster: ClusterConfig = Field(default_factory=LocalCluster)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("services")
    @classmethod
    def _no_reserved_service_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        reserved = {"default", "none", ""}
        bad = reserved & set(v.keys())
        if bad:
            raise ValueError(
                f"Reserved service names are not allowed: {sorted(bad)}. "
                f"Use semantic names (e.g. 'solver', 'judge', 'nemotron')."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _resolve_sandbox_references(cls, data: Any) -> Any:
        """Resolve string sandbox references before Pydantic validates unions."""
        if isinstance(data, dict):
            named_sandboxes = data.get("sandboxes", {})
            for bench in data.get("benchmarks", []):
                if isinstance(bench, dict) and isinstance(bench.get("sandbox"), str):
                    name = bench["sandbox"]
                    if name not in named_sandboxes:
                        raise ValueError(
                            f"benchmark sandbox reference '{name}' not found "
                            f"in sandboxes. Available: {sorted(named_sandboxes.keys())}"
                        )
                    bench["sandbox"] = named_sandboxes[name]
        return data

    @model_validator(mode="after")
    def _validate_service_references(self) -> EvalConfig:
        """Comprehensive cross-reference validation."""
        available = set(self.services.keys())
        errors: list[str] = []

        for i, bench in enumerate(self.benchmarks):
            prefix = f"benchmarks[{i}]"
            solver = bench.solver

            solver_svc = _get_solver_service(solver)
            if solver_svc is not None and solver_svc not in available:
                errors.append(f"{prefix}.solver.service={solver_svc!r} not in services {sorted(available)}")

            if solver_svc is not None and solver_svc in available:
                svc = self.services[solver_svc]
                if isinstance(
                    solver,
                    (
                        SimpleSolver,
                        HarborSolverConfig,
                        AgentSolverConfig,
                        OpenClawSolverConfig,
                        ContainerSolverConfig,
                        ToolCallingSolverConfig,
                        GymDelegationSolverConfig,
                    ),
                ):
                    if not isinstance(svc, _MODEL_SERVICE_TYPES):
                        errors.append(
                            f"{prefix}.solver.service={solver_svc!r} is a "
                            f"'{svc.type}' service, but solver type "
                            f"'{solver.type}' requires a model service"
                        )
                elif isinstance(solver, NatSolverConfig):
                    if not isinstance(svc, NatAgentService):
                        errors.append(
                            f"{prefix}.solver.service={solver_svc!r} is a "
                            f"'{svc.type}' service, but solver type 'nat' "
                            f"requires a 'nat' service"
                        )

            if isinstance(solver, ToolCallingSolverConfig):
                if solver.resource_service is not None:
                    if solver.resource_service not in available:
                        errors.append(f"{prefix}.solver.resource_service={solver.resource_service!r} not in services")
                    elif not isinstance(self.services[solver.resource_service], GymResourceService):
                        errors.append(
                            f"{prefix}.solver.resource_service={solver.resource_service!r} must be a 'gym' service"
                        )
            elif isinstance(solver, GymDelegationSolverConfig):
                if solver.gym_service not in available:
                    errors.append(f"{prefix}.solver.gym_service={solver.gym_service!r} not in services")
                elif not isinstance(self.services[solver.gym_service], GymResourceService):
                    errors.append(f"{prefix}.solver.gym_service={solver.gym_service!r} must be a 'gym' service")

            if bench.verifier and bench.verifier not in available:
                errors.append(f"{prefix}.verifier={bench.verifier!r} not in services")

            for j, metric in enumerate(bench.scoring.metrics):
                mprefix = f"{prefix}.scoring.metrics[{j}]"

                if isinstance(metric, JudgeMetric):
                    if metric.service not in available:
                        errors.append(f"{mprefix}.service={metric.service!r} not in services")
                    elif solver_svc is not None and metric.service == solver_svc and not metric.allow_self_judge:
                        errors.append(
                            f"{mprefix}.service={metric.service!r} is the "
                            f"same as the solver's service. Use a separate "
                            f"service, or set allow_self_judge: true."
                        )
                elif isinstance(metric, PairwiseJudgeMetric):
                    for field_name in ("judge_service", "baseline_service"):
                        svc_name = getattr(metric, field_name)
                        if svc_name not in available:
                            errors.append(f"{mprefix}.{field_name}={svc_name!r} not in services")
                elif isinstance(metric, EnsembleJudgeMetric):
                    for k, svc_name in enumerate(metric.services):
                        if svc_name not in available:
                            errors.append(f"{mprefix}.services[{k}]={svc_name!r} not in services")
                elif isinstance(metric, RewardModelMetric):
                    if metric.service not in available:
                        errors.append(f"{mprefix}.service={metric.service!r} not in services")

        if errors:
            raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_node_pools(self) -> EvalConfig:
        """Validate node_pool references and resource fit."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        pools = set(self.cluster.node_pools.keys())
        errors: list[str] = []

        for name, svc in self.services.items():
            pool_ref = getattr(svc, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(f"services.{name}.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}")
            tp = getattr(svc, "tensor_parallel_size", None)
            if tp and pool_ref and pool_ref in pools:
                pool = self.cluster.node_pools[pool_ref]
                if pool.gres:
                    match = re.search(r"gpu:(\d+)", pool.gres)
                    if match and int(match.group(1)) < tp:
                        errors.append(
                            f"services.{name} needs tensor_parallel_size={tp} "
                            f"but node_pool {pool_ref!r} only has {pool.gres}"
                        )

        multinode_by_pool: dict[str, list[str]] = {}
        for name, svc in self.services.items():
            num_nodes = getattr(svc, "num_nodes", 1)
            if num_nodes > 1:
                pool_ref = getattr(svc, "node_pool", None)
                if pool_ref:
                    multinode_by_pool.setdefault(pool_ref, []).append(name)
                    pool = self.cluster.node_pools.get(pool_ref)
                    if pool and pool.nodes < num_nodes:
                        errors.append(
                            f"services.{name}.num_nodes={num_nodes} exceeds "
                            f"node_pools.{pool_ref}.nodes={pool.nodes}. "
                            f"Set pool nodes >= service num_nodes."
                        )
        for pool_ref, svc_names in multinode_by_pool.items():
            if len(svc_names) > 1:
                errors.append(
                    f"Multiple multi-node services {svc_names} in pool {pool_ref!r}. "
                    f"Use separate node_pools or a het-job."
                )

        for sb_name, sb in self.sandboxes.items():
            pool_ref = getattr(sb, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(f"sandboxes.{sb_name}.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}")

        for i, bench in enumerate(self.benchmarks):
            pool_ref = getattr(bench.sandbox, "node_pool", None)
            if pool_ref and pool_ref not in pools:
                errors.append(
                    f"benchmarks[{i}].sandbox.node_pool={pool_ref!r} not in cluster.node_pools {sorted(pools)}"
                )

        if errors:
            raise ValueError("Node pool errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_shards(self) -> EvalConfig:
        """Sharding requires a single node pool (incompatible with het-jobs)."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        if self.cluster.shards is None:
            return self
        pool_refs: set[str] = set()
        for svc in self.services.values():
            p = getattr(svc, "node_pool", None)
            if p:
                pool_refs.add(p)
        for sb in self.sandboxes.values():
            p = getattr(sb, "node_pool", None)
            if p:
                pool_refs.add(p)
        for bench in self.benchmarks:
            p = getattr(bench.sandbox, "node_pool", None)
            if p:
                pool_refs.add(p)
        if len(pool_refs) > 1:
            raise ValueError(
                f"shards={self.cluster.shards} requires a single node pool, "
                f"but {len(pool_refs)} distinct pools are referenced: "
                f"{sorted(pool_refs)}. SLURM --array is incompatible with "
                f"heterogeneous jobs."
            )
        return self

    @model_validator(mode="after")
    def _validate_dependency_graph(self) -> EvalConfig:
        """Ensure depends_on references exist and form a DAG."""
        available = set(self.services.keys())
        errors: list[str] = []
        graph: dict[str, list[str]] = {}

        for name, svc in self.services.items():
            deps = getattr(svc, "depends_on", [])
            graph[name] = deps
            for dep in deps:
                if dep not in available:
                    errors.append(f"services.{name}.depends_on references {dep!r}, not in services")
                if dep == name:
                    errors.append(f"services.{name}.depends_on contains self-reference")

        visited: set[str] = set()
        in_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in graph.get(node, []):
                if _has_cycle(dep):
                    return True
            in_stack.discard(node)
            return False

        for svc_name in graph:
            if _has_cycle(svc_name):
                errors.append(f"Circular dependency detected involving service {svc_name!r}")
                break

        if errors:
            raise ValueError("Dependency errors:\n  " + "\n  ".join(errors))
        return self

    @model_validator(mode="after")
    def _validate_output_dir(self) -> EvalConfig:
        """Require absolute output.dir for remote SLURM execution.

        Relative paths break container mounts, remote mkdir, and scp —
        producing cryptic "died during startup" failures instead of a
        clear error at config time.
        """
        if not isinstance(self.cluster, SlurmCluster):
            return self
        if not self.cluster.hostname:
            return self
        d = self.output.dir
        if not d.startswith("/"):
            raise ValueError(f"output.dir must be an absolute path for remote SLURM execution, got: {d!r}")
        return self

    @model_validator(mode="after")
    def _validate_timeout_hierarchy(self) -> EvalConfig:
        """Warn if benchmark + startup timeouts exceed SLURM walltime."""
        if not isinstance(self.cluster, SlurmCluster):
            return self
        walltime_sec = _parse_walltime(self.cluster.walltime)
        max_startup = max(
            (getattr(svc, "startup_timeout", 0) for svc in self.services.values()),
            default=0,
        )
        for bench in self.benchmarks:
            total = max_startup + bench.timeout
            if total > walltime_sec:
                warnings.warn(
                    f"benchmark '{bench.name}': startup ({max_startup}s) + "
                    f"timeout ({bench.timeout}s) = {total}s exceeds "
                    f"walltime ({self.cluster.walltime} = {walltime_sec}s)",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    # ── Convenience methods for runtime consumers ──

    def get_service(self, name: str) -> ServiceConfig:
        svc = self.services.get(name)
        if svc is None:
            raise KeyError(f"Unknown service {name!r}. Available: {sorted(self.services)}")
        return svc

    def get_model_url(self, service_name: str) -> str:
        return self.get_service(service_name).base_url

    def get_model_id(self, service_name: str) -> str:
        svc = self.get_service(service_name)
        if getattr(svc, "served_model_name", None):
            return svc.served_model_name
        if getattr(svc, "is_managed", False):
            return service_name
        return getattr(svc, "model", None) or service_name

    def get_api_key(self, service_name: str) -> str | None:
        svc = self.get_service(service_name)
        return getattr(svc, "api_key", None)

    def managed_services(self) -> dict[str, ServiceConfig]:
        return {k: v for k, v in self.services.items() if v.is_managed}


# ── Parser (env-var expansion + validation) ──

_ENV_RE = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

_ESC_SENTINEL = "\x00_NEL_ESC\x00"


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in strings.

    Strict: raises ValueError if a variable has no value and no default.
    Use ``${VAR:-}`` to explicitly allow empty string fallback.

    To defer expansion to runtime (e.g. for vars only available on a compute
    node), escape with a double dollar: ``$${VAR}``.  This is converted to
    the literal ``${VAR}`` and left for the shell to resolve later.  The
    escaping applies to env-var expansion only, not to ``${.path}``
    self-references (which are resolved earlier by ``compose_config``).
    """

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        default = m.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable ${{{var_name}}} is not set "
            f"and no default was provided. "
            f"Use ${{{var_name}:-default_value}} to set a fallback."
        )

    if isinstance(value, str):
        protected = value.replace("$${", _ESC_SENTINEL)
        expanded = _ENV_RE.sub(_replace, protected)
        return expanded.replace(_ESC_SENTINEL, "${")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _resolve_playbooks(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand ``playbook:`` references in benchmark entries.

    If a benchmark entry contains ``playbook: <name>``, the named playbook
    YAML is loaded from the built-in ``playbooks/`` directory (or a file path)
    and deep-merged with any explicit overrides in the entry.
    """
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, list):
        return raw

    from pathlib import Path

    import yaml

    from nemo_evaluator.config.compose import _deep_merge

    playbooks_dir = Path(__file__).resolve().parent.parent / "playbooks"

    resolved = []
    for entry in benchmarks:
        if not isinstance(entry, dict) or "playbook" not in entry:
            resolved.append(entry)
            continue

        ref = entry.pop("playbook")
        pb_path = playbooks_dir / ref
        if not pb_path.is_file():
            for ext in (".yaml", ".yml"):
                candidate = playbooks_dir / f"{ref}{ext}"
                if candidate.is_file():
                    pb_path = candidate
                    break
        if not pb_path.is_file():
            pb_path = Path(ref)
            if not pb_path.is_absolute():
                pb_path = Path.cwd() / pb_path

        if not pb_path.is_file():
            raise FileNotFoundError(f"Playbook not found: {ref!r} (checked {playbooks_dir} and {Path.cwd()})")

        base = yaml.safe_load(pb_path.read_text()) or {}
        merged = _deep_merge(base, entry)
        resolved.append(merged)

    raw["benchmarks"] = resolved
    return raw


def parse_eval_config(raw: dict[str, Any]) -> EvalConfig:
    """Parse and validate a raw YAML dict, expanding env vars.

    This is the required entry point.  Do not call EvalConfig.model_validate()
    directly — env-var expansion would be skipped.
    """
    raw = _resolve_playbooks(raw)
    expanded = _expand_env(raw)
    return EvalConfig.model_validate(expanded)
