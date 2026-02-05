#!/usr/bin/env python3
"""
Build a complete NEL config by combining config excerpts from assets.

Usage:
    # Output to specific file
    python build_config.py \
        --execution local \
        --deployment vllm \
        --export mlflow \
        --model-type chat \
        --benchmarks standard code \
        --output my_config.yaml

    # Output to directory (auto-generates filename from combination)
    python build_config.py \
        --execution local \
        --deployment vllm \
        --export mlflow \
        --model-type chat \
        --benchmarks standard code \
        --output ./configs/

    # Output to current directory with auto-generated filename
    python build_config.py \
        --execution local \
        --deployment vllm \
        --model-type chat \
        --benchmarks standard code

Output path behavior (--output / -o):
    - File path (*.yaml): Write directly to that file
    - Directory path: Auto-generate filename and write to that directory
    - Not specified: Auto-generate filename and write to current directory

Auto-generated filenames use the pattern:
    {execution}_{deployment}_{model_type}_{benchmarks}.yaml
    Example: local_vllm_chat_standard_code.yaml

Existing files are never overwritten - a numeric suffix is added if needed.
"""

import argparse
import contextlib
import os
from pathlib import Path
from typing import List, Optional

import yaml

ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Mock environment variables for validation testing
MOCK_ENV_VARS = {
    "HF_TOKEN": "mock-hf-token",
    "NGC_API_KEY": "mock-ngc-api-key",
    "OPENAI_API_KEY": "mock-openai-api-key",
    "JUDGE_API_KEY": "mock-judge-api-key",
    "WANDB_API_KEY": "mock-wandb-api-key",
    "API_KEY": "mock-api-key",
    "TEST_KEY": "mock-test-key",
    "HF_TOKEN_FOR_GPQA_DIAMOND": "mock-hf-token-gpqa",
    "HF_TOKEN_FOR_AEGIS_V2": "mock-hf-token-aegis",
}


@contextlib.contextmanager
def mock_env_vars():
    """Context manager to set up and restore mock environment variables."""
    original_values = {}
    try:
        for key, value in MOCK_ENV_VARS.items():
            original_values[key] = os.environ.get(key)
            if original_values[key] is None:
                os.environ[key] = value
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def get_mock_overrides(
    execution: str,
    deployment: str,
    export: str,
    model_type: str,
    benchmarks: List[str],
) -> List[str]:
    """Generate mock overrides for required (???) fields based on config choices."""
    overrides = [
        "execution.output_dir=/tmp/nel-test-results",
    ]

    # SLURM configs need additional mocks
    if execution == "slurm":
        overrides.extend(
            [
                "execution.hostname=test-slurm-host.example.com",
                "execution.account=test-account",
            ]
        )

    # Deployment-specific mocks
    if deployment == "none":
        overrides.extend(
            [
                "target.api_endpoint.model_id=test/model",
                "target.api_endpoint.url=https://test-api.example.com/v1/chat/completions",
                "target.api_endpoint.api_key_name=TEST_KEY",
            ]
        )
    elif deployment == "vllm":
        overrides.extend(
            [
                "deployment.hf_model_handle=test/model",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "sglang":
        overrides.extend(
            [
                "deployment.hf_model_handle=test/model",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "nim":
        overrides.extend(
            [
                "deployment.image=nvcr.io/nim/test/model:latest",
                "deployment.served_model_name=test/model",
            ]
        )
    elif deployment == "trtllm":
        overrides.extend(
            [
                "deployment.checkpoint_path=/path/to/checkpoint",
                "deployment.served_model_name=test/model",
            ]
        )

    # Export-specific mocks
    if export == "mlflow":
        overrides.append("export.mlflow.tracking_uri=http://test-mlflow:5000")
    elif export == "wandb":
        overrides.append("export.wandb.project=test-project")

    # Model type specific mocks
    if model_type == "base":
        overrides.append(
            "evaluation.nemo_evaluator_config.config.params.extra.tokenizer=test/tokenizer"
        )
    elif model_type == "reasoning":
        overrides.append(
            "evaluation.nemo_evaluator_config.target.api_endpoint.adapter_config.custom_system_prompt=/think"
        )

    # Safety benchmark mocks (need judge URL)
    # Only apply if the safety benchmark file exists for this model type
    safety_benchmark_path = ASSETS_DIR / "evaluation" / model_type / "safety.yaml"
    if "safety" in benchmarks and safety_benchmark_path.exists():
        # Find aegis task and set judge URL
        overrides.append(
            "++evaluation.tasks.1.nemo_evaluator_config.config.params.extra.judge.url=https://test-judge.example.com/v1"
        )

    return overrides


def generate_config_filename(
    execution: str,
    deployment: str,
    model_type: str,
    benchmarks: List[str],
) -> str:
    """Generate a config filename from the combination of choices."""
    bench_str = "_".join(sorted(benchmarks))
    return f"{execution}_{deployment}_{model_type}_{bench_str}.yaml"


def get_unique_filepath(filepath: Path) -> Path:
    """Return a unique filepath by adding numeric suffix if file exists."""
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    counter = 1

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def resolve_output_path(
    output: Optional[Path],
    execution: str,
    deployment: str,
    model_type: str,
    benchmarks: List[str],
) -> Path:
    """
    Resolve the output path based on what was provided.

    Args:
        output: User-provided output path (file, directory, or None)
        execution, deployment, model_type, benchmarks: Config choices for filename

    Returns:
        Resolved output file path (unique, won't overwrite existing)
    """
    filename = generate_config_filename(execution, deployment, model_type, benchmarks)

    if output is None:
        # No output specified: use current directory with auto-generated filename
        filepath = Path.cwd() / filename
    elif output.suffix in (".yaml", ".yml"):
        # Looks like a file path: use as-is
        filepath = output
    elif output.is_dir() or not output.suffix:
        # Directory path: auto-generate filename in that directory
        output.mkdir(parents=True, exist_ok=True)
        filepath = output / filename
    else:
        # Assume it's a file path
        filepath = output

    return get_unique_filepath(filepath)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif (
            key in result and isinstance(result[key], list) and isinstance(value, list)
        ):
            # For lists (like tasks), extend rather than replace
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config(
    execution: str,
    deployment: str,
    export: str,
    model_type: str,
    benchmarks: List[str],
    output: Optional[Path] = None,
) -> dict:
    """
    Build a complete NEL config by combining excerpts.

    Args:
        execution: Execution type (local, slurm)
        deployment: Deployment type (none, vllm, sglang, nim, trtllm)
        export: Export type (none, mlflow, wandb)
        model_type: Model type (base, chat, reasoning)
        benchmarks: List of benchmark types (standard, code, math_reasoning, safety)
        output: Optional output file path

    Returns:
        Combined config dictionary
    """
    config = {}

    # 1. Load execution config (base execution settings)
    execution_path = ASSETS_DIR / "execution" / f"{execution}.yaml"
    config = deep_merge(config, load_yaml(execution_path))

    # 2. Load export config
    export_path = ASSETS_DIR / "export" / f"{export}.yaml"
    config = deep_merge(config, load_yaml(export_path))

    # 3. Load model type default config (sets default parallelism, temp, etc.)
    model_default_path = ASSETS_DIR / "evaluation" / model_type / "default.yaml"
    config = deep_merge(config, load_yaml(model_default_path))

    # 4. Load benchmark configs
    for benchmark in benchmarks:
        benchmark_path = ASSETS_DIR / "evaluation" / model_type / f"{benchmark}.yaml"
        if benchmark_path.exists():
            config = deep_merge(config, load_yaml(benchmark_path))
        else:
            print(f"Warning: Benchmark config not found: {benchmark_path}")

    # 5. Load deployment config (applied last so deployment-specific overrides take effect)
    # e.g., none.yaml sets parallelism: 1 for rate-limited external APIs
    deployment_path = ASSETS_DIR / "deployment" / f"{deployment}.yaml"
    config = deep_merge(config, load_yaml(deployment_path))

    # 6. Ensure _self_ is at the very end of defaults list (required by Hydra)
    if "defaults" in config:
        # Remove any existing _self_ entries
        config["defaults"] = [d for d in config["defaults"] if d != "_self_"]
        config["defaults"].append("_self_")
    else:
        config["defaults"] = ["_self_"]

    # 7. Write output if specified
    if output:
        with open(output, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config written to: {output}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Build NEL config from excerpts")
    parser.add_argument(
        "--execution",
        "-e",
        choices=["local", "slurm"],
        required=True,
        help="Execution type",
    )
    parser.add_argument(
        "--deployment",
        "-d",
        choices=["none", "vllm", "sglang", "nim", "trtllm"],
        required=True,
        help="Deployment type",
    )
    parser.add_argument(
        "--export",
        "-x",
        choices=["none", "mlflow", "wandb"],
        default="none",
        help="Export type (default: none)",
    )
    parser.add_argument(
        "--model-type",
        "-m",
        choices=["base", "chat", "reasoning"],
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--benchmarks",
        "-b",
        nargs="+",
        choices=["standard", "code", "math_reasoning", "safety", "multilingual"],
        required=True,
        help="Benchmark types to include",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path: file (*.yaml), directory, or omit for current dir. "
        "Auto-generates filename from config choices. Never overwrites existing files.",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate the generated config using verify_config.py",
    )

    args = parser.parse_args()

    # Resolve output path (handles file/dir/none cases, avoids overwrites)
    output_path = resolve_output_path(
        output=args.output,
        execution=args.execution,
        deployment=args.deployment,
        model_type=args.model_type,
        benchmarks=args.benchmarks,
    )

    build_config(
        execution=args.execution,
        deployment=args.deployment,
        export=args.export,
        model_type=args.model_type,
        benchmarks=args.benchmarks,
        output=output_path,
    )

    # Optionally validate the generated config
    if args.validate:
        from verify_config import resolve_config, validate_config

        try:
            # Generate mock overrides for ??? values
            overrides = get_mock_overrides(
                execution=args.execution,
                deployment=args.deployment,
                export=args.export,
                model_type=args.model_type,
                benchmarks=args.benchmarks,
            )
            # Use mock environment variables during validation
            with mock_env_vars():
                cfg = resolve_config(str(output_path), overrides)
                valid, errors, warnings = validate_config(cfg)
            if valid:
                print("✓ Configuration is valid!")
                for warn in warnings:
                    print(f"  ⚠ Warning: {warn}")
            else:
                print("✗ Configuration has errors:")
                for err in errors:
                    print(f"  - {err}")
        except Exception as e:
            print(f"✗ Validation failed: {e}")


if __name__ == "__main__":
    main()
