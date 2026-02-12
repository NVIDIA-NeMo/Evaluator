# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NeMo-Skills plugin runner for in-process evaluation execution."""

import asyncio
import difflib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional

from nemo_evaluator._nemo_skills._adapters.file_utils import jdump
from nemo_evaluator.api.api_dataclasses import Evaluation, EvaluationResult
from nemo_evaluator.client.client import NeMoEvaluatorClient
from nemo_evaluator.logging import get_logger
from nemo_evaluator.plugins.nemo_skills.metrics_adapter import translate

logger = get_logger(__name__)

# Evaluator map per C-021: maps eval_type to scorer function names (immutable)
EVALUATOR_MAP: MappingProxyType = MappingProxyType({
    "math": "score_math",
    "gsm8k": "score_math",
    "aime24": "score_math",
    "amc23": "score_math",
    "aime25": "score_math",
    "mmlu": "score_multichoice",
    "mmlu_pro": "score_multichoice",
    "gpqa": "score_multichoice",
    "mrcr": "score_mrcr",
    "ruler": "score_ruler",
    "bfcl": "score_bfcl",
    "ifeval": "score_if",
    "humaneval": "score_code",
    "mbpp": "score_code",
    "arena": "score_arena",
    "audio": "score_audio",
    "llm_judge": "score_llm_judge",
})


def evaluate(
    evaluation: Evaluation,
    client: NeMoEvaluatorClient,
    output_dir: str,
) -> EvaluationResult:
    """Native entrypoint for nemo-skills evaluation pipeline.

    Executes the 7-step runner pipeline per C-017:
    1. Extract config from evaluation.config.params.extra
    2. Load benchmark data
    3. Construct prompts
    4. Call model (per seed if num_seeds > 1)
    5. Merge responses with data
    6. Score data using appropriate scorer
    7. Compute metrics and translate to EvaluationResult

    Args:
        evaluation: Fully-merged evaluation configuration
        client: NeMo Evaluator client for model calls
        output_dir: Directory for output artifacts

    Returns:
        EvaluationResult with tasks populated

    Raises:
        ValueError: If required config fields are missing or invalid
        FileNotFoundError: If benchmark data files are not found
    """
    # Step 1: Extract config
    extra = evaluation.config.params.extra or {}
    benchmark_name = extra.get("benchmark_name")
    eval_type = extra.get("eval_type")
    data_dir = extra.get("data_dir")
    eval_split = extra.get("eval_split", "test")
    num_seeds = extra.get("num_seeds", 1)
    limit_samples = evaluation.config.params.limit_samples

    if not benchmark_name:
        raise ValueError("config.params.extra must contain 'benchmark_name'")
    if not eval_type:
        raise ValueError("config.params.extra must contain 'eval_type'")
    if not data_dir:
        raise ValueError("config.params.extra must contain 'data_dir'")
    if eval_type not in EVALUATOR_MAP:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    # Convert limit_samples to int if needed
    if isinstance(limit_samples, float):
        limit_samples = int(limit_samples)

    # Step 2: Load data
    data = load_benchmark_data(
        benchmark_name=benchmark_name,
        data_dir=data_dir,
        eval_split=eval_split,
        limit_samples=limit_samples,
    )

    # Step 3: Construct prompts
    system_prompt = extra.get("system_prompt")
    prompt_config = extra.get("prompt_config")
    prompts = construct_prompts(data, prompt_config, system_prompt)

    # Step 4: Call model (per seed)
    temperature = evaluation.config.params.temperature or 0.0
    max_tokens = evaluation.config.params.max_new_tokens or 512

    all_responses = []
    for seed_idx in range(num_seeds):
        seed = seed_idx if num_seeds > 1 else None
        responses = asyncio.run(
            call_model_batch(client, prompts, temperature, max_tokens, seed)
        )
        all_responses.append(responses)

    # Step 5: Merge responses (for now, use first seed; multi-seed logic TBD)
    for i, sample in enumerate(data):
        sample["generation"] = all_responses[0][i]

    # Step 6: Score
    scorer_fn_name = EVALUATOR_MAP[eval_type]
    scorer_fn = _SCORER_DISPATCH[scorer_fn_name]
    config = {"judge_model_call_fn": None}  # Placeholder for judge config
    scored_data = scorer_fn(data, config)

    # Step 7: Compute metrics
    ns_metrics = compute_ns_metrics(scored_data, benchmark_name, eval_type)

    # Write ns_results.json
    ns_results = {
        "_all_": ns_metrics,
        "benchmark_name": benchmark_name,
        "config": {
            "benchmark_name": benchmark_name,
            "eval_type": eval_type,
            "eval_split": eval_split,
            "num_seeds": num_seeds,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }
    output_path = Path(output_dir) / "ns_results.json"
    jdump(ns_results, str(output_path))

    # Translate to EvaluationResult (wrap in _all_ structure expected by translate)
    result = translate({"_all_": ns_metrics}, benchmark_name, config)
    return result


def load_benchmark_data(
    benchmark_name: str,
    data_dir: str,
    eval_split: str,
    limit_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Loads benchmark data from JSONL file.

    Per C-018 and OQR-001: tries {data_dir}/{benchmark}/{eval_split}.jsonl first.
    Falls back to {data_dir}/{benchmark}/test.jsonl if eval_split != "test".

    Args:
        benchmark_name: Benchmark identifier
        data_dir: Root data directory
        eval_split: Data split name (e.g., "test", "validation")
        limit_samples: Optional limit on number of samples

    Returns:
        List of sample dictionaries

    Raises:
        ValueError: If data_dir is None
        FileNotFoundError: If no data file can be found
    """
    if data_dir is None:
        raise ValueError("data_dir must not be None")

    primary_path = Path(data_dir) / benchmark_name / f"{eval_split}.jsonl"
    fallback_path = Path(data_dir) / benchmark_name / "test.jsonl"

    data_path = None
    if primary_path.exists():
        data_path = primary_path
    elif eval_split != "test" and fallback_path.exists():
        data_path = fallback_path
        logger.info(f"Split '{eval_split}' not found, falling back to test.jsonl")

    if data_path is None:
        raise FileNotFoundError(
            f"Dataset file not found: {primary_path} (and fallback {fallback_path} "
            f"also not found)"
        )

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            if limit_samples and len(data) >= limit_samples:
                break

    return data


def construct_prompts(
    data: List[Dict[str, Any]],
    prompt_config: Optional[Dict[str, Any]],
    system_prompt: Optional[str],
) -> List[List[Dict[str, str]]]:
    """Constructs message lists for each sample.

    Per C-019: detects text field (problem, question, prompt in that order).
    If none found, uses json.dumps(sample) as content.
    Prepends system message if system_prompt is not None.

    Args:
        data: List of sample dictionaries
        prompt_config: Optional prompt configuration (unused in this stub)
        system_prompt: Optional system prompt to prepend

    Returns:
        List of message lists (one per sample)
    """
    prompts = []
    for sample in data:
        # Detect text field
        content = None
        for field in ["problem", "question", "prompt"]:
            if field in sample:
                content = sample[field]
                break
        if content is None:
            content = json.dumps(sample)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        prompts.append(messages)

    return prompts


async def call_model_batch(
    client: NeMoEvaluatorClient,
    prompts: List[List[Dict[str, str]]],
    temperature: float,
    max_tokens: int,
    seed: Optional[int] = None,
) -> List[str]:
    """Calls model for each prompt, with per-sample fault isolation.

    Per C-020 and INV-009: on failure for any call, logs warning and records
    empty string. Never raises exception. Never aborts batch.

    Args:
        client: NeMo Evaluator client
        prompts: List of message lists
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        seed: Optional random seed

    Returns:
        List of response strings (same length as prompts)
    """
    responses = []
    for idx, messages in enumerate(prompts):
        try:
            response = await client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            responses.append(response)
        except Exception as e:
            logger.warning(
                f"Model call failed for sample {idx}: {type(e).__name__}: {str(e)}"
            )
            responses.append("")

    return responses


# --- Scorer Functions (C-022 through C-031) ---


def score_math(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores math problems using symbolic answer extraction.

    Per C-022: extracts predicted answers from \\boxed{} notation or
    "answer is/=/:" patterns. Sets predicted_answer, symbolic_correct, no_answer.

    Per INV-003: returns the same list object (identity), mutates in-place.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Scorer configuration (unused)

    Returns:
        The same list object with scoring fields added
    """
    for sample in data:
        generation = sample.get("generation", "")
        expected_answer = sample.get("expected_answer", sample.get("answer", ""))

        # Extract boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", generation)
        if boxed_match:
            predicted_answer = boxed_match.group(1).strip()
        else:
            # Try "answer is/=/:pattern
            answer_match = re.search(
                r"(?:answer is|answer=|answer:|the answer is)\s*([^\n.]+)",
                generation,
                re.IGNORECASE,
            )
            if answer_match:
                predicted_answer = answer_match.group(1).strip()
            else:
                predicted_answer = ""

        sample["predicted_answer"] = predicted_answer
        sample["no_answer"] = len(predicted_answer) == 0

        # Symbolic correctness: normalize and compare
        sample["symbolic_correct"] = _normalize_answer(predicted_answer) == _normalize_answer(str(expected_answer))

    return data


def score_multichoice(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores multiple choice problems.

    Per C-023: extracts predicted answers from \\boxed{A-J}, "answer/choice is (X)",
    or last standalone capital letter patterns.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Scorer configuration (unused)

    Returns:
        The same list object with scoring fields added
    """
    for sample in data:
        generation = sample.get("generation", "")
        expected_answer = sample.get("expected_answer", sample.get("answer", ""))

        # Extract boxed answer
        boxed_match = re.search(r"\\boxed\{([A-J])\}", generation)
        if boxed_match:
            predicted_answer = boxed_match.group(1).strip()
        else:
            # Try "answer/choice is (X)" pattern
            answer_match = re.search(
                r"(?:answer|choice) is \(?([A-J])\)?",
                generation,
                re.IGNORECASE,
            )
            if answer_match:
                predicted_answer = answer_match.group(1).strip()
            else:
                # Last standalone capital letter
                letters = re.findall(r"\b([A-J])\b", generation)
                predicted_answer = letters[-1] if letters else ""

        sample["predicted_answer"] = predicted_answer
        sample["symbolic_correct"] = predicted_answer.upper() == str(expected_answer).upper()

    return data


def score_mrcr(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores MRCR (reading comprehension) using sequence matching.

    Per C-024: computes difflib.SequenceMatcher ratio between generation and
    expected answer. Sets seq_match_ratio, is_correct.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Scorer configuration (unused)

    Returns:
        The same list object with scoring fields added
    """
    for sample in data:
        generation = sample.get("generation", "")
        expected_answer = sample.get("expected_answer", sample.get("answer", ""))

        ratio = difflib.SequenceMatcher(None, generation, str(expected_answer)).ratio()
        sample["seq_match_ratio"] = ratio
        sample["is_correct"] = ratio >= 0.8  # Threshold for correctness

    return data


def score_ruler(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores RULER benchmark using string containment.

    Per C-025: checks string containment. For list targets, computes fraction
    of target strings found in generation.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Scorer configuration (unused)

    Returns:
        The same list object with scoring fields added
    """
    for sample in data:
        generation = sample.get("generation", "")
        expected_answer = sample.get("expected_answer", sample.get("answer", ""))

        if isinstance(expected_answer, list):
            # Fractional match for list targets
            found_count = sum(1 for target in expected_answer if target in generation)
            sample["is_correct"] = found_count / len(expected_answer) if expected_answer else 0.0
        else:
            # Boolean match for single string
            sample["is_correct"] = str(expected_answer) in generation

    return data


def score_bfcl(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores BFCL benchmark using exact string match.

    Per C-026: performs exact string match (stripped) between generation and
    expected answer.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Scorer configuration (unused)

    Returns:
        The same list object with scoring fields added
    """
    for sample in data:
        generation = sample.get("generation", "").strip()
        expected_answer = str(sample.get("expected_answer", sample.get("answer", ""))).strip()

        sample["is_correct"] = generation == expected_answer

    return data


def score_llm_judge(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scores using LLM judge.

    Per C-027 and OQR-008: invokes LLM judge via config["judge_model_call_fn"].
    On failure, sets judge_error, is_correct=False, judge_verdict="error".
    Never raises exception.

    Args:
        data: List of sample dictionaries with "generation" field
        config: Must contain "judge_model_call_fn" callable

    Returns:
        The same list object with scoring fields added
    """
    judge_fn = config.get("judge_model_call_fn")

    for sample in data:
        if judge_fn is None:
            sample["judge_error"] = "judge_model_call_fn not provided"
            sample["is_correct"] = False
            sample["judge_verdict"] = "error"
            continue

        try:
            prompt = f"Evaluate this response: {sample.get('generation', '')}"
            verdict = judge_fn(prompt)
            sample["judge_verdict"] = verdict
            sample["is_correct"] = "correct" in verdict.lower()
            sample["judge_error"] = None
        except Exception as e:
            sample["judge_error"] = f"{type(e).__name__}: {str(e)}"
            sample["is_correct"] = False
            sample["judge_verdict"] = "error"

    return data


def score_if(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder scorer for IFEval. Per C-028."""
    for sample in data:
        sample["needs_external_scoring"] = True
    return data


def score_code(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder scorer for code execution. Per C-029."""
    for sample in data:
        sample["needs_sandbox"] = True
    return data


def score_arena(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder scorer for arena evaluation. Per C-030."""
    for sample in data:
        sample["needs_judge"] = True
    return data


def score_audio(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder scorer for audio evaluation. Per C-031."""
    for sample in data:
        sample["needs_audio_scoring"] = True
    return data


# --- Helper Functions ---


def _normalize_answer(answer: str) -> str:
    """Normalizes answer for comparison."""
    return answer.lower().strip().replace(",", "").replace(" ", "")


def compute_ns_metrics(
    data: List[Dict[str, Any]],
    benchmark_name: str,
    eval_type: str,
) -> Dict[str, Any]:
    """Computes nemo-skills style metrics from scored data.

    Returns metrics in the structure expected by metrics_adapter.translate().

    Args:
        data: Scored sample data
        benchmark_name: Benchmark identifier
        eval_type: Evaluation type

    Returns:
        Metrics dictionary with aggregation modes
    """
    # Detect primary metric field
    metric_field = None
    for field in ["symbolic_correct", "is_correct", "seq_match_ratio"]:
        if field in data[0]:
            metric_field = field
            break

    if metric_field is None:
        # Placeholder metrics
        return {
            "greedy": {
                "num_entries": len(data),
            }
        }

    # Compute metric
    if metric_field == "seq_match_ratio":
        # Average ratio
        total = math.fsum(sample.get(metric_field, 0.0) for sample in data)
        metric_value = (total / len(data)) * 100 if data else 0.0
    else:
        # Boolean accuracy
        correct_count = sum(1 for sample in data if sample.get(metric_field, False))
        metric_value = (correct_count / len(data)) * 100 if data else 0.0

    return {
        "greedy": {
            metric_field: metric_value,
            "num_entries": len(data),
        }
    }


# Explicit scorer dispatch dict (avoids globals() fragility)
_SCORER_DISPATCH: Dict[str, Callable] = {
    "score_math": score_math,
    "score_multichoice": score_multichoice,
    "score_mrcr": score_mrcr,
    "score_ruler": score_ruler,
    "score_bfcl": score_bfcl,
    "score_llm_judge": score_llm_judge,
    "score_if": score_if,
    "score_code": score_code,
    "score_arena": score_arena,
    "score_audio": score_audio,
}

# Module-level validation per INV-013
for _fn_name in EVALUATOR_MAP.values():
    if _fn_name not in _SCORER_DISPATCH:
        raise RuntimeError(f"EVALUATOR_MAP references unknown scorer: {_fn_name}")
