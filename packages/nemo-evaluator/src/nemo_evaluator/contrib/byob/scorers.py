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

"""Built-in scorer functions for BYOB benchmarks.

All scorers accept a single ``ScorerInput`` argument and return a dict
of metric name -> numeric/bool value.

Example::

    from nemo_evaluator.contrib.byob.scorers import exact_match
    from nemo_evaluator.contrib.byob.decorators import ScorerInput

    result = exact_match(ScorerInput(response="Yes", target="yes", metadata={}))
    # {"correct": True}
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.contrib.byob.decorators import ScorerInput


def _to_str(value) -> str:
    """Coerce a target value to string for text-based scoring."""
    return str(value) if not isinstance(value, str) else value


def contains(sample: ScorerInput) -> dict:
    """Check if target string is contained in response (case-insensitive)."""
    target = _to_str(sample.target)
    return {"correct": target.lower().strip() in sample.response.lower()}


def exact_match(sample: ScorerInput) -> dict:
    """Check if response exactly matches target (case-insensitive, whitespace-stripped)."""
    target = _to_str(sample.target)
    return {"correct": sample.response.strip().lower() == target.strip().lower()}


def f1_token(sample: ScorerInput) -> dict:
    """Compute token-level F1 score between response and target."""
    target = _to_str(sample.target)
    pred_tokens = sample.response.lower().split()
    ref_tokens = target.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"f1": f1, "precision": precision, "recall": recall}


def regex_match(sample: ScorerInput) -> dict:
    """Check if target regex pattern matches anywhere in response."""
    try:
        return {"correct": bool(re.search(sample.target, sample.response, re.IGNORECASE))}
    except re.error:
        return {"correct": False}


def any_of(*scorer_fns):
    """Combine scorers: correct if ANY scorer returns correct=True.

    Example::

        from nemo_evaluator.contrib.byob.scorers import contains, exact_match, any_of
        combined = any_of(contains, exact_match)
    """
    def combined(sample: ScorerInput) -> dict:
        results = {}
        any_correct = False
        for fn in scorer_fns:
            sub_result = fn(sample)
            for key, value in sub_result.items():
                results[f"{fn.__name__}_{key}"] = value
            if sub_result.get("correct"):
                any_correct = True
        results["correct"] = any_correct
        return results
    combined.__name__ = f"any_of({', '.join(f.__name__ for f in scorer_fns)})"
    return combined


def all_of(*scorer_fns):
    """Combine scorers: correct if ALL scorers return correct=True.

    Example::

        from nemo_evaluator.contrib.byob.scorers import contains, exact_match, all_of
        combined = all_of(contains, exact_match)
    """
    def combined(sample: ScorerInput) -> dict:
        results = {}
        all_correct = True
        for fn in scorer_fns:
            sub_result = fn(sample)
            for key, value in sub_result.items():
                results[f"{fn.__name__}_{key}"] = value
            if not sub_result.get("correct"):
                all_correct = False
        results["correct"] = all_correct
        return results
    combined.__name__ = f"all_of({', '.join(f.__name__ for f in scorer_fns)})"
    return combined


# ---------------------------------------------------------------------------
# BLEU scorer (sentence-level, add-1 / Laplace smoothing)
# ---------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Return a Counter of n-gram tuples from *tokens*."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(sample: ScorerInput) -> dict:
    """Compute sentence-level BLEU-1 through BLEU-4 with add-1 smoothing.

    Returns:
        dict with keys ``bleu_1``, ``bleu_2``, ``bleu_3``, ``bleu_4``.
    """
    pred_tokens = sample.response.lower().split()
    ref_tokens = sample.target.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / len(pred_tokens)) if len(pred_tokens) < len(ref_tokens) else 1.0

    scores: dict[str, float] = {}
    log_avg = 0.0

    for n in range(1, 5):
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)

        clipped = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng)
        total = sum(pred_ng.values())

        # Add-1 (Laplace) smoothing: add 1 to both numerator and denominator
        precision = (clipped + 1) / (total + 1)

        log_avg += math.log(precision) / 4
        bleu_n = bp * math.exp(log_avg * 4 / n)  # geometric mean of first n precisions
        scores[f"bleu_{n}"] = bleu_n

    return scores


# ---------------------------------------------------------------------------
# ROUGE scorer (ROUGE-1, ROUGE-2, ROUGE-L)
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    """Compute the harmonic mean (F1) of *precision* and *recall*."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    """Compute ROUGE-N F1 between prediction and reference token lists."""
    pred_ng = _ngrams(pred_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)

    overlap = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng)
    pred_count = sum(pred_ng.values())
    ref_count = sum(ref_ng.values())

    if pred_count == 0 or ref_count == 0:
        return 0.0

    precision = overlap / pred_count
    recall = overlap / ref_count
    return _f1(precision, recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Return the length of the longest common subsequence of *x* and *y*."""
    m, n = len(x), len(y)
    # Space-optimised: only keep two rows.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge(sample: ScorerInput) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Returns:
        dict with keys ``rouge_1``, ``rouge_2``, ``rouge_l``.
    """
    pred_tokens = sample.response.lower().split()
    ref_tokens = sample.target.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    rouge_1 = _rouge_n_f1(pred_tokens, ref_tokens, 1)
    rouge_2 = _rouge_n_f1(pred_tokens, ref_tokens, 2)

    lcs = _lcs_length(pred_tokens, ref_tokens)
    lcs_precision = lcs / len(pred_tokens)
    lcs_recall = lcs / len(ref_tokens)
    rouge_l = _f1(lcs_precision, lcs_recall)

    return {"rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_l": rouge_l}


# ---------------------------------------------------------------------------
# Retrieval metrics (precision@k, recall@k, MRR, NDCG)
# ---------------------------------------------------------------------------


def retrieval_metrics(sample: ScorerInput) -> dict:
    """Compute retrieval quality metrics from metadata lists.

    Expects ``sample.metadata`` to contain:

    * ``retrieved`` -- ordered list of retrieved item identifiers.
    * ``relevant``  -- list (or set) of relevant item identifiers.
    * ``k`` (optional) -- cut-off depth; defaults to ``len(retrieved)``.

    Returns:
        dict with keys ``precision_at_k``, ``recall_at_k``, ``mrr``, ``ndcg``.
    """
    zero_result: dict[str, float] = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }

    retrieved: list = sample.metadata.get("retrieved", [])  # type: ignore[union-attr]
    relevant: list = sample.metadata.get("relevant", [])  # type: ignore[union-attr]

    if not retrieved or not relevant:
        return zero_result

    relevant_set = set(relevant)
    k: int = sample.metadata.get("k", len(retrieved))  # type: ignore[union-attr]
    top_k = retrieved[:k]

    # Precision@k & Recall@k
    hits = sum(1 for item in top_k if item in relevant_set)
    precision_at_k = hits / k if k > 0 else 0.0
    recall_at_k = hits / len(relevant_set) if relevant_set else 0.0

    # Mean Reciprocal Rank (first relevant item in the full retrieved list)
    mrr = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            mrr = 1.0 / rank
            break

    # NDCG (Normalized Discounted Cumulative Gain) over top_k
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0.0 else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
    }


BUILTIN_SCORERS = {
    "contains": contains,
    "exact_match": exact_match,
    "f1_token": f1_token,
    "regex_match": regex_match,
    "bleu": bleu,
    "rouge": rouge,
    "retrieval_metrics": retrieval_metrics,
}
