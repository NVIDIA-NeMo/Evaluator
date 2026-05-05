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
from typing import TYPE_CHECKING, Optional

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
        return {
            "correct": bool(re.search(sample.target, sample.response, re.IGNORECASE))
        }
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
    bp = (
        math.exp(1 - len(ref_tokens) / len(pred_tokens))
        if len(pred_tokens) < len(ref_tokens)
        else 1.0
    )

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


# ---------------------------------------------------------------------------
# Multiple-choice loglikelihood scorer (lm-evaluation-harness parity)
# ---------------------------------------------------------------------------


def multiple_choice_acc(sample: ScorerInput) -> dict:
    """Score multiple-choice loglikelihood ranking with lm-eval-harness metrics.

    Expects ``MultipleChoiceStrategy`` to have populated
    ``sample.metadata['_choices']`` and ``sample.metadata['_choices_logprobs']``
    (and optionally ``_choices_is_greedy``). Returns:

    * ``acc``: 1.0 if argmax of raw loglikelihoods matches gold, else 0.0
      (the canonical MMLU metric).
    * ``acc_norm``: 1.0 if argmax of length-normalized loglikelihoods
      matches gold, else 0.0. Normalization divides each loglikelihood by
      ``max(len(choice.encode("utf-8")), 1)`` — this is the exact
      per-byte normalization used by lm-eval-harness for ARC and BoolQ.
    * ``acc_greedy``: 1.0 if the greedy choice (the one whose continuation
      tokens were all top-1 predictions) is also gold, else 0.0. Useful
      for diagnostic comparison against lm-eval.

    Resolves gold to an integer index by accepting any of:

    * an integer index already (``sample.target == 1``).
    * a letter ``"A"``..``"Z"`` (offset from ``"A"``).
    * a literal choice string that appears verbatim in the choice list.
    * a stringified integer (``"0"``, ``"1"``, ...).

    Returns ``{"acc": 0.0, "acc_norm": 0.0, "acc_greedy": 0.0}`` if no
    choices were supplied (e.g. user wired this scorer to a benchmark
    that didn't go through MultipleChoiceStrategy).
    """
    metadata = sample.metadata or {}
    choices = metadata.get("_choices") or []
    logprobs = metadata.get("_choices_logprobs") or []
    is_greedy = metadata.get("_choices_is_greedy") or []

    zero = {"acc": 0.0, "acc_norm": 0.0, "acc_greedy": 0.0}
    if not choices or not logprobs or len(choices) != len(logprobs):
        return zero

    gold_idx = _resolve_gold_index(sample.target, choices)
    if gold_idx is None:
        return zero

    raw_argmax = max(range(len(logprobs)), key=lambda i: logprobs[i])

    norm_scores = [
        logprobs[i] / max(len(choices[i].encode("utf-8")), 1)
        for i in range(len(choices))
    ]
    norm_argmax = max(range(len(norm_scores)), key=lambda i: norm_scores[i])

    greedy_argmax: Optional[int] = None
    if is_greedy and any(is_greedy):
        # Pick highest-loglikelihood choice that is also greedy
        greedy_indices = [i for i, g in enumerate(is_greedy) if g]
        if greedy_indices:
            greedy_argmax = max(greedy_indices, key=lambda i: logprobs[i])

    return {
        "acc": 1.0 if raw_argmax == gold_idx else 0.0,
        "acc_norm": 1.0 if norm_argmax == gold_idx else 0.0,
        "acc_greedy": (
            1.0 if greedy_argmax is not None and greedy_argmax == gold_idx else 0.0
        ),
    }


def _resolve_gold_index(target, choices: list) -> "int | None":
    """Map a heterogeneous ``target`` value to an index into ``choices``."""
    if target is None:
        return None
    if isinstance(target, bool):
        return int(target) if int(target) < len(choices) else None
    if isinstance(target, int):
        return target if 0 <= target < len(choices) else None
    if isinstance(target, str):
        s = target.strip()
        if not s:
            return None
        # Letter (A-Z)
        if len(s) == 1 and s.upper().isalpha():
            idx = ord(s.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return idx
        # Stringified int
        if s.lstrip("-").isdigit():
            try:
                idx = int(s)
                if 0 <= idx < len(choices):
                    return idx
            except ValueError:
                pass
        # Verbatim choice text
        for i, c in enumerate(choices):
            if c.strip() == s:
                return i
        # Case-insensitive verbatim
        s_low = s.lower()
        for i, c in enumerate(choices):
            if c.strip().lower() == s_low:
                return i
    return None


# ---------------------------------------------------------------------------
# MCQ letter extraction (text-mode) — hoisted from example benchmarks
# ---------------------------------------------------------------------------


_MCQ_LETTER_PATTERNS = (
    re.compile(r"\\boxed\{\s*([A-Ja-j])\s*\}"),
    re.compile(r"(?:answer\s+is\s*[:\-]?\s*\(?)\s*([A-Ja-j])\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([A-Ja-j])\)", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-Ja-j])[\)\.\:]", re.IGNORECASE),
    re.compile(r"\b([A-Ja-j])\b"),
)


def _extract_letter(response: str | None) -> str:
    """Best-effort extract a single A-J letter from a free-form response.

    Tries the regex patterns in declared precedence (LaTeX boxed, "answer
    is X", "Option (X)", leading "X)/X./X:", then a final word-boundary
    fallback).
    """
    text = (response or "").strip()
    if not text:
        return ""
    for pat in _MCQ_LETTER_PATTERNS:
        m = pat.search(text[:200])
        if m:
            return m.group(1).upper()
    return ""


# Letter-coded choices for common MCQ datasets. Some newer MMLU-Pro style
# datasets expose up to ten options (A-J).
_INT_TO_LETTER = {i: chr(ord("A") + i) for i in range(10)}


def mcq_letter_extract(sample: ScorerInput) -> dict:
    """Extract A-J from response and compare against target.

    Handles common response formats (``"A"``, ``"A)"``, ``"The answer is B"``,
    ``"B. Because..."``, ``"(C)"``, ``"Option D"``, ``"\\boxed{E}"``).
    Targets may be:

    * a letter (``"A"``..``"J"``).
    * an integer index (``0``..``9``) or its string form.
    * a verbatim choice text — matched against ``sample.metadata`` keys
      ``a``/``b``/``c``/``d`` if available.

    Returns ``{"correct": bool, "parsed": bool}``.
    """
    predicted = _extract_letter(sample.response)

    raw = sample.target
    target_letter = ""
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, int):
        target_letter = _INT_TO_LETTER.get(raw, "")
    elif isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            target_letter = _INT_TO_LETTER.get(int(s), s.upper())
        elif s and s[0].upper() in "ABCDEFGHIJ" and len(s) <= 2:
            target_letter = s[0].upper()
        else:
            # Try matching against choice text in metadata
            for letter, key in zip("ABCD", ("a", "b", "c", "d")):
                cand = sample.metadata.get(key)
                if isinstance(cand, str) and cand.strip().lower() == s.lower():
                    target_letter = letter
                    break

    return {
        "correct": bool(predicted) and predicted == target_letter,
        "parsed": bool(predicted),
    }


# ---------------------------------------------------------------------------
# GSM8K numeric-answer extraction (#### marker)
# ---------------------------------------------------------------------------


_GSM8K_HASH_PATTERN = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED_PATTERN = re.compile(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}")


def _normalize_number(s: str) -> "str | None":
    """Strip commas, trim trailing zeros after decimal, return canonical form."""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if not _NUMBER_PATTERN.fullmatch(s):
        return None
    if "." in s:
        # 12.300 -> 12.3, 12.000 -> 12
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def gsm8k_answer(sample: ScorerInput) -> dict:
    """Extract the canonical GSM8K numeric answer and compare to target.

    Extraction precedence:

    1. ``#### <number>`` marker (the canonical GSM8K answer format).
    2. ``\\boxed{<number>}`` (LaTeX-style answer markers, common in
       chain-of-thought prompts).
    3. The last number in the response (fallback for free-form output).

    The target string is parsed the same way (so gold answers stored as
    full GSM8K solutions like ``"... #### 42"`` work directly).

    Returns ``{"correct": bool, "parsed": bool}`` after stripping commas
    and normalizing trailing zeros.
    """
    response = sample.response or ""

    predicted_raw = None
    m = _GSM8K_HASH_PATTERN.search(response)
    if m:
        predicted_raw = m.group(1)
    else:
        m = _BOXED_PATTERN.search(response)
        if m:
            predicted_raw = m.group(1)
        else:
            matches = _NUMBER_PATTERN.findall(response)
            if matches:
                predicted_raw = matches[-1]

    predicted = _normalize_number(predicted_raw) if predicted_raw else None

    target_raw = sample.target
    if isinstance(target_raw, (int, float)):
        gold = _normalize_number(str(target_raw))
    else:
        target_str = str(target_raw or "")
        m = _GSM8K_HASH_PATTERN.search(target_str)
        if m:
            gold = _normalize_number(m.group(1))
        else:
            matches = _NUMBER_PATTERN.findall(target_str)
            gold = _normalize_number(matches[-1]) if matches else None

    return {
        "correct": bool(
            predicted is not None and gold is not None and predicted == gold
        ),
        "parsed": predicted is not None,
    }


# ---------------------------------------------------------------------------
# Boolean yes/no extraction
# ---------------------------------------------------------------------------


_YES_TOKENS = {"yes", "true", "correct", "right", "yeah", "yep"}
_NO_TOKENS = {"no", "false", "incorrect", "wrong", "nope", "nah"}
_TOKEN_RE = re.compile(r"[A-Za-z]+", re.UNICODE)


def boolean_yesno(sample: ScorerInput) -> dict:
    """Extract a yes/no decision from the response and compare to target.

    Heuristic: scans the first ~20 alphabetic tokens of the response and
    picks the first match in ``_YES_TOKENS`` / ``_NO_TOKENS`` (case
    insensitive). Targets may be booleans, ``"yes"``/``"no"`` strings,
    or stringified bools.

    Returns ``{"correct": bool, "parsed": bool}``.
    """
    response = (sample.response or "").lower()
    tokens = _TOKEN_RE.findall(response)[:20]
    predicted: "str | None" = None
    for tok in tokens:
        tl = tok.lower()
        if tl in _YES_TOKENS:
            predicted = "yes"
            break
        if tl in _NO_TOKENS:
            predicted = "no"
            break

    raw = sample.target
    gold: "str | None"
    if isinstance(raw, bool):
        gold = "yes" if raw else "no"
    elif isinstance(raw, (int, float)):
        gold = "yes" if int(raw) == 1 else "no" if int(raw) == 0 else None
    else:
        s = str(raw).strip().lower()
        if s in _YES_TOKENS or s in {"1", "true"}:
            gold = "yes"
        elif s in _NO_TOKENS or s in {"0", "false"}:
            gold = "no"
        else:
            gold = None

    return {
        "correct": bool(
            predicted is not None and gold is not None and predicted == gold
        ),
        "parsed": predicted is not None,
    }


# ---------------------------------------------------------------------------
# chrF / chrF++ (sacreBLEU-style character n-gram F-score)
# ---------------------------------------------------------------------------


def _char_ngrams(text: str, n: int) -> Counter:
    if not text or len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _word_ngrams(tokens: list, n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _chrf_score(
    hyp: str, ref: str, max_char_n: int = 6, max_word_n: int = 0, beta: float = 2.0
) -> float:
    """Compute chrF or chrF++ score in [0, 100].

    Mirrors the sacreBLEU formula: arithmetic mean of per-n F-beta scores
    over character n-grams (and optionally word n-grams when
    ``max_word_n > 0``, which gives chrF++).
    """
    if not hyp or not ref:
        return 0.0

    f_scores: list = []

    for n in range(1, max_char_n + 1):
        h_ng = _char_ngrams(hyp, n)
        r_ng = _char_ngrams(ref, n)
        if not h_ng or not r_ng:
            f_scores.append(0.0)
            continue
        overlap = sum((h_ng & r_ng).values())
        h_total = sum(h_ng.values())
        r_total = sum(r_ng.values())
        prec = overlap / h_total if h_total else 0.0
        rec = overlap / r_total if r_total else 0.0
        if prec + rec == 0:
            f_scores.append(0.0)
        else:
            f_scores.append((1 + beta**2) * prec * rec / (beta**2 * prec + rec))

    if max_word_n > 0:
        h_tok = hyp.split()
        r_tok = ref.split()
        for n in range(1, max_word_n + 1):
            h_ng = _word_ngrams(h_tok, n)
            r_ng = _word_ngrams(r_tok, n)
            if not h_ng or not r_ng:
                f_scores.append(0.0)
                continue
            overlap = sum((h_ng & r_ng).values())
            h_total = sum(h_ng.values())
            r_total = sum(r_ng.values())
            prec = overlap / h_total if h_total else 0.0
            rec = overlap / r_total if r_total else 0.0
            if prec + rec == 0:
                f_scores.append(0.0)
            else:
                f_scores.append((1 + beta**2) * prec * rec / (beta**2 * prec + rec))

    return 100.0 * sum(f_scores) / len(f_scores) if f_scores else 0.0


def chrf(sample: ScorerInput) -> dict:
    """Compute sentence-level chrF and chrF++ scores (sacreBLEU style).

    Returns ``{"chrf": float, "chrf_pp": float}`` in the [0, 100] range.

    * ``chrf``: chrF6 — character 1- to 6-gram F2 (the sacreBLEU default).
    * ``chrf_pp``: chrF++ — chrF6 plus word 1- and 2-gram F2, averaged.
    """
    hyp = sample.response or ""
    target = _to_str(sample.target)

    return {
        "chrf": _chrf_score(hyp, target, max_char_n=6, max_word_n=0, beta=2.0),
        "chrf_pp": _chrf_score(hyp, target, max_char_n=6, max_word_n=2, beta=2.0),
    }


BUILTIN_SCORERS = {
    "contains": contains,
    "exact_match": exact_match,
    "f1_token": f1_token,
    "regex_match": regex_match,
    "bleu": bleu,
    "rouge": rouge,
    "retrieval_metrics": retrieval_metrics,
    "multiple_choice_acc": multiple_choice_acc,
    "mcq_letter_extract": mcq_letter_extract,
    "gsm8k_answer": gsm8k_answer,
    "boolean_yesno": boolean_yesno,
    "chrf": chrf,
}
