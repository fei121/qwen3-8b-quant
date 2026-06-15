import json
from pathlib import Path


GSM8K_SCORE_KEYS = (
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
    "acc",
)

LM_EVAL_SCORE_KEYS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
    "acc",
)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def extract_gsm8k_score(result):
    if not result:
        return None
    gsm8k = result.get("results", {}).get("gsm8k", {})
    for key in GSM8K_SCORE_KEYS:
        score = gsm8k.get(key)
        if score is not None:
            return score
    return None


def _extract_score_from_metrics(metrics):
    for key in LM_EVAL_SCORE_KEYS:
        score = metrics.get(key)
        if score is not None:
            return score
    return None


def extract_lm_eval_score(result, task_name):
    if not result:
        return None
    results = result.get("results", {})
    direct_score = _extract_score_from_metrics(results.get(task_name, {}))
    if direct_score is not None:
        return direct_score

    prefix = f"{task_name}_"
    scores = []
    for name, metrics in results.items():
        if name.startswith(prefix):
            score = _extract_score_from_metrics(metrics)
            if score is not None:
                scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _percentile_map(values):
    if isinstance(values, dict):
        return {int(float(key)): value for key, value in values.items()}
    return {int(float(percentile)): value for percentile, value in values or []}


def normalize_serve_metrics(metrics):
    if not metrics:
        return None
    ttft = _percentile_map(metrics.get("percentiles_ttft_ms"))
    tpot = _percentile_map(metrics.get("percentiles_tpot_ms"))
    e2e = _percentile_map(
        metrics.get("percentiles_e2el_ms") or metrics.get("percentiles_e2e_ms")
    )

    return {
        "request_throughput": metrics.get("request_throughput"),
        "total_tokens_per_s": metrics.get("total_token_throughput")
        or metrics.get("total_tokens_per_s"),
        "output_tokens_per_s": metrics.get("output_throughput")
        or metrics.get("output_tokens_per_s"),
        "mean_ttft_ms": metrics.get("mean_ttft_ms"),
        "p50_ttft_ms": metrics.get("p50_ttft_ms") or ttft.get(50),
        "p95_ttft_ms": metrics.get("p95_ttft_ms") or ttft.get(95),
        "p99_ttft_ms": metrics.get("p99_ttft_ms") or ttft.get(99),
        "mean_tpot_ms": metrics.get("mean_tpot_ms"),
        "p50_tpot_ms": metrics.get("p50_tpot_ms") or tpot.get(50),
        "p95_tpot_ms": metrics.get("p95_tpot_ms") or tpot.get(95),
        "p99_tpot_ms": metrics.get("p99_tpot_ms") or tpot.get(99),
        "p50_e2e_latency_ms": metrics.get("p50_e2el_ms")
        or metrics.get("p50_e2e_ms")
        or e2e.get(50),
        "p95_e2e_latency_ms": metrics.get("p95_e2el_ms")
        or metrics.get("p95_e2e_ms")
        or e2e.get(95),
        "p99_e2e_latency_ms": metrics.get("p99_e2el_ms")
        or metrics.get("p99_e2e_ms")
        or e2e.get(99),
    }
