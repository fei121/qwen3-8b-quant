import unittest

from scripts.common import extract_gsm8k_score, extract_lm_eval_score, normalize_serve_metrics


class LmEvalParsingTests(unittest.TestCase):
    def test_prefers_flexible_extract_for_gsm8k(self):
        result = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.61,
                    "exact_match,flexible-extract": 0.72,
                }
            }
        }

        self.assertEqual(extract_gsm8k_score(result), 0.72)

    def test_extracts_named_lm_eval_task_score(self):
        result = {"results": {"ceval-valid": {"acc,none": 0.6812}}}

        self.assertEqual(extract_lm_eval_score(result, "ceval-valid"), 0.6812)

    def test_averages_matching_subtask_scores_when_group_is_absent(self):
        result = {
            "results": {
                "ceval-valid_accountant": {"acc,none": 0.5},
                "ceval-valid_law": {"acc": 0.75},
                "gsm8k": {"exact_match,flexible-extract": 0.2},
            }
        }

        self.assertEqual(extract_lm_eval_score(result, "ceval-valid"), 0.625)


class ServeMetricParsingTests(unittest.TestCase):
    def test_normalizes_vllm_serve_metrics(self):
        metrics = {
            "request_throughput": 3.5,
            "total_token_throughput": 201.0,
            "output_throughput": 88.0,
            "mean_ttft_ms": 41.0,
            "mean_tpot_ms": 6.0,
            "percentiles_ttft_ms": [(50, 35.0), (95, 62.0), (99, 81.0)],
            "percentiles_tpot_ms": [(50, 5.0), (95, 9.0), (99, 12.0)],
            "percentiles_e2el_ms": [(50, 820.0), (95, 1200.0), (99, 1400.0)],
        }

        normalized = normalize_serve_metrics(metrics)

        self.assertEqual(normalized["request_throughput"], 3.5)
        self.assertEqual(normalized["total_tokens_per_s"], 201.0)
        self.assertEqual(normalized["output_tokens_per_s"], 88.0)
        self.assertEqual(normalized["p95_ttft_ms"], 62.0)
        self.assertEqual(normalized["p99_tpot_ms"], 12.0)
        self.assertEqual(normalized["p50_e2e_latency_ms"], 820.0)


if __name__ == "__main__":
    unittest.main()
