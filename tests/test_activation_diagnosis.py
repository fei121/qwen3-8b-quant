import math
import unittest

from scripts.vllm_llmcompressor.activation_diagnosis import (
    PairAccumulator,
    choose_suspicious_layers,
    percentile,
)


class PercentileTests(unittest.TestCase):
    def test_interpolates_percentile_values(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 100), 4.0)
        self.assertAlmostEqual(percentile(values, 50), 2.5)


class PairAccumulatorTests(unittest.TestCase):
    def test_identical_values_have_zero_error_and_cosine_one(self):
        acc = PairAccumulator("layer_0")
        acc.update_lists([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        result = acc.finalize()

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertTrue(math.isinf(result["sqnr_db"]))

    def test_shifted_values_report_error_distribution(self):
        acc = PairAccumulator("layer_0")
        acc.update_lists([1.0, 2.0, 3.0], [2.0, 2.0, 4.0])

        result = acc.finalize()

        self.assertAlmostEqual(result["mse"], 2.0 / 3.0)
        self.assertAlmostEqual(result["abs_error_p50"], 1.0)
        self.assertGreater(result["abs_error_p99"], 0.0)
        self.assertLess(result["cosine"], 1.0)


class LayerRankingTests(unittest.TestCase):
    def test_prefers_low_cosine_then_low_sqnr_then_high_error(self):
        rows = [
            {"layer": 0, "cosine": 0.999, "sqnr_db": 40.0, "abs_error_p99": 0.1},
            {"layer": 1, "cosine": 0.930, "sqnr_db": 18.0, "abs_error_p99": 0.3},
            {"layer": 2, "cosine": 0.970, "sqnr_db": 10.0, "abs_error_p99": 1.0},
        ]

        ranked = choose_suspicious_layers(rows, limit=2)

        self.assertEqual(ranked, [1, 2])


if __name__ == "__main__":
    unittest.main()
