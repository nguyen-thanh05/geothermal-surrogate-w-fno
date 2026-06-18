import unittest

import numpy as np

from evaluation.run_eval import (
    ENERGY_LOG_DENOM,
    HOMO_CONSTANTS,
    bootstrap_weights,
    denormalize_aux_predictions,
    split_raw_aux_targets,
    summarize_matrix,
)


class RunEvalMathTests(unittest.TestCase):
    def test_mean_bootstrap_constant_values(self):
        matrix = np.full((5, 3), 2.5)
        weights = bootstrap_weights(n=5, resamples=100, seed=123)
        mean, low, high = summarize_matrix(matrix, "mean", weights)

        np.testing.assert_allclose(mean, [2.5, 2.5, 2.5])
        np.testing.assert_allclose(low, [2.5, 2.5, 2.5])
        np.testing.assert_allclose(high, [2.5, 2.5, 2.5])

    def test_rmse_summary_uses_pooled_mse(self):
        sqerr = np.array([[1.0], [4.0], [9.0]])
        weights = bootstrap_weights(n=3, resamples=0, seed=123)
        mean, low, high = summarize_matrix(sqerr, "rmse", weights)

        expected = np.sqrt((1.0 + 4.0 + 9.0) / 3.0)
        self.assertAlmostEqual(float(mean[0]), expected)
        self.assertAlmostEqual(float(low[0]), expected)
        self.assertAlmostEqual(float(high[0]), expected)

    def test_aux_prediction_and_target_orders_are_different(self):
        bhp = np.array([[1900.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0]])
        energy = np.array([[1.0, 10.0, 100.0, 1000.0, 1.0e4, 1.0e5, 1.0e6]])
        pred_aux = np.concatenate(
            [
                np.log1p(energy) / ENERGY_LOG_DENOM,
                (bhp - HOMO_CONSTANTS.pres_min) / HOMO_CONSTANTS.pres_range,
            ],
            axis=1,
        )
        raw_aux = np.concatenate([bhp, energy], axis=1)

        pred_bhp, pred_energy = denormalize_aux_predictions(pred_aux, HOMO_CONSTANTS)
        gt_bhp, gt_energy = split_raw_aux_targets(raw_aux)

        np.testing.assert_allclose(pred_bhp, gt_bhp, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(pred_energy, gt_energy, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
