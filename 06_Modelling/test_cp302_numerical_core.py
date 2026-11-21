"""Small unit tests for cp302_numerical_core; no production runs are performed."""

import unittest

import numpy as np

from cp302_numerical_core import (
    InertialState,
    Parameters,
    UnresolvedBoundaryCrossing,
    apply_inertial_endpoint_boundary,
    calculate_repulsion,
    conditional_ou_bridge_midpoint,
    conditional_split_ou_innovation,
    initialize_inertial_state,
    initialize_particles,
    make_rng_streams,
    ou_transition,
    ou_transition_matrices,
    overdamped_step,
    project_overdamped_boundary,
    sample_ou_innovation,
    inertial_step,
    _psd_factor,
)


class TestOUTransitions(unittest.TestCase):
    def setUp(self):
        self.gamma = 1.0
        self.mass = 0.1
        self.kbt = 0.01
        self.h = 0.005

    def test_ou_transition_and_innovation_covariance(self):
        rng = np.random.default_rng(12345)
        n = 50000
        innovations = sample_ou_innovation(
            self.h, self.gamma, self.mass, self.kbt, rng, (n,)
        )
        samples = innovations.reshape(-1, 2)
        _, expected = ou_transition_matrices(
            self.h, self.gamma, self.mass, self.kbt
        )
        observed = np.cov(samples, rowvar=False, bias=True)
        np.testing.assert_allclose(observed, expected, rtol=0.025, atol=2e-9)
        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=7e-4)

    def test_conditional_bridge_midpoint_moments(self):
        rng_parent = np.random.default_rng(20)
        rng_bridge = np.random.default_rng(21)
        n = 40000
        x0 = np.zeros((n, 2))
        v0 = np.zeros((n, 2))
        g = np.zeros((n, 2))
        parent = sample_ou_innovation(
            self.h, self.gamma, self.mass, self.kbt, rng_parent, (n,)
        )
        xh, vh = ou_transition(
            x0, v0, g, self.h, self.gamma, self.mass, parent
        )
        xm, vm, conditional_cov = conditional_ou_bridge_midpoint(
            x0, v0, xh, vh, g, self.h, self.gamma, self.mass,
            self.kbt, rng_bridge,
        )
        _, expected_mid = ou_transition_matrices(
            self.h / 2.0, self.gamma, self.mass, self.kbt
        )
        observed = np.cov(np.stack((xm[:, 0], vm[:, 0]), axis=1), rowvar=False, bias=True)
        np.testing.assert_allclose(observed, expected_mid, rtol=0.035, atol=2e-9)
        self.assertTrue(np.all(np.linalg.eigvalsh(conditional_cov) >= -1e-10))

    def test_parent_child_innovations_compose_exactly(self):
        rng = np.random.default_rng(32)
        n = 500
        parent = sample_ou_innovation(
            self.h, self.gamma, self.mass, self.kbt, rng, (n, 2)
        )
        left, right = conditional_split_ou_innovation(
            parent, self.h, self.h * 0.5, self.gamma, self.mass,
            self.kbt, rng,
        )
        a_right, _ = ou_transition_matrices(
            self.h * 0.5, self.gamma, self.mass, self.kbt
        )
        reconstructed = left @ a_right.T + right
        np.testing.assert_allclose(reconstructed, parent, rtol=0.0, atol=2e-15)

    def test_conditional_split_marginal_and_conditional_covariance(self):
        rng_parent = np.random.default_rng(40)
        rng_split = np.random.default_rng(41)
        n = 45000
        h = 0.02
        s = 0.37 * h  # deterministic non-dyadic split for the bridge utility
        parent = sample_ou_innovation(
            h, self.gamma, self.mass, self.kbt, rng_parent, (n,)
        )
        first, second = conditional_split_ou_innovation(
            parent, h, s, self.gamma, self.mass, self.kbt, rng_split
        )
        _, q_first = ou_transition_matrices(s, self.gamma, self.mass, self.kbt)
        _, q_second = ou_transition_matrices(h - s, self.gamma, self.mass, self.kbt)
        np.testing.assert_allclose(
            np.cov(first[:, 0, :], rowvar=False, bias=True), q_first,
            rtol=0.03, atol=2e-8,
        )
        np.testing.assert_allclose(
            np.cov(second[:, 0, :], rowvar=False, bias=True), q_second,
            rtol=0.03, atol=2e-8,
        )

        # For a fixed parent innovation, verify the conditional first-child law.
        fixed_parent = np.array([[2e-4, -1e-3]])
        repeated_parent = np.repeat(fixed_parent, n, axis=0)
        first_c, _ = conditional_split_ou_innovation(
            repeated_parent, h, s, self.gamma, self.mass,
            self.kbt, np.random.default_rng(42),
        )
        a_r, _ = ou_transition_matrices(h - s, self.gamma, self.mass, self.kbt)
        _, q_h = ou_transition_matrices(h, self.gamma, self.mass, self.kbt)
        gain = q_first @ a_r.T @ np.linalg.pinv(q_h, rcond=1e-14)
        expected_mean = gain @ fixed_parent[0]
        expected_cov = q_first - gain @ q_h @ gain.T
        np.testing.assert_allclose(first_c.mean(axis=0), expected_mean, atol=1e-4)
        np.testing.assert_allclose(
            np.cov(first_c, rowvar=False, bias=True),
            expected_cov, rtol=0.035, atol=2e-8,
        )

    def test_constant_force_deterministic_solution(self):
        x0 = np.array([[0.2, -0.1]])
        v0 = np.array([[0.4, -0.3]])
        g = np.array([[1.1, 0.7]])
        h = 0.07
        tau = self.mass / self.gamma
        a = np.exp(-h / tau)
        x, v = ou_transition(
            x0, v0, g, h, self.gamma, self.mass,
            np.zeros((1, 2, 2)),
        )
        expected_v = g + a * (v0 - g)
        expected_x = x0 + g * h + tau * (1.0 - a) * (v0 - g)
        np.testing.assert_allclose(v, expected_v, atol=1e-14)
        np.testing.assert_allclose(x, expected_x, atol=1e-14)

    def test_small_timestep_covariance_is_stable(self):
        for ratio in (1e-2, 1e-5, 1e-8, 1e-11):
            h = ratio * self.mass / self.gamma
            _, covariance = ou_transition_matrices(
                h, self.gamma, self.mass, self.kbt
            )
            self.assertTrue(np.all(np.isfinite(covariance)))
            self.assertGreaterEqual(covariance[0, 0], 0.0)
            self.assertGreaterEqual(covariance[1, 1], 0.0)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], -1e-28)
            expected_leading = (2.0 / 3.0) * (self.kbt / self.mass) * h**3 / (self.mass / self.gamma)
            if ratio <= 1e-5:
                self.assertAlmostEqual(
                    covariance[0, 0] / expected_leading, 1.0, delta=2e-5
                )

    def test_psd_factor_accepts_small_valid_ou_covariance(self):
        _, covariance = ou_transition_matrices(
            0.005, self.gamma, self.mass, 1e-20
        )
        factor = _psd_factor(covariance)
        self.assertTrue(np.all(np.isfinite(factor)))
        scale = float(np.max(np.abs(covariance)))
        np.testing.assert_allclose(
            (factor @ factor.T) / scale,
            covariance / scale,
            rtol=2e-12,
            atol=2e-14,
        )

    def test_psd_factor_rejects_small_scale_indefinite_covariance(self):
        covariance = 1e-20 * np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(FloatingPointError):
            _psd_factor(covariance)

    def test_psd_factor_accepts_exact_zero_covariance(self):
        factor = _psd_factor(np.zeros((2, 2)))
        self.assertEqual(factor.shape, (2, 2))
        np.testing.assert_array_equal(factor, np.zeros((2, 2)))

    def test_psd_factor_handles_large_finite_diagonal_covariance(self):
        covariance = 1e308 * np.eye(2)
        factor = _psd_factor(covariance)
        reconstructed = factor @ factor.T
        self.assertTrue(np.all(np.isfinite(factor)))
        self.assertTrue(np.all(np.isfinite(reconstructed)))
        np.testing.assert_allclose(reconstructed, covariance, rtol=5e-15, atol=0.0)

    def test_psd_factor_reconstructs_symmetrized_nonsymmetric_input(self):
        covariance = np.array([[4.0, 2.0], [0.0, 3.0]])
        expected_symmetric = 0.5 * (covariance + covariance.T)
        factor = _psd_factor(covariance)
        np.testing.assert_allclose(
            factor @ factor.T, expected_symmetric, rtol=1e-14, atol=1e-14
        )

    def test_bridge_midpoint_moments_for_fixed_parent_endpoints(self):
        n = 30000
        h = 0.005
        rng_parent = np.random.default_rng(111)
        rng_bridge = np.random.default_rng(112)
        parent = sample_ou_innovation(
            h, self.gamma, self.mass, self.kbt, rng_parent, (1,)
        )
        x0 = np.zeros((n, 2))
        v0 = np.zeros((n, 2))
        g = np.zeros((n, 2))
        fixed_parent = np.broadcast_to(parent, (n,) + parent.shape[1:]).copy()
        xh, vh = ou_transition(
            x0, v0, g, h, self.gamma, self.mass, fixed_parent
        )
        xm, vm, observed_conditional_cov = conditional_ou_bridge_midpoint(
            x0, v0, xh, vh, g, h, self.gamma, self.mass,
            self.kbt, rng_bridge,
        )

        _, q_half = ou_transition_matrices(
            h / 2.0, self.gamma, self.mass, self.kbt
        )
        a_remainder, _ = ou_transition_matrices(
            h / 2.0, self.gamma, self.mass, self.kbt
        )
        _, q_parent = ou_transition_matrices(
            h, self.gamma, self.mass, self.kbt
        )
        gain = q_half @ a_remainder.T @ np.linalg.pinv(q_parent, rcond=1e-14)
        expected_cov = q_half - gain @ q_parent @ gain.T
        expected_mean = gain @ parent[0, 0]
        samples = np.column_stack((xm[:, 0], vm[:, 0]))
        mean_standard_error = np.sqrt(np.diag(expected_cov) / n)
        self.assertTrue(
            np.all(np.abs(samples.mean(axis=0) - expected_mean)
                   <= 5.0 * mean_standard_error + 1e-12)
        )
        observed_cov = np.cov(samples, rowvar=False, bias=True)
        covariance_standard_error = np.sqrt(
            (expected_cov * expected_cov
             + np.outer(np.diag(expected_cov), np.diag(expected_cov))) / n
        )
        self.assertTrue(
            np.all(np.abs(observed_cov - expected_cov)
                   <= 5.0 * covariance_standard_error + 1e-12)
        )
        np.testing.assert_allclose(observed_conditional_cov, expected_cov, rtol=1e-12, atol=1e-15)


class TestModelSteps(unittest.TestCase):
    def test_rotational_diffusion_and_rng_stream_reproducibility(self):
        n = 60000
        dr = 0.1
        h = 0.005
        streams = make_rng_streams(81)
        increment = np.sqrt(2.0 * dr * h) * streams.rotation.normal(size=n)
        self.assertAlmostEqual(float(np.var(increment)), 2.0 * dr * h, delta=2.5e-5)
        self.assertAlmostEqual(float(np.mean(np.cos(increment))), np.exp(-dr * h), delta=2e-4)

        init1 = initialize_particles(50, 5.0, 81)
        init2 = initialize_particles(50, 5.0, 81)
        np.testing.assert_array_equal(init1.positions, init2.positions)
        np.testing.assert_array_equal(init1.orientations, init2.orientations)
        init3 = initialize_particles(50, 5.0, 82)
        self.assertFalse(np.array_equal(init1.positions, init3.positions))
        inertial = initialize_inertial_state(init1, 1.0)
        np.testing.assert_array_equal(inertial.positions, init1.positions)
        np.testing.assert_array_equal(inertial.orientations, init1.orientations)

    def test_overdamped_boundary_projection(self):
        proposed = np.array([[0.0, 0.0], [3.0, 4.0], [0.3, 0.4]])
        projected, count = project_overdamped_boundary(proposed, 2.0)
        self.assertEqual(count, 1)
        np.testing.assert_allclose(projected[1], [1.2, 1.6])
        np.testing.assert_array_equal(projected[[0, 2]], proposed[[0, 2]])

    def test_inertial_endpoint_projection_and_specular_reflection(self):
        x = np.array([[1.2, 0.0], [0.0, 0.5]])
        v = np.array([[3.0, 2.0], [0.0, 1.0]])
        x_out, v_out, count = apply_inertial_endpoint_boundary(x, v, 1.0)
        self.assertEqual(count, 1)
        np.testing.assert_allclose(x_out[0], [1.0, 0.0])
        np.testing.assert_allclose(v_out[0], [-3.0, 2.0])
        np.testing.assert_array_equal(x_out[1], x[1])
        np.testing.assert_array_equal(v_out[1], v[1])

    def test_wall_start_outward_inward_and_tangential_velocity(self):
        x = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        v = np.array([[2.0, 1.0], [-2.0, 1.0], [0.0, 2.0]])
        _, result, _ = apply_inertial_endpoint_boundary(x, v, 1.0)
        np.testing.assert_allclose(result[0], [-2.0, 1.0])
        np.testing.assert_array_equal(result[1], v[1])
        np.testing.assert_array_equal(result[2], v[2])

    def test_cross_and_return_refines_and_corrects_boundary(self):
        params = Parameters(
            n_particles=1,
            dt=0.2,
            dt_trans=0.0,
            dr=0.0,
            mass=0.1,
            v0=1.0,
            dish_radius=1.0,
            ordinary_leaf_width=0.000625,
            max_refinement_depth=12,
        )
        state = InertialState(
            positions=np.array([[0.99, 0.0]]),
            velocities=np.array([[1.0, 0.0]]),
            orientations=np.array([np.pi]),
        )
        rngs = make_rng_streams(5)
        final, diagnostics = inertial_step(
            state, params, rngs.inertial_translation, rngs.rotation,
            rngs.bridge,
        )
        self.assertGreater(diagnostics.refinements, 0)
        self.assertGreaterEqual(diagnostics.endpoint_corrections, 1)
        self.assertLessEqual(float(np.linalg.norm(final.positions[0])), 1.0 + 1e-12)

    def test_unresolved_cross_and_return_raises_at_hard_limit(self):
        params = Parameters(
            n_particles=1,
            dt=0.2,
            dt_trans=0.0,
            dr=0.0,
            mass=0.1,
            v0=1.0,
            dish_radius=1.0,
            ordinary_leaf_width=0.000625,
            max_refinement_depth=0,
        )
        state = InertialState(
            positions=np.array([[0.99, 0.0]]),
            velocities=np.array([[1.0, 0.0]]),
            orientations=np.array([np.pi]),
        )
        rngs = make_rng_streams(6)
        with self.assertRaises(UnresolvedBoundaryCrossing):
            inertial_step(
                state, params, rngs.inertial_translation, rngs.rotation,
                rngs.bridge,
            )

    def test_far_from_wall_interval_wider_than_target_is_not_refined(self):
        params = Parameters(
            n_particles=1, dt=0.005, dt_trans=0.0, dr=0.0,
            v0=0.0, dish_radius=1000.0,
            ordinary_leaf_width=0.000625,
        )
        state = InertialState(
            np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), np.array([0.0])
        )
        rngs = make_rng_streams(91)
        _, diagnostics = inertial_step(
            state, params, rngs.inertial_translation, rngs.rotation,
            rngs.bridge,
        )
        self.assertEqual(diagnostics.refinements, 0)
        self.assertEqual(diagnostics.maximum_depth, 0)

    def test_near_wall_interval_wider_than_target_is_refined(self):
        params = Parameters(
            n_particles=1, dt=0.005, dt_trans=0.0, dr=0.0,
            v0=0.0, dish_radius=1.0,
            ordinary_leaf_width=0.000625,
        )
        state = InertialState(
            np.array([[0.9999, 0.0]]), np.array([[-0.1, 0.0]]), np.array([np.pi])
        )
        rngs = make_rng_streams(92)
        _, diagnostics = inertial_step(
            state, params, rngs.inertial_translation, rngs.rotation,
            rngs.bridge,
        )
        self.assertGreater(diagnostics.refinements, 0)

    def test_endpoint_outside_at_target_width_is_corrected_without_refining(self):
        target = 0.000625
        params = Parameters(
            n_particles=1, dt=target, dt_trans=0.0, dr=0.0,
            v0=0.0, dish_radius=1.0,
            ordinary_leaf_width=target,
        )
        state = InertialState(
            np.array([[0.9999, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.0])
        )
        rngs = make_rng_streams(93)
        final, diagnostics = inertial_step(
            state, params, rngs.inertial_translation, rngs.rotation,
            rngs.bridge,
        )
        self.assertEqual(diagnostics.refinements, 0)
        self.assertEqual(diagnostics.endpoint_corrections, 1)
        self.assertAlmostEqual(float(np.linalg.norm(final.positions[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
