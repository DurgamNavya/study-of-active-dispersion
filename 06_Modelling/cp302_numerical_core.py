"""Numerical core for a paired CP302 overdamped/inertial ABP study.

This module deliberately has no file-writing or simulation-driver behavior.
The interacting inertial predictor-corrector is an approximation to the full
system; only its frozen-force, fixed-orientation OU subproblem is integrated
exactly. Boundary handling is a finite-resolution endpoint approximation,
not an exact continuous first-passage simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


Array = np.ndarray


class UnresolvedBoundaryCrossing(RuntimeError):
    """Raised when a cross-and-return indication survives all refinements."""


@dataclass(frozen=True)
class Parameters:
    """Dimensionless parameters copied from the CP302 baseline models."""

    n_particles: int = 100
    dt: float = 0.005
    dt_trans: float = 0.01
    dr: float = 0.1
    mobility: float = 1.0
    mass: float = 0.1
    v0: float = 1.0
    repulsion_k: float = 10.0
    particle_diameter: float = 1.0
    dish_radius: float = 40.0
    initial_cluster_radius: float = 5.0
    # Target width for ordinary near-wall refinement; not a global leaf cap.
    ordinary_leaf_width: float = 0.000625
    max_refinement_depth: int = 12
    boundary_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.n_particles < 1:
            raise ValueError("n_particles must be positive")
        if self.dt <= 0 or self.ordinary_leaf_width <= 0:
            raise ValueError("dt and ordinary_leaf_width must be positive")
        if self.dt_trans < 0 or self.dr < 0 or self.mass <= 0:
            raise ValueError("diffusivities must be nonnegative and mass positive")
        if self.mobility <= 0 or self.dish_radius <= 0:
            raise ValueError("mobility and dish_radius must be positive")
        if self.max_refinement_depth < 0:
            raise ValueError("max_refinement_depth must be nonnegative")

    @property
    def gamma(self) -> float:
        return 1.0 / self.mobility

    @property
    def tau_m(self) -> float:
        return self.mass / self.gamma

    @property
    def kbt(self) -> float:
        # Preserve the baseline Einstein relation D_T = kBT / gamma.
        return self.gamma * self.dt_trans


@dataclass
class RNGStreams:
    """Independent deterministic streams for each stochastic component."""

    initialization_positions: np.random.Generator
    initialization_orientations: np.random.Generator
    overdamped_translation: np.random.Generator
    inertial_translation: np.random.Generator
    rotation: np.random.Generator
    bridge: np.random.Generator


class InitialConditions(NamedTuple):
    positions: Array
    orientations: Array


class InertialState(NamedTuple):
    positions: Array
    velocities: Array
    orientations: Array


class StepDiagnostics(NamedTuple):
    endpoint_corrections: int
    refinements: int
    maximum_depth: int


def make_rng_streams(seed: int) -> RNGStreams:
    """Create stable, independent generators from one user-facing seed."""
    children = np.random.SeedSequence(int(seed)).spawn(6)
    generators = [np.random.default_rng(child) for child in children]
    return RNGStreams(*generators)


def initialize_particles(
    number: int,
    cluster_radius: float,
    seed: int,
) -> InitialConditions:
    """Generate positions and orientations with distinct reproducible streams."""
    if number < 1 or cluster_radius <= 0:
        raise ValueError("number and cluster_radius must be positive")
    streams = make_rng_streams(seed)
    angles = streams.initialization_positions.uniform(0.0, 2.0 * np.pi, number)
    radii = cluster_radius * np.sqrt(
        streams.initialization_positions.uniform(0.0, 1.0, number)
    )
    positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    orientations = streams.initialization_orientations.uniform(
        0.0, 2.0 * np.pi, number
    )
    return InitialConditions(positions, orientations)


def initialize_inertial_state(
    initial: InitialConditions,
    v0: float,
) -> InertialState:
    """Use the same initial arrays as overdamped ABP; set v(0)=v0*u(theta)."""
    direction = np.column_stack((np.cos(initial.orientations), np.sin(initial.orientations)))
    return InertialState(
        initial.positions.copy(),
        float(v0) * direction,
        initial.orientations.copy(),
    )


def calculate_repulsion(positions: Array, params: Parameters) -> Array:
    """Baseline harmonic pair repulsion, evaluated by a dense pair matrix."""
    positions = np.asarray(positions, dtype=float)
    displacement = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    safe = np.maximum(distances, np.finfo(float).tiny)
    overlap = np.maximum(params.particle_diameter - distances, 0.0)
    np.fill_diagonal(overlap, 0.0)
    pair_force = (
        params.repulsion_k
        * overlap[:, :, None]
        * displacement
        / safe[:, :, None]
    )
    return np.sum(pair_force, axis=1)


def project_overdamped_boundary(positions: Array, radius: float) -> tuple[Array, int]:
    """Project outside endpoints to the nearest point of the circular disk."""
    result = np.asarray(positions, dtype=float).copy()
    norms = np.linalg.norm(result, axis=1)
    outside = norms > radius
    if np.any(outside):
        result[outside] *= (radius / norms[outside])[:, None]
    return result, int(np.count_nonzero(outside))


def apply_inertial_endpoint_boundary(
    positions: Array,
    velocities: Array,
    radius: float,
    tolerance: float = 1e-12,
) -> tuple[Array, Array, int]:
    """Project outside endpoints and specularly reflect outward velocity."""
    x = np.asarray(positions, dtype=float).copy()
    v = np.asarray(velocities, dtype=float).copy()
    norms = np.linalg.norm(x, axis=1)
    outside = norms > radius
    if np.any(outside):
        normals = x[outside] / norms[outside, None]
        x[outside] = radius * normals
        outward_speed = np.sum(v[outside] * normals, axis=1)
        outward = outward_speed > 0.0
        local_v = v[outside]
        local_v[outward] -= 2.0 * outward_speed[outward, None] * normals[outward]
        v[outside] = local_v

    # If a state is on the wall and points outward, reflect at the start/event.
    norms = np.linalg.norm(x, axis=1)
    on_wall = np.abs(norms - radius) <= tolerance
    if np.any(on_wall):
        normals = x[on_wall] / np.maximum(norms[on_wall, None], np.finfo(float).tiny)
        outward_speed = np.sum(v[on_wall] * normals, axis=1)
        outward = outward_speed > 0.0
        local_v = v[on_wall]
        local_v[outward] -= 2.0 * outward_speed[outward, None] * normals[outward]
        v[on_wall] = local_v
    return x, v, int(np.count_nonzero(outside))


def overdamped_step(
    positions: Array,
    orientations: Array,
    params: Parameters,
    translation_rng: np.random.Generator,
    rotation_rng: np.random.Generator,
) -> tuple[Array, Array, int]:
    """One Euler–Maruyama ABP step with projected circular boundary."""
    x = np.asarray(positions, dtype=float)
    theta = np.asarray(orientations, dtype=float)
    force = calculate_repulsion(x, params)
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    noise = np.sqrt(2.0 * params.dt_trans * params.dt) * translation_rng.normal(
        size=x.shape
    )
    proposed = x + (params.v0 * direction + params.mobility * force) * params.dt + noise
    x_next, corrected = project_overdamped_boundary(proposed, params.dish_radius)
    theta_next = np.mod(
        theta + np.sqrt(2.0 * params.dr * params.dt) * rotation_rng.normal(size=theta.shape),
        2.0 * np.pi,
    )
    return x_next, theta_next, corrected


def _stable_dimensionless_position_variance(u: float) -> float:
    """Return 2u-3+4exp(-u)-exp(-2u), stably for small u."""
    if u < 1e-3:
        # Taylor series through u^10; avoids cancellation of O(u^3) terms.
        return (
            (2.0 / 3.0) * u**3
            - 0.5 * u**4
            + (7.0 / 30.0) * u**5
            - (1.0 / 12.0) * u**6
            + (31.0 / 1260.0) * u**7
            - (1.0 / 160.0) * u**8
            + (127.0 / 90720.0) * u**9
            - (17.0 / 60480.0) * u**10
        )
    a = np.exp(-u)
    return 2.0 * u - 3.0 + 4.0 * a - a * a


def ou_transition_matrices(
    duration: float,
    gamma: float,
    mass: float,
    kbt: float,
) -> tuple[Array, Array]:
    """Return the exact frozen-coefficient OU transition A and covariance Q."""
    if duration < 0 or gamma <= 0 or mass <= 0 or kbt < 0:
        raise ValueError("duration/kBT must be nonnegative; gamma/mass positive")
    tau = mass / gamma
    u = duration / tau
    one_minus_a = -np.expm1(-u)
    a = 1.0 - one_minus_a
    q = kbt / mass
    vv = q * (-np.expm1(-2.0 * u))
    cxv = q * tau * one_minus_a * one_minus_a
    vx = q * tau * tau * _stable_dimensionless_position_variance(u)
    # Remove only round-off level negatives from the stable series.
    vx = max(float(vx), 0.0)
    qmat = np.array([[vx, cxv], [cxv, vv]], dtype=float)
    amat = np.array([[1.0, tau * one_minus_a], [0.0, a]], dtype=float)
    return amat, qmat


def _psd_factor(covariance: Array) -> Array:
    """Return a PSD factor whose product with its transpose approximates covariance.

    The normalized eigenvalue tolerance is 100 machine epsilons times the
    largest normalized eigenvalue magnitude. This allows roundoff-scale
    negativity without imposing an absolute covariance threshold. The returned
    factor F satisfies F @ F.T approximately equal to the accepted covariance.
    """
    raw = np.asarray(covariance, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Covariance must be a square matrix")
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError("Covariance contains non-finite entries")
    scale = float(np.max(np.abs(raw)))
    if scale == 0.0:
        return np.zeros_like(raw)

    normalized = raw / scale
    normalized = 0.5 * (normalized + normalized.T)
    values, vectors = np.linalg.eigh(normalized)
    eigen_scale = float(np.max(np.abs(values)))
    relative_tolerance = 100.0 * np.finfo(float).eps * eigen_scale
    if float(np.min(values)) < -relative_tolerance:
        raise FloatingPointError("Covariance is not positive semidefinite")
    values = np.maximum(values, 0.0)
    return np.sqrt(scale) * (vectors @ np.diag(np.sqrt(values)))


def sample_ou_innovation(
    duration: float,
    gamma: float,
    mass: float,
    kbt: float,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> Array:
    """Sample joint (position-noise, velocity-noise) OU innovations.

    Returned shape is ``shape + (2 coordinates, 2 state components)``.
    """
    _, covariance = ou_transition_matrices(duration, gamma, mass, kbt)
    factor = _psd_factor(covariance)
    z = rng.normal(size=shape + (2, 2))
    return z @ factor.T


def ou_transition(
    positions: Array,
    velocities: Array,
    terminal_velocity: Array,
    duration: float,
    gamma: float,
    mass: float,
    innovation: Array,
) -> tuple[Array, Array]:
    """Apply frozen-force exact joint position-velocity OU transition."""
    tau = mass / gamma
    a = np.exp(-duration / tau)
    one_minus_a = -np.expm1(-duration / tau)
    x = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    g = np.asarray(terminal_velocity, dtype=float)
    eta = np.asarray(innovation, dtype=float)
    x_next = x + g * duration + tau * one_minus_a * (v - g) + eta[..., 0]
    v_next = g + a * (v - g) + eta[..., 1]
    return x_next, v_next


def conditional_split_ou_innovation(
    parent: Array,
    duration: float,
    split_time: float,
    gamma: float,
    mass: float,
    kbt: float,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Conditionally split a joint OU innovation at a deterministic time.

    The split time must be selected independently of this innovation. Dyadic
    callers use duration/2. The children compose exactly to the parent.
    """
    if not (0.0 < split_time < duration):
        raise ValueError("split_time must lie strictly inside the interval")
    remainder = duration - split_time
    a_remainder, q_remainder = ou_transition_matrices(remainder, gamma, mass, kbt)
    _, q_first = ou_transition_matrices(split_time, gamma, mass, kbt)
    _, q_total = ou_transition_matrices(duration, gamma, mass, kbt)
    gain = q_first @ a_remainder.T @ np.linalg.pinv(q_total, rcond=1e-14)
    conditional_cov = q_first - gain @ q_total @ gain.T
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
    factor = _psd_factor(conditional_cov)
    eta = np.asarray(parent, dtype=float)
    z = rng.normal(size=eta.shape)
    first = eta @ gain.T + z @ factor.T
    second = eta - first @ a_remainder.T
    return first, second


def conditional_ou_bridge_midpoint(
    positions_start: Array,
    velocities_start: Array,
    positions_end: Array,
    velocities_end: Array,
    terminal_velocity: Array,
    duration: float,
    gamma: float,
    mass: float,
    kbt: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array]:
    """Sample midpoint state conditional on both endpoints for frozen g."""
    midpoint_time = duration / 2.0
    parent_eta = _state_innovation(
        positions_start, velocities_start, positions_end, velocities_end,
        terminal_velocity, duration, gamma, mass,
    )
    left_eta, _ = conditional_split_ou_innovation(
        parent_eta, duration, midpoint_time, gamma, mass, kbt, rng
    )
    x_mid, v_mid = ou_transition(
        positions_start, velocities_start, terminal_velocity, midpoint_time,
        gamma, mass, left_eta,
    )
    _, q_first = ou_transition_matrices(midpoint_time, gamma, mass, kbt)
    a_right, _ = ou_transition_matrices(duration - midpoint_time, gamma, mass, kbt)
    _, q_total = ou_transition_matrices(duration, gamma, mass, kbt)
    gain = q_first @ a_right.T @ np.linalg.pinv(q_total, rcond=1e-14)
    conditional_cov = q_first - gain @ q_total @ gain.T
    return x_mid, v_mid, conditional_cov


def _state_innovation(
    x0: Array,
    v0: Array,
    x1: Array,
    v1: Array,
    g: Array,
    duration: float,
    gamma: float,
    mass: float,
) -> Array:
    tau = mass / gamma
    a = np.exp(-duration / tau)
    one_minus_a = -np.expm1(-duration / tau)
    eta_x = np.asarray(x1) - (
        np.asarray(x0) + np.asarray(g) * duration + tau * one_minus_a * (np.asarray(v0) - g)
    )
    eta_v = np.asarray(v1) - (np.asarray(g) + a * (np.asarray(v0) - g))
    return np.stack((eta_x, eta_v), axis=-1)


def split_rotational_increment(
    parent_increment: Array,
    duration: float,
    split_time: float,
    dr: float,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """Brownian-bridge split for rotational increments at deterministic time."""
    if not (0.0 < split_time < duration):
        raise ValueError("split_time must lie strictly inside the interval")
    remainder = duration - split_time
    parent = np.asarray(parent_increment, dtype=float)
    mean = (split_time / duration) * parent
    variance = 2.0 * dr * split_time * remainder / duration
    first = mean + np.sqrt(max(variance, 0.0)) * rng.normal(size=parent.shape)
    return first, parent - first


def _unit(theta: Array) -> Array:
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _predictor_corrector(
    state: InertialState,
    duration: float,
    rotation_increment: Array,
    rotation_midpoint_increment: Array,
    innovation: Array,
    params: Parameters,
) -> tuple[InertialState, Array]:
    """Midpoint-force predictor-corrector; approximation for interacting ABP."""
    x, v, theta = state
    force_start = calculate_repulsion(x, params)
    g_start = params.v0 * _unit(theta) + params.mobility * force_start
    # Since 1/gamma == mobility, this is V0*u + F/gamma.
    pred_x, _ = ou_transition(
        x, v, g_start, duration, params.gamma, params.mass, innovation
    )
    midpoint_positions = 0.5 * (x + pred_x)
    midpoint_positions, _ = project_overdamped_boundary(
        midpoint_positions, params.dish_radius
    )
    midpoint_theta = theta + rotation_midpoint_increment
    force_mid = calculate_repulsion(midpoint_positions, params)
    g_mid = params.v0 * _unit(midpoint_theta) + params.mobility * force_mid
    corrected_x, corrected_v = ou_transition(
        x, v, g_mid, duration, params.gamma, params.mass, innovation
    )
    proposed = InertialState(
        corrected_x,
        corrected_v,
        np.mod(theta + rotation_increment, 2.0 * np.pi),
    )
    return proposed, g_mid


def _segment_intersects_circle(p0: Array, p1: Array, radius: float) -> bool:
    """Whether a closed line segment intersects/touches the circle."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    aa = float(np.dot(delta, delta))
    if aa == 0.0:
        return abs(float(np.linalg.norm(p0)) - radius) <= 1e-12
    bb = 2.0 * float(np.dot(p0, delta))
    cc = float(np.dot(p0, p0) - radius * radius)
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < -1e-12:
        return False
    root = np.sqrt(max(discriminant, 0.0))
    t1 = (-bb - root) / (2.0 * aa)
    t2 = (-bb + root) / (2.0 * aa)
    return (-1e-12 <= t1 <= 1.0 + 1e-12) or (-1e-12 <= t2 <= 1.0 + 1e-12)


def _advance_inertial_interval(
    state: InertialState,
    duration: float,
    innovation: Array,
    rotation_increment: Array,
    params: Parameters,
    translation_rng: np.random.Generator,
    rotation_rng: np.random.Generator,
    bridge_rng: np.random.Generator,
    depth: int,
) -> tuple[InertialState, int, int, int]:
    """Return accepted state and correction/refinement/depth diagnostics."""
    x0, v0, theta0 = state
    norms0 = np.linalg.norm(x0, axis=1)
    on_wall_outward = (
        np.abs(norms0 - params.dish_radius) <= params.boundary_tolerance
    ) & (np.sum(v0 * (x0 / np.maximum(norms0[:, None], 1e-300)), axis=1) > 0.0)
    if np.any(on_wall_outward):
        x0 = x0.copy()
        v0 = v0.copy()
        normals = x0[on_wall_outward] / norms0[on_wall_outward, None]
        normal_speeds = np.sum(v0[on_wall_outward] * normals, axis=1)
        v0[on_wall_outward] -= 2.0 * normal_speeds[:, None] * normals
        state = InertialState(x0, v0, theta0)

    # Draw one conditional dyadic split. It supplies the inspected midpoint
    # and, if needed, the exact same child noise path for recursive refinement.
    half = duration / 2.0
    eta_left, eta_right = conditional_split_ou_innovation(
        innovation, duration, half, params.gamma, params.mass, params.kbt,
        bridge_rng,
    )
    rot_left, rot_right = split_rotational_increment(
        rotation_increment, duration, half, params.dr, bridge_rng
    )

    # This parent proposal is non-mutating until the refinement decision.
    proposal, g_mid = _predictor_corrector(
        state, duration, rotation_increment, rot_left, innovation, params
    )
    bridge_x, bridge_v = ou_transition(
        state.positions, state.velocities, g_mid, half,
        params.gamma, params.mass, eta_left,
    )

    x_end = proposal.positions
    r0 = np.linalg.norm(state.positions, axis=1)
    rm = np.linalg.norm(bridge_x, axis=1)
    r1 = np.linalg.norm(x_end, axis=1)
    endpoint_outside = r1 > params.dish_radius
    midpoint_outside = rm > params.dish_radius
    # Chord tests identify possible crossings. Outside endpoints are corrected
    # when accepted; ordinary_leaf_width targets near-wall refinement and is
    # not a global maximum interval width.
    crossing_indication = np.zeros(params.n_particles, dtype=bool)
    for i in range(params.n_particles):
        crossing_indication[i] = (
            _segment_intersects_circle(state.positions[i], bridge_x[i], params.dish_radius)
            or _segment_intersects_circle(bridge_x[i], x_end[i], params.dish_radius)
        )
    cross_and_return = (
        (r0 <= params.dish_radius)
        & (r1 <= params.dish_radius)
        & (midpoint_outside | (crossing_indication & (rm > params.dish_radius)))
    )

    _, conditional_cov = ou_transition_matrices(
        duration / 2.0, params.gamma, params.mass, params.kbt
    )
    # Position uncertainty conditional on the parent endpoints.
    right_a, _ = ou_transition_matrices(duration / 2.0, params.gamma, params.mass, params.kbt)
    _, q_parent = ou_transition_matrices(duration, params.gamma, params.mass, params.kbt)
    gain = conditional_cov @ right_a.T @ np.linalg.pinv(q_parent, rcond=1e-14)
    bridge_cov = conditional_cov - gain @ q_parent @ gain.T
    bridge_cov = 0.5 * (bridge_cov + bridge_cov.T)
    sigma_pos = np.sqrt(max(float(bridge_cov[0, 0]), 0.0))
    sampled_speeds = np.concatenate((
        np.linalg.norm(state.velocities, axis=1),
        np.linalg.norm(bridge_v, axis=1),
        np.linalg.norm(proposal.velocities, axis=1),
    ))
    threshold = 3.0 * sigma_pos + 0.5 * duration * float(np.max(sampled_speeds))
    clearance = np.minimum.reduce((
        params.dish_radius - r0,
        params.dish_radius - rm,
        params.dish_radius - r1,
    ))
    near_wall = clearance <= threshold

    ordinary_refine = (
        duration > params.ordinary_leaf_width and bool(np.any(near_wall))
    )
    unresolved_refine = bool(np.any(cross_and_return))

    # Endpoint-outside events take precedence at the configured leaf/hard cap.
    can_refine = depth < params.max_refinement_depth
    must_refine = ordinary_refine or unresolved_refine
    if must_refine and can_refine:
        left_state, c1, r1_count, d1 = _advance_inertial_interval(
            state, half, eta_left, rot_left, params, translation_rng,
            rotation_rng, bridge_rng, depth + 1,
        )
        right_state, c2, r2_count, d2 = _advance_inertial_interval(
            left_state, half, eta_right, rot_right, params, translation_rng,
            rotation_rng, bridge_rng, depth + 1,
        )
        return right_state, c1 + c2, r1_count + r2_count + 1, max(d1, d2)

    if unresolved_refine and not can_refine:
        raise UnresolvedBoundaryCrossing(
            "Cross-and-return indication persisted at max_refinement_depth"
        )

    # Commit the already computed proposal exactly once.
    corrected_x, corrected_v, corrections = apply_inertial_endpoint_boundary(
        proposal.positions, proposal.velocities, params.dish_radius,
        params.boundary_tolerance,
    )
    accepted = InertialState(corrected_x, corrected_v, proposal.orientations)
    # Forces are deliberately recomputed on the next predictor from corrected_x.
    return accepted, corrections, 0, depth


def inertial_step(
    state: InertialState,
    params: Parameters,
    translation_rng: np.random.Generator,
    rotation_rng: np.random.Generator,
    bridge_rng: np.random.Generator,
    innovation: Array | None = None,
    rotation_increment: Array | None = None,
) -> tuple[InertialState, StepDiagnostics]:
    """Advance one inertial macro-step with conditional dyadic refinement."""
    if innovation is None:
        innovation = sample_ou_innovation(
            params.dt, params.gamma, params.mass, params.kbt,
            translation_rng, (state.positions.shape[0],),
        )
    if rotation_increment is None:
        rotation_increment = np.sqrt(2.0 * params.dr * params.dt) * rotation_rng.normal(
            size=state.orientations.shape
        )
    result, corrections, refinements, max_depth = _advance_inertial_interval(
        state, params.dt, innovation, rotation_increment, params,
        translation_rng, rotation_rng, bridge_rng, 0,
    )
    return result, StepDiagnostics(corrections, refinements, max_depth)
