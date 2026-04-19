from __future__ import annotations

import numpy as np

from pk_first_pass import EPS, TAU_RATIO, unresolved_subspace


def weakest_direction_basis(h_current: np.ndarray) -> np.ndarray:
    """Return the weakest right-singular direction as a 1D basis."""
    _, _, vt = np.linalg.svd(h_current, full_matrices=False)
    return vt[-1:].T


def current_unresolved_basis(
    h_current: np.ndarray,
    tau_ratio: float = TAU_RATIO,
    fallback_to_weakest: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    u_tau, singular_values, tau = unresolved_subspace(h_current, tau_ratio=tau_ratio)
    if u_tau.shape[1] == 0 and fallback_to_weakest:
        return weakest_direction_basis(h_current), singular_values, tau
    return u_tau, singular_values, tau


def unresolved_information_matrix(
    h_block: np.ndarray,
    u_basis: np.ndarray,
    noise_var: float = 1.0,
) -> np.ndarray:
    if u_basis.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    return (u_basis.T @ h_block.T @ h_block @ u_basis) / max(noise_var, EPS)


def unresolved_posterior_covariance(
    h_current: np.ndarray,
    u_basis: np.ndarray,
    prior_var: float = 1.0,
    noise_var: float = 1.0,
) -> np.ndarray:
    k = u_basis.shape[1]
    if k == 0:
        return np.zeros((0, 0), dtype=float)
    prior_precision = np.eye(k, dtype=float) / max(prior_var, EPS)
    g_current = unresolved_information_matrix(h_current, u_basis, noise_var=noise_var)
    precision = prior_precision + g_current
    return np.linalg.inv(precision)


def unresolved_aopt_score(
    h_current: np.ndarray,
    h_candidate: np.ndarray,
    tau_ratio: float = TAU_RATIO,
    prior_var: float = 1.0,
    noise_var: float = 1.0,
    fallback_to_weakest: bool = False,
) -> tuple[float, float, int]:
    """
    Exact unresolved A-opt score under the local linear-Gaussian bridge.

    Returns:
        score: trace reduction on the current unresolved posterior covariance
        current_trace: current unresolved posterior trace
        unresolved_dim: dimension of the unresolved basis used
    """
    u_tau, _, _ = current_unresolved_basis(
        h_current,
        tau_ratio=tau_ratio,
        fallback_to_weakest=fallback_to_weakest,
    )
    k = u_tau.shape[1]
    if k == 0:
        return 0.0, 0.0, 0

    lambda_cur = unresolved_posterior_covariance(
        h_current,
        u_tau,
        prior_var=prior_var,
        noise_var=noise_var,
    )
    g_e = unresolved_information_matrix(h_candidate, u_tau, noise_var=noise_var)
    current_trace = float(np.trace(lambda_cur))
    next_cov = np.linalg.inv(np.linalg.inv(lambda_cur) + g_e)
    score = current_trace - float(np.trace(next_cov))
    return float(score), current_trace, int(k)


def weighted_cartograph_score(
    h_current: np.ndarray,
    h_candidate: np.ndarray,
    tau_ratio: float = TAU_RATIO,
    prior_var: float = 1.0,
    noise_var: float = 1.0,
    fallback_to_weakest: bool = False,
) -> tuple[float, float, int]:
    """
    Covariance-weighted unresolved projection score.

    This is a stronger surrogate than the raw projection score but still cheaper
    than exact A-opt. It is useful diagnostically when comparing the methods.
    """
    u_tau, _, _ = current_unresolved_basis(
        h_current,
        tau_ratio=tau_ratio,
        fallback_to_weakest=fallback_to_weakest,
    )
    k = u_tau.shape[1]
    if k == 0:
        return 0.0, 0.0, 0
    lambda_cur = unresolved_posterior_covariance(
        h_current,
        u_tau,
        prior_var=prior_var,
        noise_var=noise_var,
    )
    projected = h_candidate @ u_tau @ lambda_cur
    score = float(np.linalg.norm(projected, ord="fro") ** 2)
    return score, float(np.trace(lambda_cur)), int(k)
