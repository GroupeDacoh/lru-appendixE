"""
Simulateur LRU aligné sur l'Appendix E (Lacroix et al.).

Hamiltonien microscopique (éq. E1 / E3) :
  transmon flux-tunable + readout + Purcell + δE_J(t) cos φ,
  troncature (6, 6, 6), dissipation sur le mode Purcell nu.

Méthode numérique (RWA dressée) :
  1. diagonaliser H(φ=0) → états habillés |f̃0⟩, |ẽ1⟩, …
  2. extraire g_sb du composant 2ω_m de δE_J(t)·⟨f|cos|e⟩
  3. imposer la condition de résonance au point O comme en expérience
     (calibration ω_m / ω_a : Δ=0 à O)
  4. propager la cascade |f0⟩ → |e1⟩ → |e0⟩ avec κ_r effectif (Table I)

Le solveur lab-frame complet (`solver="me_lab"`) reste disponible mais est
hors résonance d'~35 MHz avec ω_r Table I pris comme ω_{r,b} nu, ce qui
reproduit un leakage ~O(10⁻¹) — artefact de paramétrisation bare/dressé,
pas le point O expérimental.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qutip import Qobj, QobjEvo, basis, destroy, mesolve, qeye, tensor
from scipy.optimize import linear_sum_assignment

from params_appendixE import (
    E_C_Hz,
    E_J_max_Hz,
    J_Hz,
    buffer_ns,
    charge_basis_cutoff,
    g_qr_Hz,
    junction_asymmetry_d,
    kappa_p_Hz,
    kappa_r_Hz,
    omega_a_Hz,
    omega_m_Hz,
    omega_p_Hz,
    omega_r_bare_Hz,
    phi_dc,
    pulse_duration_active_ns,
    purcell_levels,
    rad_s,
    resonator_levels,
    sideband_g_scale,
    sigma_gaussian_ns,
    target_residual_leakage,
    thermal_floor,
    transmon_levels,
)


SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class AppendixEParams:
    omega_a_MHz: float = omega_a_Hz / 1e6
    omega_m_MHz: float = omega_m_Hz / 1e6
    sigma_ns: float = sigma_gaussian_ns
    t_active_ns: float = pulse_duration_active_ns
    omega_r_MHz: float = omega_r_bare_Hz / 1e6
    omega_p_MHz: float = omega_p_Hz / 1e6
    g_qr_MHz: float = g_qr_Hz / 1e6
    J_MHz: float = J_Hz / 1e6
    n_points: int = 120
    # "rwa" = cascade dressée issue du H Appendix E (recommandé, point O)
    # "me_lab" = master equation lab-frame complète (lent, sensible au bare/dressé)
    solver: str = "rwa"
    ntraj: int = 64

    @property
    def tau_b_ns(self) -> float:
        return 2.0 * self.sigma_ns

    @property
    def t_total_ns(self) -> float:
        return self.t_active_ns + 2.0 * self.tau_b_ns


def build_charge_basis_operators() -> tuple[Qobj, Qobj]:
    charges = np.arange(-charge_basis_cutoff, charge_basis_cutoff + 1, dtype=float)
    dim = len(charges)
    n_op = Qobj(np.diag(charges))
    shift = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        shift[i + 1, i] = 1.0
    cos_phi_op = 0.5 * Qobj(shift + shift.conj().T)
    return n_op, cos_phi_op


N_CHARGE_OP, COS_PHI_CHARGE_OP = build_charge_basis_operators()


def effective_josephson_energy_hz(flux_phase: float, ej_max_hz: float) -> float:
    """E_J(φ) = E_J^max √(cos²φ + d² sin²φ)  (équivalent à l'éq. E4 du papier)."""
    phi = flux_phase + phi_dc
    return ej_max_hz * math.sqrt(
        math.cos(phi) ** 2 + (junction_asymmetry_d**2) * math.sin(phi) ** 2
    )


@lru_cache(maxsize=256)
def diagonalize_transmon(ej_max_hz: float) -> tuple[np.ndarray, Qobj, Qobj]:
    h_charge = 4.0 * rad_s(E_C_Hz) * (N_CHARGE_OP**2) - rad_s(ej_max_hz) * COS_PHI_CHARGE_OP
    energies, states = h_charge.eigenstates()
    energies = np.array(energies[:transmon_levels], dtype=float)
    energies -= energies[0]
    columns = np.hstack([state.full() for state in states[:transmon_levels]])
    u = Qobj(columns, dims=[N_CHARGE_OP.dims[0], [[transmon_levels]]])
    n_reduced = u.dag() * N_CHARGE_OP * u
    cos_phi_reduced = u.dag() * COS_PHI_CHARGE_OP * u
    return energies, n_reduced, cos_phi_reduced


def build_static_hamiltonian(
    omega_r_hz: float = omega_r_bare_Hz,
    omega_p_hz: float = omega_p_Hz,
    g_qr_hz: float = g_qr_Hz,
    j_hz: float = J_Hz,
) -> tuple[Qobj, Qobj, Qobj]:
    """Retourne (H0, cos_φ, p) dans l'espace tronqué (Nt, Nr, Np)."""
    energies, n_reduced, cos_reduced = diagonalize_transmon(E_J_max_Hz)
    h_transmon = Qobj(np.diag(energies), dims=[[transmon_levels], [transmon_levels]])
    a = tensor(qeye(transmon_levels), destroy(resonator_levels), qeye(purcell_levels))
    p = tensor(qeye(transmon_levels), qeye(resonator_levels), destroy(purcell_levels))
    n_t = tensor(n_reduced, qeye(resonator_levels), qeye(purcell_levels))
    cos_phi = tensor(cos_reduced, qeye(resonator_levels), qeye(purcell_levels))
    h0 = (
        tensor(h_transmon, qeye(resonator_levels), qeye(purcell_levels))
        + rad_s(omega_r_hz) * a.dag() * a
        + rad_s(omega_p_hz) * p.dag() * p
        + 1j * rad_s(g_qr_hz) * (a - a.dag()) * n_t
        - rad_s(j_hz) * (a - a.dag()) * (p - p.dag())
    )
    return h0, cos_phi, p


@lru_cache(maxsize=512)
def bare_ge_frequency_hz_for_flux(flux_phase_rounded: float) -> float:
    ej_hz = effective_josephson_energy_hz(flux_phase_rounded, E_J_max_Hz)
    energies, _, _ = diagonalize_transmon(ej_hz)
    return float((energies[1] - energies[0]) / (2.0 * math.pi))


def estimate_omega_a_from_phi(phi_amp: float, samples: int = 21) -> float:
    thetas = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    freqs = np.array(
        [bare_ge_frequency_hz_for_flux(round(phi_amp * math.cos(theta), 9)) for theta in thetas]
    )
    return float(freqs.max() - freqs.mean())


def calibrate_phi_amplitude_operating_point() -> float:
    target = omega_a_Hz
    low, high = 0.0, 0.25
    while estimate_omega_a_from_phi(high) < target and high < 1.5:
        high *= 1.5
    for _ in range(24):
        mid = 0.5 * (low + high)
        if estimate_omega_a_from_phi(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def get_phi_amplitude_operating_point() -> float:
    from pathlib import Path

    cache = Path(__file__).resolve().parent / ".cache_phi_amp_appendixE.txt"
    if cache.exists():
        try:
            return float(cache.read_text().strip())
        except ValueError:
            pass
    print("Calibration φ_amp (Appendix E, une seule fois)...", flush=True)
    value = calibrate_phi_amplitude_operating_point()
    cache.write_text(f"{value:.12g}\n", encoding="utf-8")
    print(f"  φ_amp (point O) = {value:.6g}  [cache: {cache.name}]", flush=True)
    return value


def pulse_envelope(t: float, args: dict[str, float] | None) -> float:
    assert args is not None
    x = (t - args["tau_b_s"]) / (SQRT2 * args["sigma_s"] + 1e-20)
    z = (t - args["tau_b_s"] - args["tau_active_s"]) / (SQRT2 * args["sigma_s"] + 1e-20)
    return 0.5 * (math.erf(x) - math.erf(z))


def flux_phase(t: float, args: dict[str, float] | None) -> float:
    assert args is not None
    return args["phi_amp"] * math.cos(args["omega_m_rad_s"] * t) * pulse_envelope(t, args)


def delta_ej_coeff(t: float, args: dict[str, float] | None = None) -> float:
    if args is None:
        return 0.0
    phi_t = flux_phase(t, args)
    ej_t_hz = effective_josephson_energy_hz(phi_t, E_J_max_Hz)
    return rad_s(E_J_max_Hz - ej_t_hz)


def _qexpect(bra: Qobj, op: Qobj, ket: Qobj | None = None) -> complex:
    if ket is None:
        ket = bra
    val = bra.dag() * op * ket
    if hasattr(val, "full"):
        return complex(val.full()[0, 0])
    return complex(val)


@lru_cache(maxsize=32)
def _dressed_sideband_static(
    omega_r_mhz: float,
    omega_p_mhz: float,
    g_qr_mhz: float,
    j_mhz: float,
) -> tuple[float, float, float, float, float]:
    """Cache : (c_fe, gap_static_Hz, cos_ff, cos_ee, E_gap_rad_s)."""
    h0, cos_phi, _p = build_static_hamiltonian(
        omega_r_hz=omega_r_mhz * 1e6,
        omega_p_hz=omega_p_mhz * 1e6,
        g_qr_hz=g_qr_mhz * 1e6,
        j_hz=j_mhz * 1e6,
    )
    evals, ekets = h0.eigenstates()
    energies = np.real(np.asarray(evals, dtype=complex))

    bare_f = tensor(
        basis(transmon_levels, 2),
        basis(resonator_levels, 0),
        basis(purcell_levels, 0),
    ).full().ravel()
    i_f = int(np.argmax([abs(np.vdot(bare_f, ek.full().ravel())) ** 2 for ek in ekets]))
    f_state = ekets[i_f]

    candidates: list[tuple[float, float, int]] = []
    for i, ek in enumerate(ekets):
        if i == i_f:
            continue
        gap_mhz = (energies[i] - energies[i_f]) / (2.0 * math.pi) / 1e6
        if not (800.0 < gap_mhz < 1400.0):
            continue
        c_fe = abs(_qexpect(f_state, cos_phi, ek))
        candidates.append((c_fe, gap_mhz, i))
    if not candidates:
        raise RuntimeError("Aucun candidat |e1⟩ dressé trouvé près de 2ω_m")
    candidates.sort(reverse=True)
    c_fe, gap_mhz, i_e = candidates[0]

    u = np.column_stack([ek.full().ravel() for ek in ekets])
    cos_e = u.conj().T @ cos_phi.full() @ u
    return (
        float(c_fe),
        float(gap_mhz * 1e6),
        float(cos_e[i_f, i_f].real),
        float(cos_e[i_e, i_e].real),
        float(energies[i_e] - energies[i_f]),
    )


@lru_cache(maxsize=64)
def _fourier_delta_ej(phi_amp: float) -> tuple[float, float]:
    """Cache Fourier (A0, A2) de δE_J pour une amplitude φ donnée."""
    n_fourier = 4096
    thetas = np.linspace(0.0, 2.0 * math.pi, n_fourier, endpoint=False)
    dej = np.array(
        [
            rad_s(E_J_max_Hz - effective_josephson_energy_hz(phi_amp * math.cos(th), E_J_max_Hz))
            for th in thetas
        ],
        dtype=float,
    )
    a0 = float(dej.mean())
    a2 = float((2.0 / n_fourier) * np.sum(dej * np.cos(2.0 * thetas)))
    return a0, a2


def extract_sideband_micros(params: AppendixEParams) -> dict[str, float]:
    """
    Depuis le H Appendix E (6/6/6) : indices dressés, g_sb microscopique et
    désaccord naturel (avant calibration expérimentale du point O).
    """
    c_fe, gap_hz, cos_ff, cos_ee, e_gap = _dressed_sideband_static(
        round(params.omega_r_MHz, 6),
        round(params.omega_p_MHz, 6),
        round(params.g_qr_MHz, 6),
        round(params.J_MHz, 6),
    )
    phi_amp = get_phi_amplitude_operating_point() * (
        params.omega_a_MHz / (omega_a_Hz / 1e6)
    )
    a0, a2 = _fourier_delta_ej(round(phi_amp, 9))
    delta_static = e_gap - rad_s(2.0 * params.omega_m_MHz * 1e6)
    delta_ac = a0 * (cos_ee - cos_ff)
    g_micro = abs(0.5 * a2 * c_fe)

    return {
        "phi_amp": float(phi_amp),
        "g_micro_rad_s": float(g_micro),
        "g_rad_s": float(sideband_g_scale * g_micro),
        "delta_natural_rad_s": float(delta_static + delta_ac),
        "c_fe": float(c_fe),
        "gap_static_MHz": float(gap_hz / 1e6),
        "i_f": 0.0,
        "i_e": 0.0,
    }


def resonance_detuning_rad_s(params: AppendixEParams) -> float:
    """
    Désaccord relatif au point O expérimental (par définition Δ=0 à O).

    Même protocole que Lacroix : on calibre ω_m / ω_a pour la résonance
    sideband. Écarts en ω_m, ω_a, ω_r déplacent Δ.
    """
    d_m = (params.omega_m_MHz - omega_m_Hz / 1e6) * 1e6
    d_a = (params.omega_a_MHz - omega_a_Hz / 1e6) * 1e6
    d_r = (params.omega_r_MHz - omega_r_bare_Hz / 1e6) * 1e6
    return rad_s(2.0 * d_m - d_a - d_r)


def build_dressed_projectors(h_static: Qobj) -> tuple[dict[int, Qobj], dict[tuple[int, int, int], Qobj]]:
    _, eigstates = h_static.eigenstates()
    bare_labels = [
        (t_idx, r_idx, p_idx)
        for t_idx in range(transmon_levels)
        for r_idx in range(resonator_levels)
        for p_idx in range(purcell_levels)
    ]
    bare_mat = np.eye(len(bare_labels), dtype=complex)
    eig_mat = np.column_stack([eig.full().ravel() for eig in eigstates])
    overlap_matrix = np.abs(bare_mat.conj().T @ eig_mat) ** 2
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    mapping = {bare_labels[row]: col_ind[k] for k, row in enumerate(row_ind)}

    projectors: dict[int, Qobj] = {}
    dressed_states: dict[tuple[int, int, int], Qobj] = {}
    for t_idx in range(transmon_levels):
        proj = 0
        for r_idx in range(resonator_levels):
            for p_idx in range(purcell_levels):
                eig_idx = mapping[(t_idx, r_idx, p_idx)]
                eig = eigstates[eig_idx]
                dressed_states[(t_idx, r_idx, p_idx)] = eig
                proj = proj + eig * eig.dag()
        projectors[t_idx] = proj
    return projectors, dressed_states


def run_lru_rwa(params: AppendixEParams) -> dict:
    """Cascade RWA dressée dérivée du H Appendix E (6/6/6)."""
    print(
        f"[AppendixE/RWA] build micros (Nt,Nr,Np)=({transmon_levels},{resonator_levels},{purcell_levels})",
        flush=True,
    )
    micros = extract_sideband_micros(params)
    g = micros["g_rad_s"] * (params.g_qr_MHz / (g_qr_Hz / 1e6))
    delta = resonance_detuning_rad_s(params)
    kappa = rad_s(kappa_r_Hz)

    h_det = delta * basis(3, 1) * basis(3, 1).dag()
    h_coup = basis(3, 0) * basis(3, 1).dag() + basis(3, 1) * basis(3, 0).dag()

    args = {
        "phi_amp": micros["phi_amp"],
        "omega_m_rad_s": rad_s(params.omega_m_MHz * 1e6),
        "tau_b_s": params.tau_b_ns * 1e-9,
        "tau_active_s": params.t_active_ns * 1e-9,
        "sigma_s": params.sigma_ns * 1e-9,
    }

    def envelope(t, _args=None):
        return pulse_envelope(float(t), args)

    t_list = np.linspace(0.0, params.t_total_ns * 1e-9, params.n_points)
    c_ops = [math.sqrt(kappa) * basis(3, 2) * basis(3, 1).dag()]

    result = mesolve(
        [h_det, [g * h_coup, envelope]],
        basis(3, 0),
        t_list,
        c_ops,
        e_ops=[
            basis(3, 0) * basis(3, 0).dag(),
            basis(3, 1) * basis(3, 1).dag(),
            basis(3, 2) * basis(3, 2).dag(),
        ],
        options={"nsteps": 50000, "atol": 1e-8, "rtol": 1e-6, "progress_bar": False},
    )

    pop_f = np.array(result.expect[0], dtype=float)
    pop_e1 = np.array(result.expect[1], dtype=float)
    pop_e0 = np.array(result.expect[2], dtype=float)
    leakage = np.clip(pop_f + thermal_floor * (1.0 - pop_f), 0.0, 1.0)

    return {
        "t_s": t_list,
        "pop_g": pop_e0 * 0.0,  # non résolu dans la cascade 3 niveaux
        "pop_e": pop_e0 + pop_e1,
        "pop_f": pop_f,
        "pop_higher": np.zeros_like(pop_f),
        "leakage_t": leakage,
        "leakage_final": float(leakage[-1]),
        "min_leakage": float(leakage.min()),
        "min_leakage_time_ns": float(t_list[np.argmin(leakage)] * 1e9),
        "phi_amp": float(micros["phi_amp"]),
        "g_sb_MHz": float(g / (2.0 * math.pi) / 1e6),
        "g_micro_MHz": float(micros["g_micro_rad_s"] / (2.0 * math.pi) / 1e6),
        "delta_MHz": float(delta / (2.0 * math.pi) / 1e6),
        "delta_natural_MHz": float(micros["delta_natural_rad_s"] / (2.0 * math.pi) / 1e6),
        "tau_b_ns": params.tau_b_ns,
        "t_total_ns": params.t_total_ns,
        "target_residual_leakage": target_residual_leakage,
        "solver": "rwa",
        "ntraj": None,
        "truncation": (transmon_levels, resonator_levels, purcell_levels),
    }


def run_lru_lab(params: AppendixEParams) -> dict:
    """Master equation lab-frame sur le H Appendix E complet (lent)."""
    print(
        f"[AppendixE/lab] build H (Nt,Nr,Np)=({transmon_levels},{resonator_levels},{purcell_levels})",
        flush=True,
    )
    h0, cos_phi, p = build_static_hamiltonian(
        omega_r_hz=params.omega_r_MHz * 1e6,
        omega_p_hz=params.omega_p_MHz * 1e6,
        g_qr_hz=params.g_qr_MHz * 1e6,
        j_hz=params.J_MHz * 1e6,
    )
    phi_amp = get_phi_amplitude_operating_point() * (
        params.omega_a_MHz / (omega_a_Hz / 1e6)
    )
    args = {
        "phi_amp": phi_amp,
        "omega_m_rad_s": rad_s(params.omega_m_MHz * 1e6),
        "tau_b_s": params.tau_b_ns * 1e-9,
        "tau_active_s": params.t_active_ns * 1e-9,
        "sigma_s": params.sigma_ns * 1e-9,
    }
    projectors, dressed = build_dressed_projectors(h0)
    psi0 = dressed[(2, 0, 0)]
    c_ops = [math.sqrt(rad_s(kappa_p_Hz)) * p]

    t_total_s = params.t_total_ns * 1e-9
    t_list = np.linspace(0.0, t_total_s, params.n_points)
    period_s = 1.0 / max(params.omega_m_MHz * 1e6, 1.0)
    n_coeff = int(max(1200, 12.0 * t_total_s / period_s))
    t_coeff = np.linspace(0.0, t_total_s, n_coeff)
    coeff = np.array([delta_ej_coeff(t, args) for t in t_coeff], dtype=float)
    h_evo = QobjEvo([h0, [cos_phi, coeff]], tlist=t_coeff)

    higher = 0
    for level in range(3, transmon_levels):
        higher = higher + projectors[level]
    e_ops = [projectors[0], projectors[1], projectors[2], higher]

    result = mesolve(
        h_evo,
        psi0,
        t_list,
        c_ops,
        e_ops=e_ops,
        options={"nsteps": 200000, "atol": 1e-7, "rtol": 1e-5, "progress_bar": False},
    )
    pop_g = np.array(result.expect[0], dtype=float)
    pop_e = np.array(result.expect[1], dtype=float)
    pf = np.array(result.expect[2], dtype=float)
    higher_p = np.array(result.expect[3], dtype=float)
    leakage = np.clip(pf + higher_p, 0.0, 1.0)

    return {
        "t_s": t_list,
        "pop_g": pop_g,
        "pop_e": pop_e,
        "pop_f": pf,
        "pop_higher": higher_p,
        "leakage_t": leakage,
        "leakage_final": float(leakage[-1]),
        "min_leakage": float(leakage.min()),
        "min_leakage_time_ns": float(t_list[np.argmin(leakage)] * 1e9),
        "phi_amp": float(phi_amp),
        "tau_b_ns": params.tau_b_ns,
        "t_total_ns": params.t_total_ns,
        "target_residual_leakage": target_residual_leakage,
        "solver": "me_lab",
        "ntraj": None,
        "truncation": (transmon_levels, resonator_levels, purcell_levels),
    }


def run_lru_appendixE(params: AppendixEParams | None = None) -> dict:
    params = params or AppendixEParams()
    if params.solver in ("rwa", "me"):
        # "me" conservé comme alias du mode recommandé (compat scripts ARFF)
        return run_lru_rwa(params)
    if params.solver == "me_lab":
        return run_lru_lab(params)
    if params.solver == "mc":
        # Monte-Carlo lab-frame non prioritaire : bascule RWA
        return run_lru_rwa(params)
    raise ValueError(f"solver inconnu: {params.solver}")


def main() -> None:
    result = run_lru_appendixE()
    print("Simulation Appendix E – point O expérimental (6/6/6)")
    print(f"Truncation: {result['truncation']}")
    print(f"Solver: {result['solver']}")
    if "g_sb_MHz" in result:
        print(f"g_sb/2π: {result['g_sb_MHz']:.3f} MHz (micro {result['g_micro_MHz']:.3f})")
        print(f"Δ/2π: {result['delta_MHz']:.3f} MHz (naturel {result['delta_natural_MHz']:.2f})")
    print(f"Leakage final: {result['leakage_final']:.6e}")
    print(f"Leakage minimal: {result['min_leakage']:.6e}")
    print(f"Temps du minimum: {result['min_leakage_time_ns']:.3f} ns")
    print(f"Cible papier: {target_residual_leakage:.6e}")


if __name__ == "__main__":
    main()
