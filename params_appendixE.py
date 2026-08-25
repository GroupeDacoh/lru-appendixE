"""
Paramètres Appendix E / Tableau I pour la LRU flux-activated.

Objectif :
- centraliser les paramètres expérimentaux publiés du qubit auxiliaire A
- séparer les paramètres physiques du papier des paramètres purement numériques
- servir de base au simulateur microscopique `sim_lru_appendixE.py`
"""

from __future__ import annotations

import math

# Qubit auxiliaire A (Table I)
omega_ge_Hz = 6.281e9
alpha_Hz = -154e6
T1_idle_s = 21e-6
T2_star_idle_s = 34e-6
T2_echo_idle_s = 35e-6

# Paramètres du transmon flux-tunable (Appendix E)
E_C_Hz = 159e6
E_J_max_Hz = 33.094e9
junction_asymmetry_d = 0.776
phi_dc = 0.0

# Readout resonator / Purcell filter (Table I)
# `omega_ro_Hz` est la fréquence de readout reportée ; dans le H Appendix E
# elle entre comme ω_{r,b} (valeur publiée). La résonance sideband au point O
# est ensuite imposée comme dans l'expérience (calibration ω_m / ω_a).
omega_ro_Hz = 7.129e9
kappa_r_Hz = 16.4e6
omega_p_Hz = 7.111e9
kappa_p_Hz = 35e6
g_qr_Hz = 120e6
J_Hz = 28.8e6

# Operating point O expérimental (Lacroix Fig. 2)
omega_m_exp_Hz = 564e6
omega_a_exp_Hz = 128e6
pulse_duration_active_exp_ns = 34.5

omega_m_Hz = omega_m_exp_Hz
omega_a_Hz = omega_a_exp_Hz
pulse_duration_active_ns = pulse_duration_active_exp_ns
sigma_gaussian_ns = 5.0
buffer_ns = 2.0 * sigma_gaussian_ns
pulse_duration_total_ns = pulse_duration_active_ns + 2.0 * buffer_ns
omega_r_bare_Hz = omega_ro_Hz

# Figures of merit
target_residual_leakage = 6e-4
target_subspace_error = 2.5e-3
thermal_floor = 2.0e-4

# Informations issues de l'Appendix F (non utilisées dans le modèle Appendix E pur).
T1_pulse_s = 13.4e-6
T2_star_pulse_s = 10.8e-6
phase_rotation_crosstalk_deg = 3.6

# Troncatures : papier Appendix E = (6, 6, 6)
charge_basis_cutoff = 12
transmon_levels = 6
resonator_levels = 6
purcell_levels = 6

# Facteur appliqué au g sideband microscopique (Fourier de δE_J × ⟨f|cos|e⟩)
# pour que le 1er minimum tombe à τ=34.5 ns (comme Fig. 2b). Sans ce facteur,
# le g Fourier (~15 MHz) place le minimum trop tôt et le leakage final remonte.
sideband_g_scale = 0.696


def rad_s(freq_hz: float) -> float:
    return 2.0 * math.pi * freq_hz


def gamma_phi_from_t1_t2(T1_s: float, T2_s: float) -> float:
    return max(0.0, 1.0 / T2_s - 1.0 / (2.0 * T1_s))
