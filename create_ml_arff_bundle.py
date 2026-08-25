"""
Génère 8 ARFF pulse-only (puce Table I fixe) pour l'étude ML.

Demande superviseur : 5–10 fichiers ARFF.
Chaque fichier = un protocole d'échantillonnage différent autour du point O.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from params_appendixE import (
    J_Hz,
    g_qr_Hz,
    omega_a_Hz,
    omega_m_Hz,
    omega_p_Hz,
    omega_ro_Hz,
    pulse_duration_active_ns,
    sigma_gaussian_ns,
)
from sim_lru_appendixE import AppendixEParams, resonance_detuning_rad_s, run_lru_appendixE


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "arff_ml_bundle"
N_POINTS = 50

ATTRS = [
    "omega_m_MHz",
    "omega_a_MHz",
    "sigma_ns",
    "T_active_ns",
    "T_total_ns",
    "omega_r_MHz",
    "omega_p_MHz",
    "g_qr_MHz",
    "J_MHz",
    "detuning_MHz",
    "leakage_final",
    "min_leakage",
    "min_leakage_time_ns",
]


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def chip() -> dict[str, float]:
    return {
        "omega_r_MHz": omega_ro_Hz / 1e6,
        "omega_p_MHz": omega_p_Hz / 1e6,
        "g_qr_MHz": g_qr_Hz / 1e6,
        "J_MHz": J_Hz / 1e6,
    }


def eval_cand(omega_m: float, omega_a: float, sigma: float, t_active: float) -> dict:
    cand = {
        "omega_m_MHz": float(omega_m),
        "omega_a_MHz": float(omega_a),
        "sigma_ns": float(sigma),
        "T_active_ns": float(t_active),
        **chip(),
    }
    params = AppendixEParams(
        omega_a_MHz=cand["omega_a_MHz"],
        omega_m_MHz=cand["omega_m_MHz"],
        sigma_ns=cand["sigma_ns"],
        t_active_ns=cand["T_active_ns"],
        omega_r_MHz=cand["omega_r_MHz"],
        omega_p_MHz=cand["omega_p_MHz"],
        g_qr_MHz=cand["g_qr_MHz"],
        J_MHz=cand["J_MHz"],
        n_points=N_POINTS,
        solver="rwa",
    )
    res = run_lru_appendixE(params)
    det = resonance_detuning_rad_s(params) / (2.0 * np.pi) / 1e6
    return {
        **cand,
        "T_total_ns": res["t_total_ns"],
        "detuning_MHz": float(det),
        "leakage_final": res["leakage_final"],
        "min_leakage": res["min_leakage"],
        "min_leakage_time_ns": res["min_leakage_time_ns"],
    }


def write_arff(path: Path, relation: str, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"@RELATION {relation}\n\n")
        for a in ATTRS:
            f.write(f"@ATTRIBUTE {a} NUMERIC\n")
        f.write("\n@DATA\n")
        for r in rows:
            f.write(",".join(f"{float(r[a]):.10g}" for a in ATTRS) + "\n")


def point_o() -> dict:
    return eval_cand(
        omega_m_Hz / 1e6,
        omega_a_Hz / 1e6,
        sigma_gaussian_ns,
        pulse_duration_active_ns,
    )


def build_01_full_mixed(rng: np.random.Generator, n: int = 400) -> list[dict]:
    rows = [point_o()]
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    # grille locale
    for oa in np.linspace(120.0, 136.0, 5):
        for dom in (-2.0, -1.0, 0.0, 1.0, 2.0):
            om = om0 + 0.5 * (oa - oa0) + dom
            for tact in (30.0, 32.0, 34.5, 37.0, 40.0):
                for sig in (4.0, 5.0, 6.0):
                    if len(rows) >= n:
                        return rows
                    rows.append(eval_cand(om, oa, sig, tact))
    while len(rows) < n:
        mode = rng.choice(["near", "res", "exp"], p=[0.4, 0.3, 0.3])
        if mode == "near":
            oa = clip(rng.normal(oa0, 3.0), 115, 140)
            om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0, 1.5), 545, 585)
            sig = clip(rng.normal(5.0, 0.6), 3.5, 7)
            tact = clip(rng.normal(34.5, 3.0), 28, 42)
        elif mode == "res":
            oa = clip(rng.normal(oa0, 12.0), 100, 155)
            om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0, 2.0), 525, 600)
            sig = clip(rng.normal(5.0, 1.0), 3, 8)
            tact = clip(rng.normal(34.5, 6.0), 22, 50)
        else:
            oa = clip(rng.uniform(100, 160), 100, 160)
            om = clip(rng.uniform(530, 600), 530, 600)
            sig = clip(rng.uniform(3, 8), 3, 8)
            tact = clip(rng.uniform(22, 50), 22, 50)
        rows.append(eval_cand(om, oa, sig, tact))
    return rows


def build_02_near_o(rng: np.random.Generator, n: int = 300) -> list[dict]:
    rows = [point_o()]
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    while len(rows) < n:
        oa = clip(rng.normal(oa0, 2.5), 118, 138)
        om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0, 1.0), 550, 578)
        sig = clip(rng.normal(5.0, 0.4), 4.0, 6.5)
        tact = clip(rng.normal(34.5, 2.0), 30, 40)
        rows.append(eval_cand(om, oa, sig, tact))
    return rows


def build_03_resonance(rng: np.random.Generator, n: int = 300) -> list[dict]:
    rows = [point_o()]
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    while len(rows) < n:
        oa = clip(rng.uniform(110, 150), 110, 150)
        # droite de résonance Lacroix-like : om ≈ om0 + 0.5*(oa-oa0)
        om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0, 0.8), 530, 600)
        sig = clip(rng.choice([4.0, 5.0, 6.0, 7.0]) + rng.normal(0, 0.2), 3, 8)
        tact = clip(rng.choice([28, 30, 32, 34.5, 37, 40, 45]) + rng.normal(0, 0.5), 22, 50)
        rows.append(eval_cand(om, oa, sig, tact))
    return rows


def build_04_explore(rng: np.random.Generator, n: int = 300) -> list[dict]:
    rows = [point_o()]
    while len(rows) < n:
        oa = clip(rng.uniform(100, 160), 100, 160)
        om = clip(rng.uniform(520, 610), 520, 610)
        sig = clip(rng.uniform(3, 8), 3, 8)
        tact = clip(rng.uniform(22, 50), 22, 50)
        rows.append(eval_cand(om, oa, sig, tact))
    return rows


def build_05_T_active_sweep() -> list[dict]:
    rows = [point_o()]
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    for oa in (120.0, 128.0, 136.0):
        om = om0 + 0.5 * (oa - oa0)
        for sig in (4.0, 5.0, 6.0):
            for tact in np.linspace(24.0, 48.0, 13):
                rows.append(eval_cand(om, oa, sig, float(tact)))
    return rows


def build_06_omega_m_sweep() -> list[dict]:
    rows = [point_o()]
    for oa in (120.0, 128.0, 136.0):
        for sig in (4.0, 5.0, 6.0):
            for om in np.linspace(540.0, 590.0, 21):
                rows.append(eval_cand(float(om), oa, sig, 34.5))
    return rows


def build_07_omega_a_sweep() -> list[dict]:
    rows = [point_o()]
    om0, oa0 = omega_m_Hz / 1e6, omega_a_Hz / 1e6
    for om_off in (-4.0, 0.0, 4.0):
        for sig in (4.0, 5.0, 6.0):
            for oa in np.linspace(110.0, 150.0, 21):
                om = om0 + 0.5 * (oa - oa0) + om_off
                rows.append(eval_cand(om, float(oa), sig, 34.5))
    return rows


def build_08_sigma_sweep() -> list[dict]:
    rows = [point_o()]
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    for oa in (120.0, 128.0, 136.0):
        om = om0 + 0.5 * (oa - oa0)
        for tact in (30.0, 34.5, 40.0):
            for sig in np.linspace(3.0, 8.0, 11):
                rows.append(eval_cand(om, oa, float(sig), tact))
    return rows


SPECS = [
    ("01_full_mixed", "appendixE_01_full_mixed", "Mélange grille+aléatoire autour de O (baseline)", build_01_full_mixed),
    ("02_near_O", "appendixE_02_near_O", "Échantillonnage dense près du point O", build_02_near_o),
    ("03_resonance", "appendixE_03_resonance", "Sur la droite de résonance (2Δω_m≈Δω_a)", build_03_resonance),
    ("04_explore", "appendixE_04_explore", "Contrastes hors résonance / hors O", build_04_explore),
    ("05_T_active_sweep", "appendixE_05_T_active_sweep", "Balayage systématique de T_active", lambda rng: build_05_T_active_sweep()),
    ("06_omega_m_sweep", "appendixE_06_omega_m_sweep", "Balayage systématique de ω_m", lambda rng: build_06_omega_m_sweep()),
    ("07_omega_a_sweep", "appendixE_07_omega_a_sweep", "Balayage systématique de ω_a", lambda rng: build_07_omega_a_sweep()),
    ("08_sigma_sweep", "appendixE_08_sigma_sweep", "Balayage systématique de σ", lambda rng: build_08_sigma_sweep()),
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    manifest = []

    for short, relation, desc, builder in SPECS:
        print(f"\n=== {short} : {desc} ===", flush=True)
        rows = builder(rng)
        path = OUT_DIR / f"qutip_{short}.arff"
        write_arff(path, relation, rows)
        y = np.array([r["leakage_final"] for r in rows])
        info = {
            "file": path.name,
            "description": desc,
            "n": len(rows),
            "leakage_min": float(y.min()),
            "leakage_median": float(np.median(y)),
            "leakage_max": float(y.max()),
            "n_le_7e-4": int((y <= 7e-4).sum()),
            "point_O_leakage": float(rows[0]["leakage_final"]),
            "chip_fixed_Table_I": True,
            "target": "leakage_final",
        }
        manifest.append(info)
        print(
            f"  -> {path.name} n={info['n']} "
            f"min={info['leakage_min']:.3e} med={info['leakage_median']:.3e} "
            f"<=7e-4={info['n_le_7e-4']}",
            flush=True,
        )

    readme = OUT_DIR / "README_ML_bundle.md"
    lines = [
        "# Bundle ARFF pulse-only (Appendix E, 6/6/6)",
        "",
        "Demande superviseur : 5–10 fichiers ARFF.",
        "",
        "- **Cible ML** : `leakage_final`",
        "- **Puce** : fixée Table I (`omega_r=7129`, `omega_p=7111`, `g_qr=120`, `J=28.8`)",
        "- **Variables** : `omega_m`, `omega_a`, `sigma`, `T_active`",
        "- **Ligne 1** de chaque fichier : point O expérimental",
        "",
        "| Fichier | Description | n |",
        "|---|---|---|",
    ]
    for m in manifest:
        lines.append(f"| `{m['file']}` | {m['description']} | {m['n']} |")
    lines += [
        "",
        "## Usage Weka",
        "1. Ouvrir un `.arff`",
        "2. Classe = `leakage_final`",
        "3. Algo suggéré : M5Rules (régression)",
        "",
        "Les attributs puce sont présents mais constants (choix méthodologique pulse-only).",
        "On peut les ignorer dans Weka (Remove) si besoin.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nBundle prêt : {OUT_DIR}", flush=True)
    print(f"README : {readme}", flush=True)


if __name__ == "__main__":
    main()
