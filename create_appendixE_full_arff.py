"""
Génère un ARFF Weka pulse-only (puce Table I fixe).

- Point O expérimental inclus en 1re ligne (cible ~6e-4)
- Variables : omega_m, omega_a, sigma, T_active
- Fixes : omega_r, omega_p, g_qr, J (Table I)
"""

from __future__ import annotations

import argparse
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
    target_residual_leakage,
)
from sim_lru_appendixE import AppendixEParams, resonance_detuning_rad_s, run_lru_appendixE


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def sample_pulse(rng: np.random.Generator, mode: str) -> dict[str, float]:
    oa0 = omega_a_Hz / 1e6
    om0 = omega_m_Hz / 1e6
    t0 = pulse_duration_active_ns
    if mode == "near_o":
        oa = clip(rng.normal(oa0, 3.0), 115.0, 140.0)
        om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0.0, 1.5), 545.0, 585.0)
        sig = clip(rng.normal(sigma_gaussian_ns, 0.6), 3.5, 7.0)
        tact = clip(rng.normal(t0, 3.0), 28.0, 42.0)
    elif mode == "resonance":
        oa = clip(rng.normal(oa0, 12.0), 100.0, 155.0)
        om = clip(om0 + 0.5 * (oa - oa0) + rng.normal(0.0, 2.0), 525.0, 600.0)
        sig = clip(rng.normal(sigma_gaussian_ns, 1.0), 3.0, 8.0)
        tact = clip(rng.normal(t0, 6.0), 22.0, 50.0)
    else:
        oa = clip(rng.uniform(100.0, 160.0), 100.0, 160.0)
        om = clip(rng.uniform(530.0, 600.0), 530.0, 600.0)
        sig = clip(rng.uniform(3.0, 8.0), 3.0, 8.0)
        tact = clip(rng.uniform(22.0, 50.0), 22.0, 50.0)
    return {
        "omega_m_MHz": om,
        "omega_a_MHz": oa,
        "sigma_ns": sig,
        "T_active_ns": tact,
        "omega_r_MHz": omega_ro_Hz / 1e6,
        "omega_p_MHz": omega_p_Hz / 1e6,
        "g_qr_MHz": g_qr_Hz / 1e6,
        "J_MHz": J_Hz / 1e6,
    }


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


def write_arff(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("@RELATION appendixE_666_pointO_pulse_only\n\n")
        for a in ATTRS:
            f.write(f"@ATTRIBUTE {a} NUMERIC\n")
        f.write("\n@DATA\n")
        for r in rows:
            f.write(",".join(f"{float(r[a]):.10g}" for a in ATTRS) + "\n")


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_checkpoint(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def eval_row(cand: dict[str, float], n_points: int) -> dict:
    params = AppendixEParams(
        omega_a_MHz=cand["omega_a_MHz"],
        omega_m_MHz=cand["omega_m_MHz"],
        sigma_ns=cand["sigma_ns"],
        t_active_ns=cand["T_active_ns"],
        omega_r_MHz=cand["omega_r_MHz"],
        omega_p_MHz=cand["omega_p_MHz"],
        g_qr_MHz=cand["g_qr_MHz"],
        J_MHz=cand["J_MHz"],
        n_points=n_points,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--n-points", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="qutip_param_dataset_appendixE_full.arff")
    ap.add_argument("--checkpoint", default="appendixE_666_checkpoint.jsonl")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out = root / args.output
    ckpt = root / args.checkpoint
    if args.fresh and ckpt.exists():
        ckpt.unlink()

    rng = np.random.default_rng(args.seed)
    rows = load_checkpoint(ckpt)
    print(f"Checkpoint : {len(rows)} lignes", flush=True)

    if not rows:
        print("Validation point O (pulse-only)...", flush=True)
        row = eval_row(
            {
                "omega_m_MHz": omega_m_Hz / 1e6,
                "omega_a_MHz": omega_a_Hz / 1e6,
                "sigma_ns": sigma_gaussian_ns,
                "T_active_ns": pulse_duration_active_ns,
                "omega_r_MHz": omega_ro_Hz / 1e6,
                "omega_p_MHz": omega_p_Hz / 1e6,
                "g_qr_MHz": g_qr_Hz / 1e6,
                "J_MHz": J_Hz / 1e6,
            },
            args.n_points,
        )
        rows.append(row)
        append_checkpoint(ckpt, row)
        print(
            f"  point O leakage={row['leakage_final']:.6e} "
            f"(cible ~6e-4 / seuil {target_residual_leakage:.1e})",
            flush=True,
        )

    print("Grille locale autour du point O...", flush=True)
    oa0, om0 = omega_a_Hz / 1e6, omega_m_Hz / 1e6
    for oa in np.linspace(120.0, 136.0, 5):
        for dom in (-2.0, -1.0, 0.0, 1.0, 2.0):
            om = om0 + 0.5 * (oa - oa0) + dom
            for tact in (30.0, 32.0, 34.5, 37.0, 40.0):
                for sig in (4.0, 5.0, 6.0):
                    if len(rows) >= args.n_samples:
                        break
                    key = (round(om, 4), round(oa, 4), round(sig, 4), round(tact, 4))
                    if any(
                        abs(r["omega_m_MHz"] - key[0]) < 1e-6
                        and abs(r["omega_a_MHz"] - key[1]) < 1e-6
                        and abs(r["sigma_ns"] - key[2]) < 1e-6
                        and abs(r["T_active_ns"] - key[3]) < 1e-6
                        for r in rows
                    ):
                        continue
                    try:
                        row = eval_row(
                            {
                                "omega_m_MHz": float(om),
                                "omega_a_MHz": float(oa),
                                "sigma_ns": float(sig),
                                "T_active_ns": float(tact),
                                "omega_r_MHz": omega_ro_Hz / 1e6,
                                "omega_p_MHz": omega_p_Hz / 1e6,
                                "g_qr_MHz": g_qr_Hz / 1e6,
                                "J_MHz": J_Hz / 1e6,
                            },
                            args.n_points,
                        )
                        rows.append(row)
                        append_checkpoint(ckpt, row)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[skip] {type(exc).__name__}: {exc}", flush=True)
                if len(rows) >= args.n_samples:
                    break
            if len(rows) >= args.n_samples:
                break
        if len(rows) >= args.n_samples:
            break
    print(f"  après grille : n={len(rows)}", flush=True)
    write_arff(out, rows)

    n_near = int(0.35 * args.n_samples)
    n_res = int(0.25 * args.n_samples)
    i0 = len(rows)
    while len(rows) < args.n_samples:
        i = len(rows) - i0 + 1
        if i <= n_near:
            mode = "near_o"
        elif i <= n_near + n_res:
            mode = "resonance"
        else:
            mode = "explore"
        try:
            row = eval_row(sample_pulse(rng, mode), args.n_points)
            rows.append(row)
            append_checkpoint(ckpt, row)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {type(exc).__name__}: {exc}", flush=True)
        if len(rows) % 25 == 0:
            ys = [r["leakage_final"] for r in rows]
            write_arff(out, rows)
            print(
                f"[{len(rows)}/{args.n_samples}] min={min(ys):.3e} "
                f"median={np.median(ys):.3e} n<=7e-4={sum(y <= 7e-4 for y in ys)} "
                f"mode={mode}",
                flush=True,
            )

    write_arff(out, rows)
    y = np.array([r["leakage_final"] for r in rows])
    print("\n=== Dataset pulse-only ===", flush=True)
    print(f"ARFF : {out.resolve()}", flush=True)
    print(f"n={len(rows)} min={y.min():.4e} median={np.median(y):.4e} max={y.max():.4e}", flush=True)
    print(f"<=7e-4 : {(y <= 7e-4).sum()} ({100 * (y <= 7e-4).mean():.1f}%)", flush=True)
    print(f"Point O : {rows[0]['leakage_final']:.6e}", flush=True)


if __name__ == "__main__":
    main()
