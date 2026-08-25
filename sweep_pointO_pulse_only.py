from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from params_appendixE import J_Hz, g_qr_Hz, omega_p_Hz, omega_ro_Hz
from sim_lru_appendixE import AppendixEParams, run_lru_appendixE


@dataclass(frozen=True)
class Best:
    leakage_final: float
    omega_m_MHz: float
    omega_a_MHz: float
    sigma_ns: float
    t_active_ns: float
    min_leakage: float
    min_leakage_time_ns: float


def frange(center: float, half_span: float, step: float) -> list[float]:
    n = int(np.floor((2.0 * half_span) / step + 0.5))
    start = center - 0.5 * n * step
    return [start + i * step for i in range(n + 1)]


def iter_grid(
    omega_m_vals: Iterable[float],
    omega_a_vals: Iterable[float],
    sigma_vals: Iterable[float],
    t_active_vals: Iterable[float],
):
    yield from product(omega_m_vals, omega_a_vals, sigma_vals, t_active_vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep pulse-only autour du point O (puce fixée Table I)."
    )
    ap.add_argument("--n-points", type=int, default=30)
    ap.add_argument("--omega-m-center", type=float, default=564.0)
    ap.add_argument("--omega-a-center", type=float, default=128.0)
    ap.add_argument("--sigma-center", type=float, default=5.0)
    ap.add_argument("--t-active-center", type=float, default=34.5)
    ap.add_argument("--omega-m-halfspan", type=float, default=20.0)
    ap.add_argument("--omega-a-halfspan", type=float, default=25.0)
    ap.add_argument("--sigma-halfspan", type=float, default=3.0)
    ap.add_argument("--t-active-halfspan", type=float, default=18.0)
    ap.add_argument("--omega-m-step", type=float, default=2.0)
    ap.add_argument("--omega-a-step", type=float, default=2.0)
    ap.add_argument("--sigma-step", type=float, default=0.5)
    ap.add_argument("--t-active-step", type=float, default=2.0)
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    omega_r_MHz = omega_ro_Hz / 1e6
    omega_p_MHz = omega_p_Hz / 1e6
    J_MHz = J_Hz / 1e6
    g_MHz = g_qr_Hz / 1e6

    omega_m_vals = frange(args.omega_m_center, args.omega_m_halfspan, args.omega_m_step)
    omega_a_vals = frange(args.omega_a_center, args.omega_a_halfspan, args.omega_a_step)
    sigma_vals = frange(args.sigma_center, args.sigma_halfspan, args.sigma_step)
    t_active_vals = frange(
        args.t_active_center, args.t_active_halfspan, args.t_active_step
    )

    best: Best | None = None
    failures = 0

    for i, (omega_m, omega_a, sigma, t_active) in enumerate(
        iter_grid(omega_m_vals, omega_a_vals, sigma_vals, t_active_vals), start=1
    ):
        try:
            params = AppendixEParams(
                omega_m_MHz=float(omega_m),
                omega_a_MHz=float(omega_a),
                sigma_ns=float(sigma),
                t_active_ns=float(t_active),
                omega_r_MHz=float(omega_r_MHz),
                omega_p_MHz=float(omega_p_MHz),
                J_MHz=float(J_MHz),
                g_qr_MHz=float(g_MHz),
                n_points=int(args.n_points),
            )
            res = run_lru_appendixE(params)
            leak = float(res["leakage_final"])

            if best is None or leak < best.leakage_final:
                best = Best(
                    leakage_final=leak,
                    omega_m_MHz=float(omega_m),
                    omega_a_MHz=float(omega_a),
                    sigma_ns=float(sigma),
                    t_active_ns=float(t_active),
                    min_leakage=float(res["min_leakage"]),
                    min_leakage_time_ns=float(res["min_leakage_time_ns"]),
                )
                print(
                    "BEST",
                    f"leakage_final={best.leakage_final:.6e}",
                    f"omega_m={best.omega_m_MHz:.2f}MHz",
                    f"omega_a={best.omega_a_MHz:.2f}MHz",
                    f"sigma={best.sigma_ns:.2f}ns",
                    f"t_active={best.t_active_ns:.2f}ns",
                    f"min_leakage={best.min_leakage:.6e}",
                    f"tmin={best.min_leakage_time_ns:.2f}ns",
                    flush=True,
                )

            if args.print_every > 0 and (i % args.print_every == 0):
                print(f"[{i}] leak={leak:.6e} failures={failures}", flush=True)
        except Exception:
            failures += 1
            if args.print_every > 0 and (i % args.print_every == 0):
                print(f"[{i}] FAILED failures={failures}", flush=True)

    total = len(omega_m_vals) * len(omega_a_vals) * len(sigma_vals) * len(t_active_vals)
    print("DONE", f"total={total}", f"failures={failures}", flush=True)
    if best is None:
        raise SystemExit(2)

    print(
        "BEST_FINAL",
        f"leakage_final={best.leakage_final:.6e}",
        f"omega_m={best.omega_m_MHz:.2f}MHz",
        f"omega_a={best.omega_a_MHz:.2f}MHz",
        f"sigma={best.sigma_ns:.2f}ns",
        f"t_active={best.t_active_ns:.2f}ns",
        f"min_leakage={best.min_leakage:.6e}",
        f"tmin={best.min_leakage_time_ns:.2f}ns",
        flush=True,
    )


if __name__ == "__main__":
    main()

