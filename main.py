"""
Bicycle Frame Geometry Optimiser
=================================
Genetic algorithm that evolves UCI-legal road bicycle frame geometries to
minimise aerodynamic drag (CdA) at 45 km/h using:

    • OpenAI Gymnasium  — simulation environment wrapper
    • DEAP              — genetic algorithm engine
    • Martin et al. (1998) cycling power model — aerodynamic physics

Usage:
    python main.py                    # run 100 generations, pop=50
    python main.py --generations 200  # longer run
    python main.py --seed 42          # reproducible run
    python main.py --no-show          # save plots without displaying

Outputs (in the current working directory):
    winning_frame.json    — full parameter set of the best design
    winning_frame.png     — side-profile matplotlib visualisation
    convergence.png       — CdA vs generation curve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

# Add project directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from physics import (
    PARAM_NAMES, genome_to_dict,
    calculate_cda, martin_power, compute_wheelbase,
    RIDER_MASS,
)
from constraints import check_uci_compliance
from bike_env import BikeFrameEnv, evaluate_frame
from ga_optimizer import run_evolution
from visualization import plot_frame, plot_convergence


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evolve UCI-legal bicycle frames to minimise aerodynamic CdA'
    )
    parser.add_argument('--generations', type=int, default=1000,
                        help='Number of GA generations (default: 1000)')
    parser.add_argument('--population', type=int, default=200,
                        help='Population size (default: 200)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--cx-prob', type=float, default=0.7,
                        help='Crossover probability (default: 0.7)')
    parser.add_argument('--mut-prob', type=float, default=0.3,
                        help='Mutation probability (default: 0.3)')
    parser.add_argument('--no-show', action='store_true',
                        help='Save plots without calling plt.show()')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output files (default: current dir)')
    return parser.parse_args()


# ── Reporting helpers ─────────────────────────────────────────────────────────

def print_banner() -> None:
    print("=" * 70)
    print("  BICYCLE FRAME GEOMETRY OPTIMISER")
    print("  Genetic Algorithm using DEAP + OpenAI Gymnasium")
    print("  Physics: Martin et al. (1998) Road Cycling Power Model")
    print("  Objective: Minimise power at 45 km/h — UCI legal designs only")
    print("=" * 70)
    print()


def print_winning_design(params: dict, cda: float) -> None:
    wb    = compute_wheelbase(params)
    cr    = check_uci_compliance(params)
    power = martin_power(cda, RIDER_MASS + params['frame_weight'], 12.5)  # raw CdA

    print()
    print("=" * 70)
    print("  WINNING DESIGN PARAMETERS")
    print("=" * 70)
    for name in PARAM_NAMES:
        unit = _param_unit(name)
        print(f"  {name:<25s}  {params[name]:>10.4f}  {unit}")
    print()
    print(f"  Derived metrics:")
    print(f"    CdA             = {cda:.4f} m²")
    print(f"    Power @ 45 km/h = {power:.1f} W")
    print(f"    Wheelbase       = {wb:.1f} mm")
    f_wd = params['front_wheel_diameter']
    r_wd = params['rear_wheel_diameter']
    wheel_diff = abs(f_wd - r_wd)
    wheel_str = f"{f_wd:.0f}mm F / {r_wd:.0f}mm R"
    if wheel_diff > 50:
        wheel_str += f"  ★ ASYMMETRIC (+{wheel_diff:.0f}mm)"
    print(f"    Wheels          = {wheel_str}")
    legal = 'YES' if cr.is_legal else 'NO'
    print(f"    UCI Legal       = {legal}")
    if cr.violations:
        for v in cr.violations:
            print(f"      VIOLATION: {v}")
    print()

    # Put the "interesting" insight — what makes this non-intuitive
    from physics import _trunk_angle_degrees
    trunk = _trunk_angle_degrees(params['stack'], params['reach'])
    sr    = params['stack'] / params['reach']
    print("  Key design insights:")
    print(f"    Stack/Reach ratio = {sr:.3f}  →  trunk angle ≈ {trunk:.1f}°")
    print(f"    Avg tube aspect   = "
          f"{(params['down_tube_aspect']+params['seat_tube_aspect']+params['top_tube_aspect']+params['chainstay_aspect'])/4:.3f}"
          f"  (1.0=round, 2.67=max aero)")
    if sr < 1.35:
        print("    → Exceptionally aggressive position; most engineers would "
              "consider this too extreme for road riding.")
    if params['head_tube_length'] < 100:
        print("    → Extremely short head tube forces a very low front end.")
    avg_aspect = (params['down_tube_aspect'] + params['seat_tube_aspect'] +
                  params['top_tube_aspect'] + params['chainstay_aspect']) / 4
    if avg_aspect > 2.4:
        print("    → All tubes pushed to the UCI aero-section limit —")
        print("      a design optimisation humans rarely apply consistently.")
    print(f"    Handlebar reach = {params['handlebar_reach']:.0f} mm  "
          f"drop = {params['handlebar_drop']:.0f} mm  "
          f"width = {params['handlebar_width']:.0f} mm")
    print()


def _param_unit(name: str) -> str:
    if 'angle' in name:
        return 'deg'
    if 'aspect' in name:
        return 'ratio'
    if 'weight' in name:
        return 'kg'
    if 'diameter' in name:
        return 'mm'
    return 'mm'


# ── Output file writers ───────────────────────────────────────────────────────

def save_json(params: dict, cda: float, generation: int, outdir: str) -> str:
    """Save winning design to JSON."""
    wb    = compute_wheelbase(params)
    power = martin_power(cda, RIDER_MASS + params['frame_weight'], 12.5)  # raw CdA
    cr    = check_uci_compliance(params)
    from physics import _trunk_angle_degrees
    trunk = _trunk_angle_degrees(params['stack'], params['reach'])

    data = {
        "meta": {
            "description": "Evolved UCI-legal bicycle frame — minimum power @ 45 km/h (extreme exploration mode)",
            "model":        "Martin et al. (1998) Road Cycling Power",
            "generation":   generation,
            "uci_legal":    cr.is_legal,
            "violations":   cr.violations,
        },
        "performance": {
            "CdA_m2":                round(cda, 6),
            "power_at_45kmh_W":      round(power, 1),
            "trunk_angle_deg":       round(trunk, 2),
            "wheelbase_mm":          round(wb, 2),
            "stack_reach_ratio":     round(params['stack'] / params['reach'], 4),
            "wheel_asymmetry_mm":    round(abs(params['front_wheel_diameter'] - params['rear_wheel_diameter']), 1),
            "handlebar_reach_mm":    round(params['handlebar_reach'], 1),
            "handlebar_drop_mm":     round(params['handlebar_drop'], 1),
            "handlebar_width_mm":    round(params['handlebar_width'], 1),
        },
        "frame_parameters": {
            name: round(params[name], 4) for name in PARAM_NAMES
        },
        "parameter_units": {
            name: _param_unit(name) for name in PARAM_NAMES
        },
    }

    path = os.path.join(outdir, 'winning_frame.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Winning design saved to: {path}")
    return path


# ── Gymnasium environment smoke-test ─────────────────────────────────────────

def env_smoke_test() -> None:
    """Verify the Gymnasium env works before starting the GA."""
    print("  Smoke-testing BikeFrameEnv ... ", end='', flush=True)
    env = BikeFrameEnv(render_mode=None)
    obs, info = env.reset(seed=0)
    assert obs.shape == (19,), f"unexpected obs shape {obs.shape}"
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert terminated, "env should terminate after one step"
    assert 'cda' in info2, "info dict missing 'cda'"
    assert 'power_at_45kmh' in info2, "info dict missing 'power_at_45kmh'"
    env.close()
    print(f"OK  (sample CdA = {info2['cda']:.4f} m²  P = {info2['power_at_45kmh']:.1f} W)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    show_plots = not args.no_show

    print_banner()
    env_smoke_test()

    print()
    print(f"  Starting evolution:")
    print(f"    Generations : {args.generations}")
    print(f"    Population  : {args.population}")
    print(f"    Seed        : {args.seed if args.seed is not None else 'random'}")
    print(f"    Cx prob     : {args.cx_prob}")
    print(f"    Mut prob    : {args.mut_prob}")
    print()
    print("-" * 70)

    t0 = time.time()
    population, logbook, best_individual = run_evolution(
        n_generations  = args.generations,
        pop_size       = args.population,
        cx_prob        = args.cx_prob,
        mut_prob       = args.mut_prob,
        seed           = args.seed,
        verbose        = True,
    )
    elapsed = time.time() - t0

    print("-" * 70)
    print(f"\n  Evolution complete in {elapsed:.1f} s")

    # ── Retrieve best design ──────────────────────────────────────────────────
    best_params   = genome_to_dict(list(best_individual))
    best_fitness  = best_individual.fitness.values[0]   # penalised power (W)
    raw_cda       = calculate_cda(best_params)           # raw CdA for display

    print_winning_design(best_params, raw_cda)

    # ── Save outputs ──────────────────────────────────────────────────────────
    json_path = save_json(best_params, raw_cda, args.generations, args.output_dir)

    frame_png = os.path.join(args.output_dir, 'winning_frame.png')
    plot_frame(
        params     = best_params,
        generation = args.generations,
        fitness    = raw_cda,
        save_path  = frame_png,
        show       = show_plots,
    )

    conv_png = os.path.join(args.output_dir, 'convergence.png')
    plot_convergence(logbook, save_path=conv_png, show=show_plots)

    print()
    print("  Output files:")
    print(f"    {json_path}")
    print(f"    {frame_png}")
    print(f"    {conv_png}")
    print()


if __name__ == '__main__':
    main()
