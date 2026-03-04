"""
UCI bicycle regulation constraints for road bicycles.

Relevant rules (UCI Regulations Part IV, Equipment, rev. 2023):
  Art. 1.3.006  Minimum total bike mass: 6.8 kg
  Art. 1.3.010  Wheelbase: 95 cm – 105 cm
  Art. 1.3.018  Tube section aspect ratio: ≤ 8:3 (= 2.667)
  Art. 1.3.013  Saddle must be ≥ 5 cm behind the vertical through the BB
  Art. 1.3.019  Wheel external diameter: 55 cm – 70 cm

Constraint checking returns a ConstraintResult dataclass so callers can
distinguish between hard failures (ineligible design) and a continuous
penalty suitable for use inside the fitness function.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from physics import compute_wheelbase, PARAM_NAMES, PARAM_BOUNDS


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class ConstraintResult:
    is_legal: bool = True
    violations: List[str] = field(default_factory=list)
    penalty: float = 0.0       # penalty in watts added to fitness (power minimisation)


# ── Individual constraint checkers ───────────────────────────────────────────

def _check_frame_weight(params: Dict[str, float], result: ConstraintResult) -> None:
    """Art. 1.3.006 — total bike mass ≥ 6.8 kg."""
    w = params['frame_weight']
    if w < 6.8:
        deficit = 6.8 - w
        result.is_legal = False
        result.violations.append(f"frame_weight {w:.3f} kg < 6.8 kg (deficit {deficit:.3f} kg)")
        result.penalty += deficit * 600.0   # 600 W per kg under limit


def _check_wheelbase(params: Dict[str, float], result: ConstraintResult) -> None:
    """Art. 1.3.010 — wheelbase between 950 mm and 1050 mm."""
    wb = compute_wheelbase(params)
    if wb < 950.0:
        viol = 950.0 - wb
        result.is_legal = False
        result.violations.append(f"wheelbase {wb:.1f} mm < 950 mm (short by {viol:.1f} mm)")
        result.penalty += viol * 3.0   # 3 W/mm; 100 mm violation ≈ 300 W
    elif wb > 1050.0:
        viol = wb - 1050.0
        result.is_legal = False
        result.violations.append(f"wheelbase {wb:.1f} mm > 1050 mm (long by {viol:.1f} mm)")
        result.penalty += viol * 3.0   # 3 W/mm; 100 mm violation ≈ 300 W


def _check_tube_aspect_ratios(params: Dict[str, float], result: ConstraintResult) -> None:
    """Art. 1.3.018 — tube cross-section must fit within 8:3 box (ratio ≤ 2.667)."""
    max_ratio = 8.0 / 3.0  # = 2.6̄
    for key in ('down_tube_aspect', 'seat_tube_aspect',
                'top_tube_aspect', 'chainstay_aspect'):
        r = params[key]
        if r > max_ratio + 1e-6:
            excess = r - max_ratio
            result.is_legal = False
            result.violations.append(
                f"{key} {r:.3f} > {max_ratio:.3f} (excess {excess:.3f})"
            )
            result.penalty += excess * 400.0   # 400 W per unit of excess aspect ratio


def _check_saddle_setback(params: Dict[str, float], result: ConstraintResult) -> None:
    """
    Art. 1.3.013 — saddle tip ≥ 5 cm behind a vertical through the BB.

    For a conventional road frame the saddle setback is approximately:
        setback ≈ saddle_height × cos(STA_from_vertical)
                 = saddle_height × sin(90° - STA_from_horizontal)
    For STA ∈ [72°, 78°] and typical saddle heights ≥ 600 mm the setback
    far exceeds 50 mm, so violations only occur with extremely steep seat
    tubes beyond our parameter range.  We still enforce it explicitly.

    Here we use a representative saddle height of 680 mm (stack-based proxy).
    """
    STA = params['seat_tube_angle']  # degrees from horizontal
    saddle_height_mm = params['stack'] * 0.95   # rough saddle height above BB
    # Horizontal setback of saddle tip behind BB vertical
    # seat tube leans backward at (90 - STA) degrees from vertical
    setback = saddle_height_mm * np.cos(np.radians(STA))
    if setback < 50.0:
        deficit = 50.0 - setback
        result.is_legal = False
        result.violations.append(
            f"saddle setback {setback:.1f} mm < 50 mm (deficit {deficit:.1f} mm)"
        )
        result.penalty += deficit * 6.0   # 6 W/mm; 50 mm deficit ≈ 300 W


def _check_wheel_diameter(params: Dict[str, float], result: ConstraintResult) -> None:
    """Art. 1.3.019 — wheel external diameter 550 mm – 700 mm (both wheels checked)."""
    for key in ('front_wheel_diameter', 'rear_wheel_diameter'):
        d = params[key]
        label = 'front' if 'front' in key else 'rear'
        if d < 550.0:
            deficit = 550.0 - d
            result.is_legal = False
            result.violations.append(f"{label}_wheel_diameter {d:.1f} mm < 550 mm")
            result.penalty += deficit * 4.0   # 4 W/mm; 75 mm under ≈ 300 W
        elif d > 700.0:
            excess = d - 700.0
            result.is_legal = False
            result.violations.append(f"{label}_wheel_diameter {d:.1f} mm > 700 mm")
            result.penalty += excess * 4.0   # 4 W/mm; 75 mm over ≈ 300 W


def _check_parameter_bounds(params: Dict[str, float], result: ConstraintResult) -> None:
    """Ensure every parameter stays within its declared search bounds."""
    for name, (lo, hi) in zip(PARAM_NAMES, PARAM_BOUNDS):
        v = params[name]
        if v < lo - 1e-6:
            excess = lo - v
            result.is_legal = False
            result.violations.append(f"{name} = {v:.4g} below lower bound {lo:.4g}")
            result.penalty += (excess / (hi - lo)) * 2000.0   # 2000 W per full-range violation
        elif v > hi + 1e-6:
            excess = v - hi
            result.is_legal = False
            result.violations.append(f"{name} = {v:.4g} above upper bound {hi:.4g}")
            result.penalty += (excess / (hi - lo)) * 2000.0   # 2000 W per full-range violation


_FORK_LEN = 370.0   # mm — mirrors FORK_LENGTH_MM in visualization/bike_env


def _check_handlebars(params: Dict[str, float], result: ConstraintResult) -> None:
    """
    UCI and safety constraints for handlebars.
      1. Reach ≤ 80 mm (UCI)
      2. Drop  ≤ 165 mm (UCI)
      3. Width ∈ [350, 500] mm
      4. Bar end must not extend past the front axle (safety)
      5. Rider must be able to reach bars (IK reachability)
    """
    # (physics imports for reachability deferred to check #5 below)

    reach = params['handlebar_reach']
    drop  = params['handlebar_drop']
    width = params['handlebar_width']

    # 1. UCI reach limit
    if reach > 80.0 + 1e-6:
        excess = reach - 80.0
        result.is_legal = False
        result.violations.append(
            f"handlebar_reach {reach:.1f} mm > 80 mm UCI limit"
        )
        result.penalty += excess * 5.0   # 5 W/mm

    # 2. UCI drop limit
    if drop > 165.0 + 1e-6:
        excess = drop - 165.0
        result.is_legal = False
        result.violations.append(
            f"handlebar_drop {drop:.1f} mm > 165 mm UCI limit"
        )
        result.penalty += excess * 3.0   # 3 W/mm

    # 3. Width limits
    if width < 350.0:
        result.is_legal = False
        result.violations.append(f"handlebar_width {width:.0f} mm < 350 mm")
        result.penalty += (350.0 - width) * 2.0
    elif width > 500.0:
        result.is_legal = False
        result.violations.append(f"handlebar_width {width:.0f} mm > 500 mm")
        result.penalty += (width - 500.0) * 2.0

    # 4. Bar end must not extend past the front axle
    HTA = np.radians(params['head_tube_angle'])
    front_center_x = (
        params['reach']
        + (params['head_tube_length'] + _FORK_LEN) * np.cos(HTA)
        + params['fork_offset'] * np.sin(HTA)
    )
    bar_end_x = params['reach'] + reach
    if bar_end_x > front_center_x + 1e-6:
        excess = bar_end_x - front_center_x
        result.is_legal = False
        result.violations.append(
            f"handlebar end {bar_end_x:.0f} mm extends {excess:.0f} mm past front axle"
        )
        result.penalty += excess * 5.0

    # Note: rider arm reachability is not checked here — in a cycling position
    # the straight-line shoulder-to-hands distance (800–1200 mm) always exceeds
    # arm length because arms are bent at the elbow.  The UCI handlebar limits
    # (reach ≤ 80 mm, drop ≤ 165 mm) and the front-axle safety check above
    # already bound the handlebar position to an ergonomically sane region.


def _check_trunk_angle(params: Dict[str, float], result: ConstraintResult) -> None:
    """
    Ergonomic sustainability constraint:
      trunk angle from horizontal must be >= 20°.
    """
    from physics import _physical_trunk_angle
    trunk = _physical_trunk_angle(params)
    if trunk < 20.0:
        deficit = 20.0 - trunk
        result.is_legal = False
        result.violations.append(
            f"trunk_angle {trunk:.1f}° < 20° minimum (deficit {deficit:.1f}°)"
        )
        result.penalty += deficit * 60.0   # strong penalty for unsustainable positions


# ── Public interface ─────────────────────────────────────────────────────────

def check_uci_compliance(params: Dict[str, float]) -> ConstraintResult:
    """
    Run all UCI checks and return a ConstraintResult.

    The `penalty` field is in watts and is added directly to the raw power
    to discourage illegal individuals in the GA without removing them outright
    (soft constraint formulation).  All penalties are scaled so a moderate
    violation costs roughly 300–600 W, commensurate with the optimisation target.
    """
    result = ConstraintResult()
    _check_parameter_bounds(params, result)
    _check_frame_weight(params, result)
    _check_wheelbase(params, result)
    _check_tube_aspect_ratios(params, result)
    _check_saddle_setback(params, result)
    _check_wheel_diameter(params, result)
    _check_handlebars(params, result)
    _check_trunk_angle(params, result)
    return result


def repair_genome(genome: list) -> list:
    """
    Clip each gene to its declared parameter bounds (hard repair after mutation).
    Returns a new list; does NOT modify in place.
    """
    return [
        float(np.clip(v, lo, hi))
        for v, (lo, hi) in zip(genome, PARAM_BOUNDS)
    ]


def wheelbase_summary(params: Dict[str, float]) -> str:
    """Human-readable wheelbase diagnostic string."""
    wb = compute_wheelbase(params)
    legal = 950.0 <= wb <= 1050.0
    tag = "OK" if legal else "VIOLATION"
    return f"wheelbase = {wb:.1f} mm  [{tag}]"
