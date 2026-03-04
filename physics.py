"""
Martin et al. cycling power model and CdA calculation.

Reference:
    Martin, J. C., Milliken, D. L., Cobb, J. E., McFadden, K. L., & Coggan, A. R. (1998).
    Validation of a Mathematical Model for Road Cycling Power.
    Journal of Applied Biomechanics, 14(3), 276-291.

The model predicts total mechanical power as:
    P = v * (F_aero + F_rolling + F_gravity)   [steady-state, flat road]

where:
    F_aero   = 0.5 * rho * CdA * v²
    F_rolling = Crr * m_total * g
    F_gravity = m_total * g * sin(theta)        [zero on flat road]
    Bearing friction is absorbed into a small additive term Vb * m_total * g
"""

import numpy as np
from typing import Dict, Tuple

# ── Physical constants ────────────────────────────────────────────────────────
RHO_AIR    = 1.204          # air density, kg/m³ (15°C, sea level)
GRAVITY    = 9.81           # m/s²
CRR        = 0.0040         # rolling resistance coefficient (racing clincher)
VB         = 0.0044         # bearing friction velocity equivalent, m/s (Martin et al.)

# ── Rider constants (assumed 70 kg, 1.80 m amateur racer) ───────────────────
RIDER_MASS = 70.0           # kg
RIDER_HEIGHT = 1.80         # m
RIDER_WIDTH  = 0.45         # m (shoulder breadth)
RIDER_TORSO_MM = 530.0      # hip-to-shoulder segment length, mm
RIDER_ARM_MM   = 580.0      # shoulder-to-hands segment length (upper + lower arm), mm

# ── Frame parameter layout ───────────────────────────────────────────────────
#  Index  Name                     Unit     Extreme-exploration range
PARAM_NAMES = [
    'head_tube_angle',      # 0   deg from horizontal        [62,  82]
    'seat_tube_angle',      # 1   deg from horizontal        [58,  88]
    'top_tube_length',      # 2   mm (effective/horiz)       [380, 700]
    'head_tube_length',     # 3   mm                         [10,  300]
    'chainstay_length',     # 4   mm                         [340, 480]
    'bb_drop',              # 5   mm (BB below rear axle)    [-30, 130]
    'fork_offset',          # 6   mm (fork rake)             [0,   80]
    'stack',                # 7   mm (BB to HT top, vert)    [350, 700]
    'reach',                # 8   mm (BB to HT top, horiz)   [280, 520]
    'down_tube_aspect',     # 9   width/height ratio         [1.0, 2.667]
    'seat_tube_aspect',     # 10  width/height ratio         [1.0, 2.667]
    'top_tube_aspect',      # 11  width/height ratio         [1.0, 2.667]
    'chainstay_aspect',     # 12  width/height ratio         [1.0, 2.667]
    'frame_weight',         # 13  kg                         [6.8, 9.0]
    'front_wheel_diameter', # 14  mm (independent from rear) [550, 700]
    'rear_wheel_diameter',  # 15  mm (independent from front)[550, 700]
    'handlebar_reach',      # 16  mm, stem extension forward [0,   80]
    'handlebar_drop',       # 17  mm, bar-end below stem     [0,  165]
    'handlebar_width',      # 18  mm, centre-to-centre       [350, 500]
]

# (lower_bound, upper_bound)
PARAM_BOUNDS = [
    (62.0,  82.0),    # head_tube_angle
    (58.0,  88.0),    # seat_tube_angle
    (380.0, 700.0),   # top_tube_length
    (10.0,  300.0),   # head_tube_length
    (340.0, 480.0),   # chainstay_length
    (40.0,  120.0),   # bb_drop
    (0.0,   80.0),    # fork_offset
    (350.0, 650.0),   # stack
    (280.0, 520.0),   # reach
    (1.0,   2.667),   # down_tube_aspect
    (1.0,   2.667),   # seat_tube_aspect
    (1.0,   2.667),   # top_tube_aspect
    (1.0,   2.667),   # chainstay_aspect
    (6.8,   9.0),     # frame_weight
    (550.0, 700.0),   # front_wheel_diameter
    (550.0, 700.0),   # rear_wheel_diameter
    (0.0,   80.0),    # handlebar_reach
    (0.0,  165.0),    # handlebar_drop
    (350.0, 500.0),   # handlebar_width
]

N_PARAMS = len(PARAM_NAMES)


def genome_to_dict(genome: list) -> Dict[str, float]:
    """Convert a DEAP individual (list of floats) to a named parameter dict."""
    return {name: val for name, val in zip(PARAM_NAMES, genome)}


def dict_to_genome(params: Dict[str, float]) -> list:
    """Convert a named parameter dict back to a genome list."""
    return [params[name] for name in PARAM_NAMES]


# ── Rider position model ─────────────────────────────────────────────────────

def _trunk_angle_degrees(stack: float, reach: float) -> float:
    """
    Estimate rider trunk angle from horizontal (degrees) based on
    stack/reach ratio.  Calibrated against empirical wind-tunnel literature:

        stack/reach ~ 1.10  →  trunk ≈ 10°  (TT extreme)
        stack/reach ~ 1.50  →  trunk ≈ 32°  (aggressive road)
        stack/reach ~ 1.70  →  trunk ≈ 46°  (comfortable road)
        stack/reach ~ 2.00  →  trunk ≈ 65°  (upright touring)
    """
    sr = stack / reach
    # Piecewise linear fit through the calibration points above
    trunk = 8.0 + 57.0 * (sr - 1.10) / (2.00 - 1.10)
    return float(np.clip(trunk, 5.0, 80.0))


def _rider_cda(trunk_angle: float) -> float:
    """
    Rider CdA (m²) as a function of trunk angle from horizontal.
    Linear fit through wind-tunnel data (Blocken et al. 2013, Crouch et al. 2014):
        trunk=10° → CdA ≈ 0.210 m²  (very aero)
        trunk=25° → CdA ≈ 0.265 m²  (aggressive road)
        trunk=45° → 0.325 m²         (drops/hoods)
        trunk=70° → 0.410 m²         (upright)
    """
    return 0.190 + 0.00305 * trunk_angle


def _frame_cda(down_tube_aspect: float, seat_tube_aspect: float,
               top_tube_aspect: float, chainstay_aspect: float) -> float:
    """
    Frame CdA contribution (m²).  Aerofoil-section tubes reduce drag versus
    round tubes; effect validated by CFD (Sayers & Stanley 2007).
    Round tubes  (aspect=1.00): frame CdA ≈ 0.030 m²
    Max-legal    (aspect=2.667): frame CdA ≈ 0.019 m²  (≈37 % reduction)
    """
    avg_aspect = (down_tube_aspect + seat_tube_aspect +
                  top_tube_aspect + chainstay_aspect) / 4.0
    # Fraction of maximum aspect gain
    aero_frac = (avg_aspect - 1.0) / (2.667 - 1.0)
    return 0.030 * (1.0 - 0.37 * aero_frac)


def _compute_frame_tilt_deg(params: Dict[str, float]) -> float:
    """
    Frame tilt (degrees) caused by different front and rear wheel sizes.
    Positive = nose up; negative = nose down (more aero rider position).

    A smaller front wheel tilts the frame forward, reducing the effective
    trunk angle and improving aerodynamics.  At 45 km/h over a 1000 mm
    wheelbase, a 550 mm front / 700 mm rear difference gives ≈ −4.3°
    tilt, saving ~15 W.
    """
    r_front = params['front_wheel_diameter'] / 2.0
    r_rear  = params['rear_wheel_diameter']  / 2.0
    wb = max(compute_wheelbase(params), 100.0)   # guard against near-zero WB
    return float(np.degrees(np.arctan2(r_front - r_rear, wb)))


def _compute_hip_xy(params: Dict[str, float]) -> np.ndarray:
    """Hip position (mm), BB = origin. Derived from seat-tube geometry."""
    STA      = np.radians(params['seat_tube_angle'])
    st_dir   = np.array([-np.cos(STA), np.sin(STA)])
    st_top_y = params['stack'] * 0.88
    st_len   = st_top_y / np.sin(STA)
    ST_top   = st_len * st_dir
    return ST_top + 100.0 * st_dir   # 100 mm seatpost above ST_top


def _compute_hands_xy(params: Dict[str, float]) -> np.ndarray:
    """Handlebar end (hands) position (mm), BB = origin."""
    return np.array([
        params['reach'] + params['handlebar_reach'],
        params['stack'] - params['handlebar_drop'],
    ])


def _physical_trunk_angle(params: Dict[str, float]) -> float:
    """
    Trunk angle from horizontal (degrees), capped at 70°.
    Uses the calibrated stack/reach empirical formula (validated against
    wind-tunnel data); result is hard-capped at 70° per design requirements.
    """
    return float(np.clip(_trunk_angle_degrees(params['stack'], params['reach']), 5.0, 70.0))


def _compute_shoulder_xy(params: Dict[str, float]) -> np.ndarray:
    """
    Shoulder position (mm), BB = origin, via forward kinematics.
    Extends RIDER_TORSO_MM backward-and-upward from hip along the trunk direction.
    """
    trunk_rad = np.radians(_physical_trunk_angle(params))
    hip       = _compute_hip_xy(params)
    # Trunk direction: backward (−x) and upward (+y) from hip
    trunk_dir = np.array([-np.cos(trunk_rad), np.sin(trunk_rad)])
    return hip + RIDER_TORSO_MM * trunk_dir


def calculate_cda(params: Dict[str, float]) -> float:
    """
    Total CdA (m²) for the frame + rider system at race speed.
    CdA = Cd × A, where A is projected frontal area and Cd is drag coefficient.

    Rider trunk angle is derived from the stack/reach ratio and adjusted for
    any frame tilt caused by different front and rear wheel diameters.
    """
    trunk_base = _physical_trunk_angle(params)
    frame_tilt = _compute_frame_tilt_deg(params)
    trunk = float(np.clip(trunk_base + frame_tilt, 5.0, 70.0))
    cda_rider = _rider_cda(trunk)
    cda_frame = _frame_cda(
        params['down_tube_aspect'], params['seat_tube_aspect'],
        params['top_tube_aspect'],  params['chainstay_aspect'],
    )
    return cda_rider + cda_frame


# ── Martin et al. power model ────────────────────────────────────────────────

def martin_power(
    cda: float,
    total_mass_kg: float,
    velocity_ms: float,
    grade: float = 0.0,
    wind_speed_ms: float = 0.0,
    rho: float = RHO_AIR,
) -> float:
    """
    Predict required mechanical power (W) using Martin et al. (1998).

    Args:
        cda:            Drag area, m²
        total_mass_kg:  Combined rider + bike mass, kg
        velocity_ms:    Ground speed, m/s
        grade:          Road gradient (decimal, e.g. 0.03 = 3 %)
        wind_speed_ms:  Headwind speed, m/s (positive = headwind)
        rho:            Air density, kg/m³

    Returns:
        Mechanical power in Watts.
    """
    velocity_ms = max(velocity_ms, 0.1)          # guard against division by zero
    v_air = velocity_ms + wind_speed_ms          # air speed
    theta = np.arctan(grade)

    F_aero    = 0.5 * rho * cda * v_air ** 2
    F_rolling = (CRR + VB / velocity_ms) * total_mass_kg * GRAVITY * np.cos(theta)
    F_gravity = total_mass_kg * GRAVITY * np.sin(theta)

    return (F_aero + F_rolling + F_gravity) * velocity_ms


def velocity_from_power(
    power_w: float,
    cda: float,
    total_mass_kg: float,
    grade: float = 0.0,
    rho: float = RHO_AIR,
    v_guess: float = 12.0,
) -> float:
    """
    Numerically invert Martin et al. model to find velocity (m/s) at given power.
    Uses Newton-Raphson iteration.
    """
    v = v_guess
    for _ in range(50):
        P_calc = martin_power(cda, total_mass_kg, v, grade, 0.0, rho)
        # Derivative dP/dv = F_aero * 3 + F_rolling_v + F_gravity_v (approx)
        dP_dv = (martin_power(cda, total_mass_kg, v + 0.01, grade, 0.0, rho) -
                 P_calc) / 0.01
        if abs(dP_dv) < 1e-9:
            break
        v_new = v - (P_calc - power_w) / dP_dv
        v_new = max(1.0, v_new)
        if abs(v_new - v) < 1e-6:
            break
        v = v_new
    return v


def compute_wheelbase(params: Dict[str, float], fork_length_mm: float = 370.0) -> float:
    """
    Derive wheelbase (mm) from frame geometry.

    Standard 2D derivation:
        front_center = reach + (ht_len + fork_length) * cos(HTA) + fork_offset * sin(HTA)
        wheelbase    = front_center + chainstay_length
    where HTA is measured from horizontal.
    """
    HTA = np.radians(params['head_tube_angle'])
    front_center = (
        params['reach']
        + (params['head_tube_length'] + fork_length_mm) * np.cos(HTA)
        + params['fork_offset'] * np.sin(HTA)
    )
    return front_center + params['chainstay_length']
