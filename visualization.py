"""
Matplotlib side-profile visualisation of an evolved bicycle frame geometry.

The frame is rendered as a precise 2D silhouette using the parameter geometry:
    • Both wheels (circles)
    • Fork (two parallel lines = fork blades)
    • Head tube, top tube, down tube, seat tube, chainstays, seatstays
    • Bottom bracket shell
    • Saddle (simplified horizontal bar)
    • Rider silhouette (basic stick figure at derived trunk angle)

All coordinates are in millimetres, x = forward, y = up, BB at origin.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle, FancyArrowPatch
from typing import Dict, Tuple

from physics import (
    PARAM_NAMES, compute_wheelbase,
    calculate_cda, martin_power, RIDER_MASS,
    _trunk_angle_degrees, _physical_trunk_angle,
    RIDER_TORSO_MM, RIDER_ARM_MM,
    _compute_hip_xy, _compute_hands_xy, _compute_shoulder_xy,
)
from constraints import check_uci_compliance, wheelbase_summary

FORK_LENGTH_MM = 370.0


# ── Geometry calculation ──────────────────────────────────────────────────────

def _frame_points(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Compute 2D (x, y) coordinates of all key frame joints in mm.
    Origin = bottom bracket centre.
    """
    HTA = np.radians(p['head_tube_angle'])   # head tube angle from horizontal
    STA = np.radians(p['seat_tube_angle'])   # seat tube angle from horizontal

    ht_len   = p['head_tube_length']
    cs_len   = p['chainstay_length']
    bb_drop  = p['bb_drop']
    reach    = p['reach']
    stack    = p['stack']
    fork_off = p['fork_offset']
    r_rear   = p['rear_wheel_diameter']  / 2.0
    r_front  = p['front_wheel_diameter'] / 2.0

    # ── Key joints ────────────────────────────────────────────────────────────
    BB = np.array([0.0, 0.0])

    # Rear axle: behind BB by chainstay, above BB by bb_drop
    rear_axle = np.array([-cs_len, bb_drop])

    # Head tube: top defined by (reach, stack) from BB
    HT_top = np.array([reach, stack])

    # Head tube direction: from top downward-forward at angle HTA from horizontal
    #   cos(HTA) = horizontal component (forward),  -sin(HTA) = vertical (down)
    ht_dir  = np.array([np.cos(HTA), -np.sin(HTA)])
    HT_bot  = HT_top + ht_len * ht_dir

    # Fork: continues from HT_bot for fork_length, plus fork_offset perpendicular
    #   perp of ht_dir (CCW 90°): (sin(HTA), cos(HTA)) — points forward+up
    fork_perp  = np.array([np.sin(HTA), np.cos(HTA)])
    front_axle = HT_bot + FORK_LENGTH_MM * ht_dir + fork_off * fork_perp

    # Seat tube top: BB → direction (-cos(STA), sin(STA))
    #   height of junction ≈ 88 % of stack (typical road frame proportion)
    st_dir = np.array([-np.cos(STA), np.sin(STA)])
    st_top_y = stack * 0.88
    st_len   = st_top_y / np.sin(STA)
    ST_top   = BB + st_len * st_dir

    # Saddle tip: ~100 mm of seatpost above ST_top
    saddle_tip = ST_top + 100.0 * st_dir

    # Fork blade split points (two fork blades, ±20 mm of HT_bot perpendicularly)
    fork_left  = HT_bot + 20.0 * fork_perp
    fork_right = HT_bot - 20.0 * fork_perp

    return {
        'BB':         BB,
        'rear_axle':  rear_axle,
        'HT_top':     HT_top,
        'HT_bot':     HT_bot,
        'ST_top':     ST_top,
        'saddle_tip': saddle_tip,
        'front_axle': front_axle,
        'r_rear':     r_rear,
        'r_front':    r_front,
        'st_dir':     st_dir,
        'ht_dir':     ht_dir,
        'fork_perp':  fork_perp,
        'fork_left':  fork_left,
        'fork_right': fork_right,
    }


# ── Tube width scaling ────────────────────────────────────────────────────────

def _tube_lw(aspect: float, base_lw: float = 3.0) -> float:
    """Line width scaled by aspect ratio (wider = more aero-section visible)."""
    return base_lw * (0.7 + 0.6 * (aspect - 1.0) / 1.667)


# ── Rider stick figure ────────────────────────────────────────────────────────

def _circle_intersections(
    c0: np.ndarray,
    r0: float,
    c1: np.ndarray,
    r1: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return two circle intersection points; falls back to midpoint if degenerate."""
    d = np.linalg.norm(c1 - c0)
    if d < 1e-9:
        m = 0.5 * (c0 + c1)
        return m, m
    d_clamped = float(np.clip(d, abs(r0 - r1) + 1e-6, r0 + r1 - 1e-6))
    ex = (c1 - c0) / d
    a = (r0 ** 2 - r1 ** 2 + d_clamped ** 2) / (2.0 * d_clamped)
    h_sq = max(r0 ** 2 - a ** 2, 0.0)
    h = np.sqrt(h_sq)
    p2 = c0 + a * ex
    ey = np.array([-ex[1], ex[0]])
    return p2 + h * ey, p2 - h * ey


def _draw_rider(ax: plt.Axes, pts: Dict, p: Dict, color: str = '#2060C0') -> None:
    """Draw rider as linked body segments: legs, torso, arms, and head."""
    hip = _compute_hip_xy(p)
    hands = _compute_hands_xy(p)
    shoulder = _compute_shoulder_xy(p)

    # Two pedals 180° apart on a fixed crank circle around BB.
    crank_len = 170.0
    pedal_phase = np.radians(20.0)
    BB = pts['BB']
    pedal_a = BB + crank_len * np.array([np.cos(pedal_phase), -np.sin(pedal_phase)])
    pedal_b = BB - crank_len * np.array([np.cos(pedal_phase), -np.sin(pedal_phase)])

    # Legs: lower legs pedal->knee and upper legs knee->hip.
    upper_leg = 420.0
    lower_leg = 430.0
    knee_a_1, knee_a_2 = _circle_intersections(hip, upper_leg, pedal_a, lower_leg)
    knee_b_1, knee_b_2 = _circle_intersections(hip, upper_leg, pedal_b, lower_leg)
    knee_a = knee_a_1 if knee_a_1[1] >= knee_a_2[1] else knee_a_2
    knee_b = knee_b_1 if knee_b_1[1] >= knee_b_2[1] else knee_b_2

    # Arms: upper arms shoulder->elbow and forearms elbow->hands.
    upper_arm = 300.0
    forearm = 300.0
    elbow_1, elbow_2 = _circle_intersections(shoulder, upper_arm, hands, forearm)
    elbow = elbow_1 if elbow_1[1] <= elbow_2[1] else elbow_2
    arm_offset = np.array([0.0, 12.0])  # second arm slight offset for visibility

    # Head: above shoulders.
    head_r = 60.0
    head_c = shoulder + np.array([0.0, 1.35 * head_r])

    lw = 3.4
    # Torso
    ax.plot(*zip(hip, shoulder), color=color, lw=lw, solid_capstyle='round', zorder=6)
    # Left leg
    ax.plot(*zip(pedal_a, knee_a), color=color, lw=lw, solid_capstyle='round', zorder=6)
    ax.plot(*zip(knee_a, hip), color=color, lw=lw, solid_capstyle='round', zorder=6)
    # Right leg
    ax.plot(*zip(pedal_b, knee_b), color=color, lw=lw, alpha=0.9, solid_capstyle='round', zorder=6)
    ax.plot(*zip(knee_b, hip), color=color, lw=lw, alpha=0.9, solid_capstyle='round', zorder=6)
    # Left arm
    ax.plot(*zip(shoulder, elbow), color=color, lw=lw, solid_capstyle='round', zorder=6)
    ax.plot(*zip(elbow, hands), color=color, lw=lw, solid_capstyle='round', zorder=6)
    # Right arm (offset duplicate in side view)
    ax.plot(*zip(shoulder + arm_offset, elbow + arm_offset), color=color, lw=lw - 0.4,
            alpha=0.8, solid_capstyle='round', zorder=6)
    ax.plot(*zip(elbow + arm_offset, hands + arm_offset), color=color, lw=lw - 0.4,
            alpha=0.8, solid_capstyle='round', zorder=6)

    # Joints + head
    for joint in (hip, shoulder, knee_a, knee_b, elbow, pedal_a, pedal_b, hands):
        ax.add_patch(Circle(joint, 8.0, color=color, zorder=7))
    ax.add_patch(Circle(head_c, head_r, fill=False, edgecolor=color, linewidth=lw, zorder=7))


# ── Main plot function ────────────────────────────────────────────────────────

def plot_frame(
    params: Dict[str, float],
    generation: int,
    fitness: float,
    save_path: str = 'winning_frame.png',
    show: bool = True,
) -> None:
    """
    Render a side-profile of the bicycle frame and save to file.

    Args:
        params:     Frame parameter dict (output of genome_to_dict).
        generation: Generation number (for title).
        fitness:    Best CdA value (for annotation).
        save_path:  Output file path.
        show:       Whether to call plt.show().
    """
    pts = _frame_points(params)
    BB         = pts['BB']
    rear_axle  = pts['rear_axle']
    HT_top     = pts['HT_top']
    HT_bot     = pts['HT_bot']
    ST_top     = pts['ST_top']
    front_axle = pts['front_axle']
    saddle_tip = pts['saddle_tip']
    r_rear     = pts['r_rear']
    r_front    = pts['r_front']

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1A1A2E')
    ax.set_facecolor('#16213E')

    frame_color   = '#E94560'
    wheel_color   = '#0F3460'
    tire_color    = '#C0C0C0'
    tube_color    = '#E94560'
    saddle_color  = '#F5A623'

    def seg(a, b, lw=3, color=frame_color, zorder=3):
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=color, lw=lw, solid_capstyle='round', zorder=zorder)

    # ── Wheels ────────────────────────────────────────────────────────────────
    for axle_pt, wr in ((rear_axle, r_rear), (front_axle, r_front)):
        rim = Circle(axle_pt, wr, fill=False,
                     edgecolor=tire_color, linewidth=5, zorder=2)
        hub = Circle(axle_pt, 15, color=tire_color, zorder=4)
        spokes_r = Circle(axle_pt, wr * 0.95, fill=False,
                          edgecolor=wheel_color, linewidth=1, linestyle=':', zorder=2)
        ax.add_patch(rim)
        ax.add_patch(hub)
        ax.add_patch(spokes_r)
        # Draw a few spokes
        for angle_deg in np.linspace(0, 360, 20, endpoint=False):
            angle_rad = np.radians(angle_deg)
            spoke_end = axle_pt + (wr * 0.93) * np.array([np.cos(angle_rad),
                                                           np.sin(angle_rad)])
            ax.plot([axle_pt[0], spoke_end[0]], [axle_pt[1], spoke_end[1]],
                    color=wheel_color, lw=0.8, alpha=0.6, zorder=2)

    # Ground line (based on rear wheel)
    ground_y = rear_axle[1] - r_rear
    ax.axhline(ground_y, color='#404060', linewidth=1.5, linestyle='--', alpha=0.5)

    # ── Frame tubes ───────────────────────────────────────────────────────────
    # Seat tube: BB → ST_top
    seg(BB, ST_top,
        lw=_tube_lw(params['seat_tube_aspect'], 4),
        color=tube_color)

    # Top tube: ST_top → HT_top
    seg(ST_top, HT_top,
        lw=_tube_lw(params['top_tube_aspect'], 3.5),
        color=tube_color)

    # Down tube: BB → HT_bot
    seg(BB, HT_bot,
        lw=_tube_lw(params['down_tube_aspect'], 5),
        color=tube_color)

    # Head tube: HT_top → HT_bot
    seg(HT_top, HT_bot, lw=7, color='#FF8C00')   # highlighted

    # Chainstays: BB → rear_axle
    seg(BB, rear_axle,
        lw=_tube_lw(params['chainstay_aspect'], 3.5),
        color=tube_color)

    # Seatstay: ST_top → rear_axle
    seg(ST_top, rear_axle, lw=3.5, color=tube_color)

    # Fork (single line for side view)
    seg(HT_bot, front_axle, lw=3, color='#FF8C00')

    # BB shell (small circle)
    bb_shell = Circle(BB, 25, color='#FF8C00', zorder=5)
    ax.add_patch(bb_shell)

    # ── Saddle ────────────────────────────────────────────────────────────────
    # Seatpost: ST_top → saddle_tip
    seg(ST_top, saddle_tip, lw=2.5, color='#888888')
    # Saddle rail (horizontal bar 100 mm wide)
    saddle_mid = saddle_tip
    st_perp = np.array([pts['st_dir'][1], -pts['st_dir'][0]])   # perpendicular
    ax.plot(
        [saddle_mid[0] - 50 * st_perp[0], saddle_mid[0] + 70 * st_perp[0]],
        [saddle_mid[1] - 50 * st_perp[1], saddle_mid[1] + 70 * st_perp[1]],
        color=saddle_color, lw=6, solid_capstyle='round', zorder=6,
    )

    # ── Handlebars ────────────────────────────────────────────────────────────
    bar_center = np.array([
        params['reach'] + params['handlebar_reach'],
        params['stack'] - params['handlebar_drop'],
    ])
    # Stem: HT_top → bar_center
    seg(HT_top, bar_center, lw=3, color='#FF8C00')
    # Bar end marker (side view — show as a short cross-bar)
    ax.plot(
        [bar_center[0] - 8, bar_center[0] + 8],
        [bar_center[1],     bar_center[1]],
        color='#FF8C00', lw=6, solid_capstyle='round', zorder=6,
    )

    # ── Rider ─────────────────────────────────────────────────────────────────
    _draw_rider(ax, pts, params, color='#7FDBFF')

    # ── Annotations ───────────────────────────────────────────────────────────
    wb = compute_wheelbase(params)
    cr = check_uci_compliance(params)
    cda = calculate_cda(params)
    total_mass = RIDER_MASS + params['frame_weight']
    power_w = martin_power(cda, total_mass, 12.5)
    trunk_deg = _physical_trunk_angle(params)
    legal_str = 'UCI LEGAL' if cr.is_legal else 'UCI VIOLATION'
    legal_col = '#00FF7F' if cr.is_legal else '#FF4040'

    info_lines = [
        f"Generation {generation}  —  Evolved Frame Profile",
        "",
        f"  CdA = {fitness:.4f} m²   |   P@45 km/h = {power_w:.1f} W",
        f"  Stack = {params['stack']:.0f} mm   Reach = {params['reach']:.0f} mm",
        f"  Trunk angle = {trunk_deg:.1f}°   Wheelbase = {wb:.0f} mm",
        f"  Head tube angle = {params['head_tube_angle']:.1f}°   "
        f"Seat tube angle = {params['seat_tube_angle']:.1f}°",
        f"  Head tube length = {params['head_tube_length']:.0f} mm",
        f"  Chainstay = {params['chainstay_length']:.0f} mm   BB drop = {params['bb_drop']:.0f} mm",
        f"  Tube aspects (DT/ST/TT/CS): "
        f"{params['down_tube_aspect']:.2f} / {params['seat_tube_aspect']:.2f} / "
        f"{params['top_tube_aspect']:.2f} / {params['chainstay_aspect']:.2f}",
        f"  Frame weight = {params['frame_weight']:.2f} kg   "
        + (f"★ ASYMMETRIC WHEELS: {params['front_wheel_diameter']:.0f}F / {params['rear_wheel_diameter']:.0f}R mm"
           if abs(params['front_wheel_diameter'] - params['rear_wheel_diameter']) > 50
           else f"Wheels: {params['front_wheel_diameter']:.0f}F / {params['rear_wheel_diameter']:.0f}R mm"),
        f"  Handlebar reach = {params['handlebar_reach']:.0f} mm   "
        f"drop = {params['handlebar_drop']:.0f} mm   "
        f"width = {params['handlebar_width']:.0f} mm",
        "",
        f"  {legal_str}",
    ]
    text_box = "\n".join(info_lines)
    ax.text(
        0.01, 0.99, text_box,
        transform=ax.transAxes,
        fontsize=9, color='white',
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0A0A1A', alpha=0.85),
    )

    # Legal status badge
    ax.text(
        0.99, 0.99, legal_str,
        transform=ax.transAxes, fontsize=12, color=legal_col,
        ha='right', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0A0A1A', alpha=0.9),
    )

    # Dimension arrows: wheelbase
    axle_y = ground_y - 60
    ax.annotate(
        '', xy=(front_axle[0], axle_y), xytext=(rear_axle[0], axle_y),
        arrowprops=dict(arrowstyle='<->', color='#AAAAAA', lw=1.5),
    )
    ax.text(
        (front_axle[0] + rear_axle[0]) / 2, axle_y - 30,
        f"WB = {wb:.0f} mm", ha='center', va='top', color='#AAAAAA', fontsize=8,
    )

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    margin = 120
    all_x = [rear_axle[0] - r_rear, max(front_axle[0] + r_front, bar_center[0] + 20)]
    all_y = [ground_y - 100, max(saddle_tip[1], HT_top[1]) + 200]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y), max(all_y) + 100)

    ax.set_aspect('equal')
    ax.tick_params(colors='#AAAAAA')
    ax.spines[:].set_color('#404060')
    ax.set_xlabel('Position (mm)', color='#AAAAAA', fontsize=9)
    ax.set_ylabel('Height (mm)', color='#AAAAAA', fontsize=9)
    ax.tick_params(labelcolor='#AAAAAA')

    # Tube legend
    legend_elements = [
        mlines.Line2D([], [], color=tube_color, lw=4, label='Frame tubes'),
        mlines.Line2D([], [], color='#FF8C00', lw=4, label='Head tube / fork'),
        mlines.Line2D([], [], color=saddle_color, lw=4, label='Saddle'),
        mlines.Line2D([], [], color='#7FDBFF', lw=3, label='Rider'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              facecolor='#0A0A1A', edgecolor='#404060',
              labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Frame visualisation saved to: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ── Convergence plot ──────────────────────────────────────────────────────────

def plot_convergence(
    logbook: 'tools.Logbook',
    save_path: str = 'convergence.png',
    show: bool = True,
) -> None:
    """Plot the GA convergence curve (best power per generation)."""
    gens  = logbook.select('gen')
    mins  = logbook.select('min')
    means = logbook.select('mean')

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1A1A2E')
    ax.set_facecolor('#16213E')
    ax.plot(gens, mins,  color='#E94560', lw=2.0, label='Best power')
    ax.plot(gens, means, color='#7FDBFF', lw=1.5, linestyle='--', alpha=0.7,
            label='Mean power')
    ax.fill_between(gens, mins, means, alpha=0.1, color='#E94560')
    ax.set_xlabel('Generation', color='white')
    ax.set_ylabel('Power (W)', color='white')
    ax.set_title('Genetic Algorithm Convergence — Bicycle Frame Power Minimisation',
                 color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#404060')
    ax.legend(facecolor='#0A0A1A', edgecolor='#404060', labelcolor='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Convergence plot saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
