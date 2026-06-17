"""
Animate a sphere cross-section inflating and deflating over 5 seconds.
Peak inflation at t=2.5s (halfway). Only r0 changes; t0 stays constant.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse

# --- Parameters ---
R0_MIN = 2.8   # inner radius at rest (cm)
R0_MAX = 4.0   # inner radius at peak inflation (cm)
T0 = 0.6       # wall thickness (constant, cm)
DURATION = 5.0 # seconds
FPS = 30

# --- Derived ---
N_FRAMES = int(DURATION * FPS)
PADDING = 0.5


def r0_at(t):
    """Sinusoidal inflation: min → max at t=2.5s → min at t=5s."""
    return R0_MIN + (R0_MAX - R0_MIN) * 0.5 * (1 - np.cos(2 * np.pi * t / DURATION))


def draw_sphere_cross_section(ax, r0, t0):
    ax.clear()

    r_outer = r0 + t0
    lim = R0_MAX + T0 + PADDING

    # --- Two unfilled circles (wall outline only) ---
    outer_circle = plt.Circle((0, 0), r_outer, fill=False, edgecolor="black", linewidth=2, zorder=2)
    inner_circle = plt.Circle((0, 0), r0,      fill=False, edgecolor="black", linewidth=2, zorder=2)
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)

    # --- Dashed equator ellipse (scales with r0 — represents the inner equator) ---
    equator = patches.Ellipse(
        (0, 0), width=2 * r0, height=2 * r0 * 0.22,
        linestyle="--", edgecolor="white", facecolor="none", linewidth=0.1, zorder=3
    )
    ax.add_patch(equator)

    # --- r0 arrow (center → inner surface) ---
    ax.annotate(
        "", xy=(0, r0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
        zorder=4
    )
    ax.text(0.12, r0 * 0.52, r"$r_0$", fontsize=20, fontweight="bold", zorder=5)

    # --- t0 arrow (inner → outer surface, upper-right) ---
    angle_rad = np.radians(42)
    p1 = (r0 * np.cos(angle_rad), r0 * np.sin(angle_rad))
    p2 = (r_outer * np.cos(angle_rad), r_outer * np.sin(angle_rad))
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.8),
        zorder=4
    )
    label_dist = r_outer + 0.45
    ax.text(label_dist * np.cos(angle_rad), label_dist * np.sin(angle_rad),
            r"$t_0$", fontsize=16, fontweight="bold", va="center", ha="left", zorder=5)

    # --- Time label ---
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="FILE",
                        help="save animation to file (e.g. sphere.mp4 or sphere.gif)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")

    # Time counter text
    time_text = fig.text(0.5, 0.04, "", ha="center", fontsize=12, color="#333")

    def update(frame):
        t = frame / FPS
        r0 = r0_at(t)
        draw_sphere_cross_section(ax, r0, T0)
        time_text.set_text(f"t = {t:.2f} s   |   r₀ = {r0:.2f} cm   |   t₀ = {T0:.2f} cm")
        return []

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)

    if args.save:
        if args.save.endswith(".gif"):
            ani.save(args.save, writer="pillow", fps=FPS)
        else:
            writer = FFMpegWriter(fps=FPS, bitrate=1800)
            ani.save(args.save, writer=writer)
        print(f"Saved to {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
