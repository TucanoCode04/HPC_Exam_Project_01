"""
Turns the three sweep CSVs produced by sweep_assignment3.sh into report-ready
PNGs, analogous to assignment_1/plots but sourced from our own timers instead
of VTune. Run locally after scp-ing size_sweep.csv, thread_sweep.csv and
blocksize_sweep.csv down from the cluster:

    pip install pandas matplotlib
    python plot_sweeps.py

Reads the three CSVs from the current directory, writes PNGs into ./plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BG = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order (validated for colorblind-safety) -- one color per
# scenario, used consistently across every chart so "sim2" always means the
# same hue.
SCENARIO_COLORS = {
    "sim1_ppm": "#2a78d6",
    "sim2_ppm": "#eb6834",
    "sim3_ppm": "#1baf7a",
}
SCENARIO_LABELS = {
    "sim1_ppm": "sim1 (single wave)",
    "sim2_ppm": "sim2 (two waves, t=0)",
    "sim3_ppm": "sim3 (two waves, delayed)",
}

PHASE_COLORS = {"colorize+write": "#2a78d6", "compute": "#eb6834"}


def style_axes(ax, xlabel, ylabel, title):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)


def plot_size_sweep(df, out_dir):
    fig, (ax_time, ax_phase) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)

    for scenario, group in df.groupby("scenario"):
        group = group.sort_values("matrix_size")
        ax_time.plot(group["matrix_size"], group["elapsed_s"], marker="o",
                     markersize=5, linewidth=2,
                     color=SCENARIO_COLORS.get(scenario, MUTED),
                     label=SCENARIO_LABELS.get(scenario, scenario))
    style_axes(ax_time, "Matrix size (M)", "Execution time (s)",
               "Execution time vs. grid size")

    phase_avg = df.groupby("matrix_size")[["color_time_s", "compute_time_s"]].mean()
    ax_phase.plot(phase_avg.index, phase_avg["color_time_s"], marker="o",
                  markersize=5, linewidth=2, color=PHASE_COLORS["colorize+write"],
                  label="GPU colorize + write")
    ax_phase.plot(phase_avg.index, phase_avg["compute_time_s"], marker="o",
                  markersize=5, linewidth=2, color=PHASE_COLORS["compute"],
                  label="OpenMP compute_next")
    style_axes(ax_phase, "Matrix size (M)", "Time (s, avg. across ranks)",
               "Where the time goes vs. grid size")

    fig.tight_layout()
    fig.savefig(out_dir / "size_sweep_execution_time.png", dpi=200, facecolor=BG)
    plt.close(fig)


def _speedup_efficiency(df, time_column):
    baseline = df[df["omp_threads"] == df["omp_threads"].min()][time_column].mean()
    grouped = df.groupby("omp_threads")[time_column].mean().sort_index()
    speedup = baseline / grouped
    efficiency = 100.0 * speedup / grouped.index
    return speedup, efficiency


def plot_thread_sweep(df, out_dir):
    fig, (ax_speedup, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)

    full_speedup, full_eff = _speedup_efficiency(df, "elapsed_s")
    compute_speedup, compute_eff = _speedup_efficiency(df, "compute_time_s")

    threads = full_speedup.index
    ax_speedup.plot(threads, threads, linestyle="--", color=AXIS,
                     linewidth=1.5, label="Ideal speedup")
    ax_speedup.plot(threads, full_speedup, marker="o", markersize=5,
                     linewidth=2, color=SCENARIO_COLORS["sim1_ppm"],
                     label="Full pipeline (GPU + OpenMP)")
    ax_speedup.plot(threads, compute_speedup, marker="s", markersize=5,
                     linewidth=2, color=SCENARIO_COLORS["sim2_ppm"],
                     label="OpenMP compute_next only")
    style_axes(ax_speedup, "OpenMP threads", "Speedup vs. 1 thread",
               "Speedup: full pipeline vs. compute-only")

    ax_eff.plot(threads, full_eff, marker="o", markersize=5, linewidth=2,
                color=SCENARIO_COLORS["sim1_ppm"],
                label="Full pipeline (GPU + OpenMP)")
    ax_eff.plot(threads, compute_eff, marker="s", markersize=5, linewidth=2,
                color=SCENARIO_COLORS["sim2_ppm"],
                label="OpenMP compute_next only")
    style_axes(ax_eff, "OpenMP threads", "Parallel efficiency (%)",
               "Parallel efficiency: full pipeline vs. compute-only")

    fig.tight_layout()
    fig.savefig(out_dir / "thread_sweep_speedup_efficiency.png", dpi=200, facecolor=BG)
    plt.close(fig)


def plot_blocksize_sweep(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG)

    for scenario, group in df.groupby("scenario"):
        group = group.sort_values("block_size")
        ax.plot(group["block_size"], group["color_time_s"], marker="o",
                markersize=5, linewidth=2,
                color=SCENARIO_COLORS.get(scenario, MUTED),
                label=SCENARIO_LABELS.get(scenario, scenario))
    ax.set_xscale("log", base=2)
    style_axes(ax, "CUDA threads per block", "GPU colorize + write time (s)",
               "colorize_kernel time vs. block size")

    fig.tight_layout()
    fig.savefig(out_dir / "blocksize_sweep_kernel_time.png", dpi=200, facecolor=BG)
    plt.close(fig)


CONTENTION_MODE_LABELS = {
    "solo": "Solo\n(1 rank, exclusive GPU)",
    "shared": "Shared\n(3 ranks, 1 GPU)",
    "distributed": "Distributed\n(3 ranks, 3 GPUs, 1 node)",
    "multinode": "Multi-node\n(3 ranks, 3 nodes,\n1 GPU each)",
}
CONTENTION_MODE_ORDER = ["solo", "shared", "distributed", "multinode"]


def plot_contention_sweep(df, out_dir):
    # Only sim1 exists under "solo" (rank 0's scenario); compare against
    # sim1's own row under every other mode for an apples-to-apples set.
    # Handles however many modes are actually present in the CSV -- not
    # every experiment (solo/shared/distributed/multinode) has to have
    # been run yet for this to produce a plot.
    sim1 = df[df["scenario"] == "sim1_ppm"].set_index("mode")
    modes = [m for m in CONTENTION_MODE_ORDER if m in sim1.index]
    modes += [m for m in sim1.index if m not in CONTENTION_MODE_ORDER]
    labels = [CONTENTION_MODE_LABELS.get(m, m) for m in modes]
    compute_vals = [sim1.loc[m, "compute_time_s"] for m in modes]
    color_vals = [sim1.loc[m, "color_time_s"] for m in modes]

    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(modes), 4.5))
    fig.patch.set_facecolor(BG)

    x = range(len(modes))
    ax.bar(x, compute_vals, width=0.5, color=PHASE_COLORS["compute"],
           label="OpenMP compute_next")
    ax.bar(x, color_vals, width=0.5, bottom=compute_vals,
           color=PHASE_COLORS["colorize+write"], label="GPU colorize + write")
    for i, (c, g) in enumerate(zip(compute_vals, color_vals)):
        ax.text(i, c + g + max(color_vals) * 0.02, f"{c + g:.2f}s", ha="center",
                 color=INK, fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    style_axes(ax, "", "Time (s)", "sim1: GPU contention across execution modes")

    fig.tight_layout()
    fig.savefig(out_dir / "contention_sweep_solo_vs_shared.png", dpi=200, facecolor=BG)
    plt.close(fig)


def main():
    here = Path(__file__).parent
    out_dir = here / "plots"
    out_dir.mkdir(exist_ok=True)

    plot_size_sweep(pd.read_csv(here / "size_sweep.csv"), out_dir)
    plot_thread_sweep(pd.read_csv(here / "thread_sweep.csv"), out_dir)
    plot_blocksize_sweep(pd.read_csv(here / "blocksize_sweep.csv"), out_dir)

    written = ["size_sweep_execution_time.png", "thread_sweep_speedup_efficiency.png",
               "blocksize_sweep_kernel_time.png"]

    contention_csv = here / "contention_sweep.csv"
    if contention_csv.exists():
        plot_contention_sweep(pd.read_csv(contention_csv), out_dir)
        written.append("contention_sweep_solo_vs_shared.png")
    else:
        print(f"Skipping contention plot: {contention_csv} not found yet.")

    print(f"Wrote {', '.join(written)} to {out_dir}")


if __name__ == "__main__":
    main()
