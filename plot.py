import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import BASELINES_PATH

BEST_COLOR = "#2a78d6"
MEAN_COLOR = "#eb6834"
REFERENCE_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

REFERENCE_LABELS = {
    "random": "Random buttons",
    "spin_left_fire": "Spin left + fire",
    "spin_right_fire": "Spin right + fire",
}


def read_history(path):
    with open(path) as file:
        rows = list(csv.DictReader(file))
    generations = [int(row["generation"]) for row in rows]
    best = [float(row["best"]) for row in rows]
    mean = [float(row["mean"]) for row in rows]
    worst = [float(row["worst"]) for row in rows]
    return generations, best, mean, worst


def read_references(path):
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        return {
            row["agent"]: float(row["mean"])
            for row in csv.DictReader(file)
            if row["agent"] in REFERENCE_LABELS
        }


def plot(run_dir):
    generations, best, mean, worst = read_history(os.path.join(run_dir, "history.csv"))
    references = read_references(BASELINES_PATH)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    ax.fill_between(generations, worst, best, color=BEST_COLOR, alpha=0.08, linewidth=0,
                    label="Population range (worst to best)")
    ax.plot(generations, best, color=BEST_COLOR, linewidth=1, label="Best agent of the generation")
    ax.plot(generations, mean, color=MEAN_COLOR, linewidth=2, label="Population mean")

    for name, score in references.items():
        ax.axhline(score, color=REFERENCE_COLOR, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(generations[-1], score, f" {REFERENCE_LABELS[name]} ({score:.0f})",
                color=TEXT_SECONDARY, fontsize=9, va="center", ha="left")

    ax.set_title("Asteroids neuroevolution: score per generation",
                 color=TEXT_PRIMARY, fontsize=13, loc="left")
    ax.set_xlabel("Generation", color=TEXT_SECONDARY)
    ax.set_ylabel("Average score per game", color=TEXT_SECONDARY)
    ax.set_xlim(generations[0], generations[-1])
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_SECONDARY)

    fig.tight_layout()
    path = os.path.join(run_dir, "evolution.png")
    fig.savefig(path, facecolor="white")
    print(f"Saved {path}")


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    plot(run_dir)
