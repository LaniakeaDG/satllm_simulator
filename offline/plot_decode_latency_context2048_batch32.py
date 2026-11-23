"""绘制初始上下文 2048、batch 32 的逐层 Decode 单步时延，不平滑或删点。"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "data/profiles/profile.json"
OUTPUT_BASE = PROJECT_ROOT / "data/profiles/decode_context2048_batch32"
INITIAL_CONTEXT = 2048
BATCH_SIZE = 32
LAYERS = (0, 14, 27)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def main():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["decode"]
    rows = sorted(
        (row for row in profile["records"]
         if row["initial_context_length"] == INITIAL_CONTEXT
         and row["batch_size"] == BATCH_SIZE),
        key=lambda row: (row["model_layers"][0], row["step"]),
    )
    expected = {(layer, step) for layer in LAYERS for step in range(64)}
    actual = {(row["model_layers"][0], row["step"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Expected exactly 64 steps for each of layers 0, 14, 27")
    if any(row["context_length"] != INITIAL_CONTEXT + row["step"] for row in rows):
        raise ValueError("Context length does not match initial context + step")

    output = OUTPUT_BASE
    if output.exists():
        output = output.with_name(
            output.name + datetime.now(timezone.utc).strftime("_%Y%m%d_%H%M%S_%f")
        )
    output.mkdir(parents=True, exist_ok=False)
    table = [
        {"initial_context_length": INITIAL_CONTEXT, "batch_size": BATCH_SIZE,
         "layer": row["model_layers"][0], "step": row["step"],
         "context_length": row["context_length"], "latency_ms": row["latency_ms"]}
        for row in rows
    ]
    with (output / "decode_latency_by_context.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 18,
        "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.fontsize": 18, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(13, 6.5), layout="constrained")
    for layer, color in zip(LAYERS, COLORS):
        selected = [row for row in table if row["layer"] == layer]
        ax.plot(
            [row["context_length"] for row in selected],
            [row["latency_ms"] for row in selected],
            color=color, linestyle="-", linewidth=2, marker="o", markersize=6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6,
            label=f"Layer {layer}",
        )
        print(f"Layer {layer}: {min(row['latency_ms'] for row in selected):.6f}"
              f"–{max(row['latency_ms'] for row in selected):.6f} ms")
    ax.set_xlabel("Context length (tokens)", labelpad=12)
    ax.set_ylabel("Decode step latency (ms)", labelpad=12)
    ax.set_xticks([2048, 2060, 2072, 2084, 2096, 2111])
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlim(2046, 2113)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"decode_latency_by_context.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Source: {PROFILE_PATH}")
    print("192 records; latency_ms is the mean of 30 replay measurements per step; no smoothing or filtering.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
