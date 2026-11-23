"""Prefill 不同 chunk size 的上下文—时延、功率、显存图；对三个层取平均。"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data/profiles/profile.json"
OUTPUT_DIR = ROOT / "data/profiles/prefill"
CHUNKS = (64, 128, 256, 512)
LAYERS = (0, 14, 27)
INPUT_LENGTH = 2048
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8357c5")
MARKERS = ("s", "o", "^", "v")


def summarize(records):
    expected = {(chunk, index, (layer,)) for chunk in CHUNKS
                for index in range(INPUT_LENGTH // chunk) for layer in LAYERS}
    actual = {(r["chunk_size"], r["chunk_index"], tuple(r["model_layers"])) for r in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("Expected one complete 2048-token request for four chunk sizes and three layers")
    if len({r["request_id"] for r in records}) != 1:
        raise ValueError("Expected one profiled request")
    groups = defaultdict(list)
    for row in records:
        if (row["batch_size"] != 1 or row["input_length"] != INPUT_LENGTH
                or row["query_tokens"] != row["chunk_size"]
                or row["context_length"] != row["chunk_size"] * row["chunk_index"]):
            raise ValueError("Unexpected Prefill shape or context")
        if any(not math.isfinite(row[field]) or row[field] <= 0
               for field in ("latency_ms", "avg_power_w", "peak_memory_bytes")):
            raise ValueError("Invalid metric")
        if not row["peak_memory_bytes"] >= row["baseline_memory_bytes"] > 0:
            raise ValueError("Invalid peak or baseline memory")
        groups[row["chunk_size"], row["context_length"]].append(row)
    table = []
    for (chunk, context), rows in sorted(groups.items()):
        table.append({
            "chunk_size": chunk, "context_length": context, "batch_size": 1,
            "chunk_index": rows[0]["chunk_index"], "layers": "0;14;27", "records": len(rows),
            "mean_latency_ms": statistics.fmean(r["latency_ms"] for r in rows),
            "mean_power_w": statistics.fmean(r["avg_power_w"] for r in rows),
            "mean_peak_memory_bytes": statistics.fmean(r["peak_memory_bytes"] for r in rows),
            "mean_peak_memory_gib": statistics.fmean(r["peak_memory_bytes"] for r in rows) / 2**30,
        })
    return table


def plot_axis(ax, table, field, ylabel, memory=False):
    for chunk, color, marker in zip(CHUNKS, COLORS, MARKERS):
        rows = [r for r in table if r["chunk_size"] == chunk]
        ax.plot([r["context_length"] for r in rows], [r[field] for r in rows],
                color=color, label=f"Chunk {chunk}", linestyle="-", linewidth=4.5,
                marker=marker, markersize=16, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("Context length", labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xticks([0, 512, 1024, 1536, 2048])
    ax.set_xlim(-80, 2128)
    ax.set_ylim(0, max(r[field] for r in table) * 1.10)
    ax.grid(axis="both", color="#bdbdbd", linewidth=1.0, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=6, width=1.3)
    legend = ax.legend(loc="lower right",
                       ncol=1 if memory else 2, frameon=True,
                       fancybox=False, edgecolor="black", facecolor="white",
                       framealpha=1, fontsize=20, columnspacing=1.2,
                       handlelength=2.0, borderpad=0.5)
    legend.get_frame().set_linewidth(1.3)


def save_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["prefill"]
    table = summarize(profile["records"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "means.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 24, "axes.labelsize": 28,
        "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 20,
        "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
        "ytick.color": "black", "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": 1.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
    plot_axis(axes[0], table, "mean_latency_ms", "Latency (ms)")
    plot_axis(axes[1], table, "mean_power_w", "Power (W)")
    plot_axis(axes[2], table, "mean_peak_memory_gib", "Memory (GiB)", memory=True)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.17, top=0.96, wspace=0.32)
    save_figure(fig, "metrics")
    metadata = {
        "source": str(PROFILE_PATH), "batch_size": 1, "input_length": INPUT_LENGTH,
        "requests": 1, "layers": list(LAYERS),
        "aggregation": "Arithmetic mean over three measured layers at each chunk size and context; no averaging across chunk positions, smoothing or filtering.",
        "x_axis": "Historical KV context length before the current chunk (chunk_index * chunk_size); each curve ends at 2048 - chunk_size, with no extrapolation.",
        "latency": "Mean single-layer chunk latency, not per-token or full-model latency. Each record is the mean of 30 replays.",
        "power": "Mean independently measured board power, not time-weighted power or energy; not simultaneous with latency samples.",
        "memory_scope": profile["metadata"]["memory_scope"],
        "memory": "Mean process peak allocated memory over layers, including preallocated all-layer KV; not per-layer-exclusive or full-model-resident memory. GiB = 2**30 bytes.",
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Verified {len(profile['records'])} records -> {len(table)} points x 3 layers")
    print(f"Saved (overwrite): {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
