"""三层×64步平均的 Decode 时延、功率、显存图；固定目录，重复运行覆盖输出。"""

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data/profiles/profile.json"
OUTPUT_DIR = ROOT / "data/profiles/decode"
BATCHES = (1, 32, 64, 128)
CONTEXTS = (128, 512, 1024, 2048)
LAYERS = (0, 14, 27)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8357c5")
MARKERS = ("s", "o", "^", "v")


def summarize(records):
    expected = {(b, c, (l,), s) for b in BATCHES for c in CONTEXTS
                for l in LAYERS for s in range(64)}
    actual = {(r["batch_size"], r["initial_context_length"], tuple(r["model_layers"]), r["step"])
              for r in records}
    if len(records) != len(expected) or actual != expected:
        raise ValueError("Expected exactly 4 batches x 4 contexts x 3 layers x 64 steps")
    for row in records:
        if row["context_length"] != row["initial_context_length"] + row["step"]:
            raise ValueError("Unexpected context length")
        if any(not math.isfinite(row[field]) or row[field] <= 0
               for field in ("latency_ms", "avg_power_w", "peak_memory_bytes")):
            raise ValueError("Invalid metric")
        if not row["peak_memory_bytes"] >= row["baseline_memory_bytes"] > 0:
            raise ValueError("Invalid peak or baseline memory")
    table = []
    for batch in BATCHES:
        for context in CONTEXTS:
            rows = [r for r in records if r["batch_size"] == batch
                    and r["initial_context_length"] == context]
            table.append({
                "batch_size": batch, "initial_context_length": context,
                "layers": "0;14;27", "steps_per_layer": 64, "records": len(rows),
                "mean_latency_ms": statistics.fmean(r["latency_ms"] for r in rows),
                "mean_power_w": statistics.fmean(r["avg_power_w"] for r in rows),
                "mean_peak_memory_bytes": statistics.fmean(r["peak_memory_bytes"] for r in rows),
                "mean_peak_memory_gib": statistics.fmean(r["peak_memory_bytes"] for r in rows) / 2**30,
            })
    return table


def plot_axis(ax, table, field, ylabel, memory=False):
    for batch, color, marker in zip(BATCHES, COLORS, MARKERS):
        rows = [r for r in table if r["batch_size"] == batch]
        ax.plot([r["initial_context_length"] for r in rows], [r[field] for r in rows],
                color=color, label=f"Batch {batch}", linestyle="-", linewidth=4.5,
                marker=marker, markersize=16, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("Context length", labelpad=12)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xticks(CONTEXTS)
    ax.set_xlim(30, 2170)
    ax.set_ylim(0, max(r[field] for r in table) * 1.10)
    ax.grid(axis="both", color="#bdbdbd", linewidth=1.0, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=6, width=1.3)
    legend = ax.legend(loc="upper left" if memory else "lower right",
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
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["decode"]
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
        "source": str(PROFILE_PATH),
        "aggregation": "Arithmetic mean over layers 0,14,27 and 64 steps (192 records per point); no filtering or smoothing.",
        "latency": "Mean single-layer latency, not the sum of three layers or full-model latency. Each step is the mean of 30 replay samples.",
        "power": "Mean independently measured board power, not time-weighted power, energy or the latency measurement window.",
        "memory_scope": profile["metadata"]["memory_scope"],
        "cache_policy": profile["metadata"]["cache_policy"],
        "memory": "Mean of per-measurement process peaks, not the maximum across steps or full-model weight memory. GiB = 2**30 bytes. KV is preallocated.",
        "x_axis": "Initial context C0; actual measured contexts span C0 through C0+63.",
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Verified {len(profile['records'])} records -> {len(table)} points x 192 records")
    print(f"Saved (overwrite): {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
