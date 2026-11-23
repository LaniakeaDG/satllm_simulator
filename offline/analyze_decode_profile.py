"""固定路径的 Decode 诊断图与逐条表；只读原始结果，不清洗或重新测量。"""

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = PROJECT_ROOT / "data/profiles"
MERGED_FILE = PROFILE_DIR / "profile.json"
OUTPUT_BASE = PROFILE_DIR / "decode_analysis_20261121"
LAYERS = [0, 14, 27]
BATCHES = [1, 32, 64, 128]
CONTEXTS = [128, 512, 1024, 2048]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# 仅标记“相对同初始上下文/Batch/层的 64 步中位数 >2 倍”，不是污染判据。
REVIEW_RATIO = 2.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
    "ytick.color": "black", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "lines.markersize": 6, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def case_key(row):
    return (row["initial_context_length"], row["batch_size"], row["step"], row["model_layers"][0])


def group_key(row):
    return (row["initial_context_length"], row["batch_size"], row["model_layers"][0])


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def style_axis(ax):
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def group_rows(rows):
    result = defaultdict(list)
    for row in rows:
        result[group_key(row)].append(row)
    for values in result.values():
        values.sort(key=lambda row: row["step"])
    return result


def overview(groups, output):
    fig, axes = plt.subplots(2, 4, figsize=(23, 11), layout="constrained")
    for column, batch in enumerate(BATCHES):
        for metric_row, (field, label) in enumerate([
                ("latency_ms", "Median step latency (ms)"), ("avg_power_w", "Median board power (W)")]):
            ax = axes[metric_row, column]
            for layer, color in zip(LAYERS, COLORS):
                samples = [[row[field] for row in groups[context, batch, layer]] for context in CONTEXTS]
                ax.plot(CONTEXTS, [np.median(values) for values in samples], "o-", color=color,
                        label=f"Layer {layer}", markeredgecolor="white", markeredgewidth=0.8)
                ax.fill_between(CONTEXTS, [np.percentile(values, 10) for values in samples],
                                [np.percentile(values, 90) for values in samples], color=color, alpha=0.09)
            ax.set_xlabel(f"Initial context (tokens)\nBatch {batch}")
            ax.set_xticks(CONTEXTS, ["128", "512", "1024", "2048"], rotation=25)
            ax.set_ylim(bottom=0 if metric_row == 0 else 180)
            if column == 0:
                ax.set_ylabel(label)
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    save_figure(fig, output, "decode_context_latency_power")


def steps(groups, output, field, ylabel, filename, limits):
    fig, axes = plt.subplots(4, 4, figsize=(24, 17), sharex=True, sharey=True, layout="constrained")
    for i, context in enumerate(CONTEXTS):
        for j, batch in enumerate(BATCHES):
            ax = axes[i, j]
            for layer, color in zip(LAYERS, COLORS):
                rows = groups[context, batch, layer]
                ax.plot([row["step"] for row in rows], [row[field] for row in rows], "o-",
                        color=color, label=f"Layer {layer}", markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.4, linewidth=1.1)
            ax.set_ylim(*limits)
            ax.set_xticks([0, 16, 32, 48, 63])
            if j == 0:
                ax.set_ylabel(f"{ylabel}\nInitial context {context}")
            if i == 3:
                ax.set_xlabel(f"Step (0-based)\nBatch {batch}")
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    save_figure(fig, output, filename)


def sample_diagnostics(rows, output):
    fig, axes = plt.subplots(2, 4, figsize=(23, 11), layout="constrained")
    for column, batch in enumerate(BATCHES):
        chosen = [row for row in rows if row["batch_size"] == batch]
        values = [np.median(row["latency_samples_ms"]) for row in chosen]
        ax = axes[0, column]
        ax.hist(values, bins=np.linspace(0.3, 1.3, 51), color=COLORS[0], rwidth=0.9)
        outside = sum(value < 0.3 or value > 1.3 for value in values)
        ax.text(0.04, 0.95, f"Batch {batch}\nOutside x range: {outside}/{len(values)}",
                transform=ax.transAxes, va="top", fontsize=14)
        ax.set_xlabel("Within-record sample median (ms)")
        ax.set_xlim(0.3, 1.3)
        if column == 0:
            ax.set_ylabel("Records (all contexts / layers)")
        style_axis(ax)
        worst = max(chosen, key=lambda row: row["latency_ms"])
        ax = axes[1, column]
        ax.plot(range(1, 31), worst["latency_samples_ms"], "o-", color=COLORS[0])
        ax.text(0.04, 0.96, f'C0={worst["initial_context_length"]}, B={batch}\n'
                f'step={worst["step"]}, layer={worst["model_layers"][0]}',
                transform=ax.transAxes, va="top", fontsize=14)
        ax.set_xlabel("Replay sample index\nHighest-mean record for this batch")
        ax.set_ylim(0, 8.2)
        if column == 0:
            ax.set_ylabel("Sample latency (ms)")
        style_axis(ax)
    save_figure(fig, output, "decode_latency_modes_and_replay_samples")


def memory_figure(groups, config, settings, output):
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.6), layout="constrained")
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[settings["dtype"]]
    for ax, batch in zip(axes, BATCHES):
        measured = [np.median([row["peak_memory_bytes"] for layer in LAYERS
                               for row in groups[context, batch, layer]]) / 2**30 for context in CONTEXTS]
        kv = [2 * config["num_hidden_layers"] * batch * (context + settings["decode_steps"])
              * config["num_key_value_heads"] * config["head_dim"] * dtype_bytes / 2**30
              for context in CONTEXTS]
        ax.plot(CONTEXTS, measured, "o-", color=COLORS[0], label="Measured process peak")
        ax.plot(CONTEXTS, kv, "o-", color=COLORS[1], label="All-layer KV allocation")
        ax.set_xlabel(f"Initial context (tokens)\nBatch {batch}")
        ax.set_xticks(CONTEXTS, ["128", "512", "1024", "2048"], rotation=25)
        ax.set_ylim(bottom=0)
        style_axis(ax)
    axes[0].set_ylabel("GPU allocated memory (GiB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    save_figure(fig, output, "decode_memory_vs_preallocated_kv")


def timeline(rows, baselines, output):
    new = [row for row in rows if row.get("measurement_started_at_utc")]
    fig, axes = plt.subplots(3, 1, figsize=(23, 13), sharex=True, layout="constrained")
    fields = [(lambda row: row["latency_ms"] / baselines[group_key(row)], "Latency / group-layer median"),
              (lambda row: row["avg_power_w"], "Board power (W)"),
              (lambda row: (datetime.fromisoformat(row["measurement_finished_at_utc"])
                            - datetime.fromisoformat(row["measurement_started_at_utc"])).total_seconds(),
               "Full benchmark window (s)")]
    for ax, (value, label) in zip(axes, fields):
        for layer, color in zip(LAYERS, COLORS):
            # 各组合之间不连线，避免将 KV 准备间隙画成有测量值的时间段。
            for context, batch in dict.fromkeys((row["initial_context_length"], row["batch_size"]) for row in new):
                selected = [row for row in new if group_key(row) == (context, batch, layer)]
                times = [datetime.fromisoformat(row["measurement_started_at_utc"]) for row in selected]
                ax.plot(times, [value(row) for row in selected], "o-", color=color,
                        label=f"Layer {layer}" if (context, batch) == (1024, 128) else None,
                        markersize=4.5, linewidth=0.8)
        ax.set_ylabel(label)
        style_axis(ax)
    axes[0].axhline(REVIEW_RATIO, color="#52514e", linewidth=1)
    axes[0].text(0.005, 0.98, "2x: review cue only; not proof of GPU sharing",
                 transform=axes[0].transAxes, va="top", fontsize=14)
    candidates = sorted([row for row in new if row["latency_ms"] / baselines[group_key(row)] > REVIEW_RATIO],
                        key=lambda row: row["measurement_started_at_utc"])
    for index, row in enumerate(candidates, 1):
        time = datetime.fromisoformat(row["measurement_started_at_utc"])
        ratio = row["latency_ms"] / baselines[group_key(row)]
        axes[0].annotate(str(index), (time, ratio), xytext=(0, 10), textcoords="offset points",
                         ha="center", fontsize=13)
    axes[0].set_ylim(0, 5.4)
    axes[2].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    axes[2].set_xlabel("2026-11-20 UTC; points use full-benchmark start time (new records only)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    save_figure(fig, output, "decode_timestamped_measurement_windows")


def paired_figure(interrupted, reference, output):
    fig, axes = plt.subplots(2, 1, figsize=(23, 9), sharex=True, layout="constrained")
    for ax, field, label in zip(axes, ["latency_ms", "avg_power_w"],
                               ["Interrupted / rerun latency", "Interrupted / rerun power"]):
        for layer, color in zip(LAYERS, COLORS):
            indices = [i for i, row in enumerate(interrupted) if row["model_layers"][0] == layer]
            ratios = [interrupted[i][field] / reference[case_key(interrupted[i])][field] for i in indices]
            ax.plot(indices, ratios, "o-", color=color, label=f"Layer {layer}",
                    markersize=4, linewidth=0.6, alpha=0.85)
        ax.axhline(1, color="#52514e", linewidth=1)
        ax.set_ylabel(label)
        style_axis(ax)
    for i, row in enumerate(interrupted):
        if i == 0 or (row["initial_context_length"], row["batch_size"]) != (
                interrupted[i-1]["initial_context_length"], interrupted[i-1]["batch_size"]):
            for ax in axes:
                ax.axvline(i, color="#c3c2b7", linewidth=0.6)
    axes[1].set_xlabel("Interrupted-run record index (0-based); no per-record wall-clock timestamps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    save_figure(fig, output, "decode_interrupted_vs_rerun_paired_cases")


def main():
    sources = {"merged": MERGED_FILE}
    content = {name: path.read_bytes() for name, path in sources.items()}
    data = {"merged": json.loads(content["merged"])["decode"]}
    rows = data["merged"]["records"]
    metadata = data["merged"]["metadata"]
    assert metadata["status"] == "complete" and len(rows) == 3072
    assert len({case_key(row) for row in rows}) == len(rows)
    expected = {(context, batch, step, layer) for context in CONTEXTS for batch in BATCHES
                for step in range(64) for layer in LAYERS}
    assert {case_key(row) for row in rows} == expected
    output = OUTPUT_BASE
    if output.exists():
        output = output.with_name(output.name + datetime.now(timezone.utc).strftime("_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    groups = group_rows(rows)
    baselines = {key: float(np.median([row["latency_ms"] for row in values])) for key, values in groups.items()}
    tables, summaries = [], []
    for index, row in enumerate(rows):
        latency_samples = np.array(row["latency_samples_ms"])
        assert len(latency_samples) == 30 and np.all(np.isfinite(latency_samples)) and np.all(latency_samples > 0)
        assert np.isclose(latency_samples.mean(), row["latency_ms"], rtol=1e-6)
        assert row["peak_memory_bytes"] >= row["baseline_memory_bytes"] > 0
        source = row["measurement_source"]
        tables.append({"merged_record_index": index, "initial_context_length": row["initial_context_length"],
            "batch_size": row["batch_size"], "step": row["step"], "context_length": row["context_length"],
            "layer": row["model_layers"][0], "latency_ms": row["latency_ms"],
            "sample_median_ms": float(np.median(latency_samples)), "sample_max_ms": float(latency_samples.max()),
            "latency_over_group_layer_median": row["latency_ms"] / baselines[group_key(row)],
            "sample_max_over_sample_median": float(latency_samples.max() / np.median(latency_samples)),
            "avg_power_w": row["avg_power_w"], "peak_memory_gib": row["peak_memory_bytes"] / 2**30,
            "measurement_started_at_utc": row.get("measurement_started_at_utc", ""),
            "measurement_finished_at_utc": row.get("measurement_finished_at_utc", ""),
            "source_path": source.get("path", str(MERGED_FILE)),
            "source_record_index": source.get("record_index", ""),
            "review_cue_only": row["latency_ms"] / baselines[group_key(row)] > REVIEW_RATIO})
    for key, values in groups.items():
        latencies = np.array([row["latency_ms"] for row in values])
        summaries.append({"initial_context_length": key[0], "batch_size": key[1], "layer": key[2],
            "records": len(values), "latency_median_ms": float(np.median(latencies)),
            "latency_p10_ms": float(np.percentile(latencies, 10)), "latency_p90_ms": float(np.percentile(latencies, 90)),
            "power_median_w": float(np.median([row["avg_power_w"] for row in values])),
            "memory_median_gib": float(np.median([row["peak_memory_bytes"] for row in values]) / 2**30)})
    candidates = sorted([row for row in tables if row["review_cue_only"] and row["measurement_started_at_utc"]],
                        key=lambda row: row["measurement_started_at_utc"])
    write_csv(output / "decode_records.csv", tables)
    write_csv(output / "decode_group_layer_summary.csv", summaries)
    write_csv(output / "decode_timestamped_review_windows.csv", [{"figure_marker": i, **row}
              for i, row in enumerate(candidates, 1)])
    overview(groups, output)
    steps(groups, output, "latency_ms", "Layer latency (ms)", "decode_latency_by_step", (0, 4.1))
    steps(groups, output, "avg_power_w", "Board power (W)", "decode_power_by_step", (180, 540))
    sample_diagnostics(rows, output)
    memory_figure(groups, metadata["model_id"]["structure"], metadata["settings"], output)
    timeline(rows, baselines, output)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(content[name]).hexdigest()}
                   for name, path in sources.items()],
        "records": len(rows), "legacy_records_without_per_record_utc": sum(not row["measurement_started_at_utc"] for row in tables),
        "timestamped_records": sum(bool(row["measurement_started_at_utc"]) for row in tables),
        "exploratory_review_ratio": REVIEW_RATIO,
        "review_cues_all_records": sum(row["review_cue_only"] for row in tables),
        "timestamped_review_windows": len(candidates),
        "sample_max_above_2x_within_record_median": sum(row["sample_max_over_sample_median"] > 2 for row in tables),
        "interpretation": "No deletion or correction. Review cues are not evidence of external interference. UTC intervals cover warmup, latency and separate power replay, not latency alone.",
        "overview_bands": "10th–90th percentiles over 64 steps; not confidence intervals; step values are means of 30 replays",
        "excluded_run_comparison": "Unavailable: excluded GPU-sharing run samples were removed during profile consolidation; exclusion audit is retained in profile.json.",
        "table_views": ["decode_records.csv", "decode_group_layer_summary.csv",
                        "decode_timestamped_review_windows.csv"]}
    with (output / "decode_analysis_manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    for name, path in sources.items():
        assert path.read_bytes() == content[name], f"Input changed during analysis: {path}"
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved analysis: {output}")


if __name__ == "__main__":
    main()
