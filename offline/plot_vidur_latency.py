"""绘制 A100/Llama-2-7B/TP1 Vidur 预测单层时延；固定路径，无 CLI。"""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution.computation import A100VidurLatencyModel

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/analyse/vidur_predict"
BATCHES = (1, 32, 64, 128)
CONTEXTS = (128, 512, 1024, 2048)
CHUNKS = (64, 128, 256, 512)
INPUT_LENGTH = 2048
DECODE_STEPS = 64
TP = 1
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8357c5")
MARKERS = ("s", "o", "^", "v")
COMMON_COMPONENTS = (
    "input_layernorm", "attn_pre_proj", "attn_rope", "attn_post_proj",
    "post_attention_layernorm", "mlp_up_proj", "mlp_act", "mlp_down_proj", "add",
)


def predict_layer(model, phase, batch, query, past):
    # 直接使用组件预测器保留真实历史 KV；任务接口仅支持历史 KV=0 的完整 Prefill。
    features = {
        "num_tokens": batch * query,
        "batch_size": batch,
        "kv_cache_size": past,
        "prefill_chunk_size_squared": query ** 2,
    }
    components = (*COMMON_COMPONENTS, f"attn_{phase}")
    latency = sum(model._predict_component(TP, component, features) for component in components)
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError(f"Invalid predicted latency: {phase}, {features}, {latency}")
    attention_measured = (
        (TP, batch, query, past) in model.prefill_shapes
        if phase == "prefill" else (TP, batch, past) in model.decode_shapes
    )
    return latency, (TP, batch * query) in model.token_shapes and attention_measured


def predict_tables(model):
    decode_samples = []
    decode_means = []
    for batch in BATCHES:
        for context in CONTEXTS:
            rows = []
            for step in range(DECODE_STEPS):
                latency, measured = predict_layer(model, "decode", batch, 1, context + step)
                rows.append({
                    "batch_size": batch, "initial_context_length": context,
                    "step": step, "context_length": context + step,
                    "latency_ms": latency, "in_measured_domain": measured,
                })
            decode_samples.extend(rows)
            decode_means.append({
                "batch_size": batch, "initial_context_length": context,
                "steps": DECODE_STEPS,
                "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
                "in_domain_steps": sum(row["in_measured_domain"] for row in rows),
                "out_of_domain_steps": sum(not row["in_measured_domain"] for row in rows),
            })
    prefill = []
    for chunk in CHUNKS:
        for index, past in enumerate(range(0, INPUT_LENGTH, chunk)):
            latency, measured = predict_layer(model, "prefill", 1, chunk, past)
            prefill.append({
                "batch_size": 1, "chunk_size": chunk, "chunk_index": index,
                "context_length": past, "latency_ms": latency,
                "in_measured_domain": measured,
            })
    return decode_samples, decode_means, prefill


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency(rows, phase):
    is_decode = phase == "decode"
    group = "batch_size" if is_decode else "chunk_size"
    x_field = "initial_context_length" if is_decode else "context_length"
    y_field = "mean_latency_ms" if is_decode else "latency_ms"
    groups = BATCHES if is_decode else CHUNKS
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for value, color, marker in zip(groups, COLORS, MARKERS):
        selected = [row for row in rows if row[group] == value]
        ax.plot(
            [row[x_field] for row in selected], [row[y_field] for row in selected],
            label=f"{'Batch' if is_decode else 'Chunk'} {value}",
            color=color, linestyle="-", linewidth=4.5,
            marker=marker, markersize=16, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.8,
        )
    ax.set_xlabel("Context length", labelpad=12)
    ax.set_ylabel("Latency (ms)", labelpad=12)
    ax.set_xticks(CONTEXTS if is_decode else (0, 512, 1024, 1536, 2048))
    ax.set_xlim((30, 2170) if is_decode else (-80, 2128))
    ax.set_ylim(0, max(row[y_field] for row in rows) * 1.10)
    ax.grid(axis="both", color="#bdbdbd", linewidth=1.0, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=6, width=1.3)
    legend = ax.legend(
        loc="upper left" if is_decode else "lower right", ncol=2, frameon=True, fancybox=False,
        edgecolor="black", facecolor="white", framealpha=1,
        fontsize=20, columnspacing=1.2, handlelength=2.0, borderpad=0.5,
    )
    legend.get_frame().set_linewidth(1.3)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.97)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{phase}_latency.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    model = A100VidurLatencyModel()
    decode_samples, decode_means, prefill = predict_tables(model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "decode_predictions.csv", decode_samples)
    write_csv(OUTPUT_DIR / "decode_means.csv", decode_means)
    write_csv(OUTPUT_DIR / "prefill_predictions.csv", prefill)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 24, "axes.labelsize": 28,
        "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 20,
        "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
        "ytick.color": "black", "axes.spines.top": True, "axes.spines.right": True,
        "axes.linewidth": 1.5, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    plot_latency(decode_means, "decode")
    plot_latency(prefill, "prefill")
    manifest_path = model.profile_root / "manifest.json"
    metadata = {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "device_type": "a100", "dtype": "float16", "tensor_parallel_size": TP,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "components": list(COMMON_COMPONENTS) + ["attn_prefill or attn_decode"],
        "latency_scope": "Sum of predicted component median latencies for one decoder layer; not measured end-to-end latency or full-model latency. No KV-save, communication, CPU, embedding, final head or scheduling costs.",
        "decode": {
            "batches": list(BATCHES), "initial_contexts": list(CONTEXTS),
            "steps_per_point": DECODE_STEPS,
            "aggregation": "Arithmetic mean of 64 predictions at historical KV lengths C0 through C0+63. Same predictor for all layers; no artificial replication over layers 0,14,27.",
            "prediction_count": len(decode_samples),
            "out_of_domain_predictions": sum(not row["in_measured_domain"] for row in decode_samples),
        },
        "prefill": {
            "batch_size": 1, "chunk_sizes": list(CHUNKS), "input_length": INPUT_LENGTH,
            "x_axis": "Historical KV length before current chunk, from 0 to 2048-chunk_size.",
            "prediction_count": len(prefill),
            "out_of_domain_predictions": sum(not row["in_measured_domain"] for row in prefill),
            "adapter": "Component predictor queried directly with actual historical KV. Does not change full-prompt Prefill semantics of execution/computation.py.",
        },
        "limitations": [
            "RF predictions outside exact measured combinations are retained and flagged; extrapolation accuracy is not validated.",
            "Reference charts use Qwen2.5-7B-Instruct on RTX 5090; these predictions use Llama-2-7B on A100. Matching grids/style do not constitute accuracy validation.",
            "No power or memory curves: exported component predictors only provide latency.",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(decode_samples)} Decode predictions, {len(decode_means)} Decode means and {len(prefill)} Prefill predictions to {OUTPUT_DIR}")
    print(f"Out-of-domain: Decode={metadata['decode']['out_of_domain_predictions']}; Prefill={metadata['prefill']['out_of_domain_predictions']}")


if __name__ == "__main__":
    main()
