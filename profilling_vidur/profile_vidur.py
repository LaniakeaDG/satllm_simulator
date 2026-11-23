"""Fixed-grid Vidur-inspired component RF experiment, not original Sarathi/Vidur.
Run with /home/zdg/.conda/envs/orbitserve/bin/python <this file>.
All experiment outputs and local optional dependencies stay beside this script.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "dependencies"))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
import torch
from torch.nn import functional as F
import yaml
from offline.decoder import DecoderLayer, apply_rotary, causal_mask, read_model_config

WARMUP = 5
REPEATS = 30
SEED = 42
RF = dict(n_estimators=200, max_depth=None, min_samples_leaf=1,
          max_features=1.0, random_state=SEED, n_jobs=1)
PREFILL_CHUNKS = [64, 128, 256, 512]
PREFILL_CONTEXTS = [0, 512, 1024, 1536]
DECODE_BATCHES = [1, 32, 64, 128]
DECODE_CONTEXTS = [128, 512, 1024, 2048]
COMPONENTS = ["prepare_qkv", "attention", "output_mlp"]


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args], text=True).strip()
    processes = query(["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    foreign = [line for line in processes.splitlines() if line.split(",")[0].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f"Foreign GPU compute process detected: {foreign}")
    return {"utc": datetime.now(timezone.utc).isoformat(), "processes": processes,
            "device": query(["--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu", "--format=csv"])}


def prepare_qkv(layer, hidden, cache, offset):
    normalized = layer.input_layernorm(hidden)
    batch, length, _ = hidden.shape
    query = layer.q_proj(normalized).view(batch, length, layer.heads, layer.head_dim).transpose(1, 2)
    key = layer.k_proj(normalized).view(batch, length, layer.kv_heads, layer.head_dim).transpose(1, 2)
    value = layer.v_proj(normalized).view(batch, length, layer.kv_heads, layer.head_dim).transpose(1, 2)
    positions = torch.arange(offset, offset + length, device=hidden.device, dtype=torch.float32)
    frequencies = layer.theta ** (-torch.arange(0, layer.head_dim, 2, device=hidden.device,
                                               dtype=torch.float32) / layer.head_dim)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)[None, None]
    cosine, sine = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
    query, key = apply_rotary(query, cosine, sine), apply_rotary(key, cosine, sine)
    end = offset + length
    cache[0][:, :, offset:end].copy_(key)
    cache[1][:, :, offset:end].copy_(value)
    return query


def attention(layer, query, cache, offset, length, mask):
    return F.scaled_dot_product_attention(
        query, cache[0][:, :, :offset + length], cache[1][:, :, :offset + length],
        attn_mask=mask, is_causal=offset == 0 and length > 1,
        enable_gqa=layer.heads != layer.kv_heads)


def output_mlp(layer, hidden, attended):
    batch, length, _ = hidden.shape
    attended = attended.transpose(1, 2).reshape(batch, length, layer.heads * layer.head_dim)
    hidden = hidden + layer.o_proj(attended)
    normalized = layer.post_attention_layernorm(hidden)
    return hidden + layer.down_proj(F.silu(layer.gate_proj(normalized)) * layer.up_proj(normalized))


def measure(fn):
    # One eager call per event pair; synchronization is outside elapsed interval.
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return samples


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(row, component):
    if component != "attention":
        return [row["batch"] * row["chunk"]]
    if row["phase"] == "prefill":
        return [row["context"], row["chunk"] ** 2]
    return [row["batch"], row["context"]]


def metrics(rows):
    predicted = np.array([r["predicted_ms"] for r in rows])
    measured = np.array([r["measured_ms"] for r in rows])
    return {"cases": len(rows), "mape_percent": float(np.mean(np.abs(predicted - measured) / measured) * 100),
            "mae_ms": float(np.mean(np.abs(predicted - measured))),
            "rmse_ms": float(np.sqrt(np.mean((predicted - measured) ** 2))),
            "mean_signed_relative_error_percent": float(np.mean((predicted - measured) / measured) * 100),
            "max_absolute_relative_error_percent": float(np.max(np.abs(predicted - measured) / measured) * 100)}


@torch.inference_mode()
def main():
    started = datetime.now(timezone.utc).isoformat()
    before = gpu_snapshot()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config_file = ROOT / "configs/profile.yaml"
    model_dir = yaml.safe_load(config_file.read_text())["models"]["Qwen2.5-7B-Instruct"]
    config = read_model_config(model_dir)
    assert config["model_type"] == "qwen2", "Component split is specifically Qwen2.5."
    with torch.device("cuda:0"):
        layer = DecoderLayer(config, torch.bfloat16).eval()
    cases = []
    for chunk in PREFILL_CHUNKS:
        for context in PREFILL_CONTEXTS:
            cases.append(dict(phase="prefill", batch=1, chunk=chunk, context=context, holdout=chunk == 256))
    for batch in DECODE_BATCHES:
        for context in DECODE_CONTEXTS:
            cases.append(dict(phase="decode", batch=batch, chunk=1, context=context, holdout=batch == 32))
    # Seeded order reduces systematic shape/temperature ordering; it never selects RF hyperparameters.
    np.random.default_rng(SEED).shuffle(cases)
    raw, medians, snapshots = [], [], []
    for index, case in enumerate(cases):
        snapshots.append({"case": case, "when": "before", **gpu_snapshot()})
        batch, chunk, context = case["batch"], case["chunk"], case["context"]
        hidden = torch.randn(batch, chunk, config["hidden_size"], device="cuda", dtype=torch.bfloat16)
        cache = tuple(torch.randn(batch, layer.kv_heads, context + chunk, layer.head_dim,
                                  device="cuda", dtype=torch.bfloat16) for _ in range(2))
        mask = causal_mask(context, chunk, hidden.device)
        query = prepare_qkv(layer, hidden, cache, context)
        attended = attention(layer, query, cache, context, chunk, mask)
        split_output = output_mlp(layer, hidden, attended)
        reference = layer(hidden, cache, context, mask)
        torch.testing.assert_close(split_output, reference, rtol=0, atol=0)
        assert torch.isfinite(reference).all().item()
        functions = {
            "prepare_qkv": lambda: prepare_qkv(layer, hidden, cache, context),
            "attention": lambda: attention(layer, query, cache, context, chunk, mask),
            "output_mlp": lambda: output_mlp(layer, hidden, attended),
            "full_layer": lambda: layer(hidden, cache, context, mask),
        }
        for component, fn in functions.items():
            samples = measure(fn)
            for repeat, latency in enumerate(samples):
                raw.append({**case, "component": component, "repeat": repeat, "latency_ms": latency})
            medians.append({**case, "component": component, "median_ms": float(np.median(samples)),
                            "min_ms": min(samples), "max_ms": max(samples),
                            "p25_ms": float(np.percentile(samples, 25)), "p75_ms": float(np.percentile(samples, 75))})
        snapshots.append({"case": case, "when": "after", **gpu_snapshot()})
        print(f"{index + 1}/{len(cases)} {case}", flush=True)
        del functions, query, attended, split_output, reference, cache, hidden, mask
    write_csv("samples.csv", raw)
    write_csv("medians.csv", medians)
    predictions, audits, component_predictions = [], [], []
    for phase in ["prefill", "decode"]:
        models = {}
        for component in COMPONENTS:
            train = [r for r in medians if r["phase"] == phase and r["component"] == component and not r["holdout"]]
            # Collapse repeated token shapes to one median training label for token-only components.
            grouped = {}
            for row in train:
                grouped.setdefault(tuple(features(row, component)), []).append(row["median_ms"])
            x = list(grouped)
            y = [float(np.median(grouped[key])) for key in x]
            model = RandomForestRegressor(**RF).fit(x, y)
            models[component] = model
            joblib.dump(model, HERE / f"{phase}_{component}_rf.joblib")
            audits.append({"phase": phase, "component": component,
                           "features": "tokens" if component != "attention" else ("context,chunk_squared" if phase == "prefill" else "batch,context"),
                           "training_cases": len(train), "unique_training_features": [list(key) for key in x],
                           "training_labels_ms": y, "holdout_rows_used": 0})
        for case in cases:
            if case["phase"] != phase or not case["holdout"]:
                continue
            values = {}
            for component in COMPONENTS:
                predicted = float(models[component].predict([features(case, component)])[0])
                measured = next(r["median_ms"] for r in medians if all(r[k] == case[k] for k in case) and r["component"] == component)
                values[component] = predicted
                component_predictions.append({**case, "component": component, "predicted_ms": predicted, "measured_ms": measured,
                                              "absolute_error_ms": abs(predicted - measured), "relative_error": abs(predicted - measured) / measured})
            measured = next(r["median_ms"] for r in medians if all(r[k] == case[k] for k in case) and r["component"] == "full_layer")
            component_sum = sum(r["median_ms"] for r in medians if all(r[k] == case[k] for k in case) and r["component"] in COMPONENTS)
            predicted = sum(values.values())
            predictions.append({**case, "predicted_ms": predicted, "measured_ms": measured,
                                "absolute_error_ms": abs(predicted - measured), "relative_error": abs(predicted - measured) / measured,
                                "measured_component_sum_ms": component_sum,
                                "component_sum_relative_gap": (component_sum - measured) / measured})
    write_csv("holdout_predictions.csv", predictions)
    write_csv("holdout_component_predictions.csv", component_predictions)
    consistency = []
    for case in cases:
        rows = [r for r in medians if all(r[k] == case[k] for k in case)]
        total = sum(r["median_ms"] for r in rows if r["component"] in COMPONENTS)
        full = next(r["median_ms"] for r in rows if r["component"] == "full_layer")
        consistency.append({**case, "component_sum_ms": total, "full_layer_ms": full, "relative_gap": (total - full) / full})
    write_csv("component_consistency.csv", consistency)
    summary = {phase: metrics([r for r in predictions if r["phase"] == phase]) for phase in ["prefill", "decode"]}
    summary["overall"] = metrics(predictions)
    limitations = [
        "Vidur-inspired adaptation: real GPU component measurements plus independent per-phase random forests; not the original Vidur/Sarathi kernels or a reproduction of their numerical results.",
        "One reference Qwen2.5-7B BF16 decoder layer with seeded dummy nn.Linear default random parameters and RMSNorm ones. Only config.json is read; no checkpoint weights or old profile data are loaded. Synthetic activations and KV, not real model inference.",
        "Three exhaustive components: input RMSNorm + QKV projections/views + RoPE construction/application + KV copy; SDPA attention; attention transpose/reshape + output projection/residual + RMSNorm + SwiGLU/MLP/residual. All operations of DecoderLayer.forward are covered in order. Bitwise split-versus-original output equivalence verified for all 32 cases.",
        "Independent full-layer timing calls the unchanged offline.decoder.DecoderLayer.forward. Mask construction, input/cache allocation, parameter initialization are outside both component and full-layer timing, matching its forward API. KV writes overwrite the same fixed offset, not growing autoregressive execution.",
        "Eager CUDA event timing, five warmups and thirty individual repetitions, synchronized after each pair; no CUDA graphs, torch.compile, profiler, or forced SDPA backend. GPU stream elapsed time can contain CPU launch starvation gaps. It is not CPU wall-clock latency, full-model latency, or end-to-end serving latency.",
        "Separate component timing changes allocation lifetime, cache warmth and launch gaps; sums of medians need not equal the median full-layer latency. component_consistency.csv preserves this discrepancy, without correction or calibration.",
        "Prefill attention uses [historical_context, chunk_squared]; decode attention uses [batch, historical_context]. Token components use batch*chunk. Fixed configuration RF, no tuning/CV or holdout-based selection. All components exclude the entire respective held-out group; independent phase models prevent cross-phase token-shape leakage.",
        "Training has only twelve cases per phase, three distinct token counts for each token model, and twelve attention points. RF is piecewise constant and does not interpolate/extrapolate like an analytical model. Four held-out cases per phase do not establish broad generalization.",
        "GPU compute processes checked before/after every case; no competing process observed if run completes. Checks do not exclude short-lived processes between checks, graphics activity, thermal/clock changes or OS scheduling noise. No clocks locked and no samples filtered.",
        "relative_error denotes absolute_error_ms/measured_ms (fraction); MAPE summaries are percentages. Measurements and failed predictions are preserved, not optimized against holdout labels."
    ]
    metadata = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "command": f"/home/zdg/.conda/envs/orbitserve/bin/python {Path(__file__).resolve()}",
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "sklearn": sklearn.__version__, "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(), "gpu_initial": before, "gpu_snapshots": snapshots,
                "model_dir": str(model_dir), "model_config": config, "dtype": "bfloat16", "seed": SEED,
                "warmup": WARMUP, "repeats": REPEATS, "rf_parameters": RF,
                "cases": cases, "training_audit": audits, "summary": summary, "limitations": limitations,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "offline/decoder.py", config_file, Path(model_dir) / "config.json"]}}
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["Vidur-inspired single-layer profiling and group-holdout RF validation", "", json.dumps(summary, indent=2), "", *limitations, "", "Held-out cases:"]
    lines.extend(json.dumps(row) for row in predictions)
    (HERE / "report.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
