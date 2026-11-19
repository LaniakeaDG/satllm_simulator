"""逐层加载本地真实权重，测量 GPU 上 chunked Prefill 和单 token Decode。

固定读取 configs/profile.yaml，不下载模型，不选择数据集。当前提供显式启用的
synthetic_hidden_states 微基准输入；这不是整模型、文本数据集或端到端推理测量。
"""

import csv
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .decoder import DecoderLayer, WeightStore, causal_mask, read_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs/profile.yaml"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def project_path(value):
    return (PROJECT_ROOT / value).resolve()


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Profile 配置必须为 YAML 映射。")
    if settings.get("input_mode") != "synthetic_hidden_states":
        raise ValueError("输入数据待定：请先明确输入方案；微基准可显式设置 input_mode: synthetic_hidden_states。")
    if settings.get("dtype") not in DTYPES:
        raise ValueError(f"dtype 必须为 {list(DTYPES)} 中的一项。")
    for key, minimum in (("prefill_total_tokens", 1), ("prefill_batch_size", 1),
                         ("warmup", 0), ("repeats", 1), ("seed", 0)):
        if type(settings.get(key)) is not int or settings[key] < minimum:
            raise ValueError(f"{key} 必须显式设置为不小于 {minimum} 的整数。")
    for key, minimum in (("prefill_chunk_sizes", 1), ("decode_batch_sizes", 1),
                         ("decode_cache_lengths", 0)):
        values = settings.get(key)
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or value < minimum for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{key} 必须是非空、无重复、不小于 {minimum} 的整数列表。")
    if type(settings.get("allow_busy_gpu")) is not bool:
        raise ValueError("allow_busy_gpu 必须为布尔值。")
    if not isinstance(settings.get("output_dir"), str) or not settings["output_dir"]:
        raise ValueError("output_dir 必须是非空路径。")
    device = torch.device(settings.get("device", "cuda:0"))
    if device.type != "cuda":
        raise ValueError("性能 Profile 仅支持 CUDA GPU，不以 CPU 时间代替 GPU 结果。")
    models = settings.get("models")
    if not isinstance(models, dict):
        raise ValueError("models 必须是模型名称到本地目录的映射，null 表示尚未配置。")
    selected = {}
    for name, directory in models.items():
        if directory is None:
            continue
        if (not isinstance(name, str) or not name or name in (".", "..")
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name)):
            raise ValueError("模型名称只能包含英文字母、数字、下划线和连字符。")
        if not isinstance(directory, str) or not directory:
            raise ValueError(f"{name} 的模型目录必须为非空路径或 null。")
        model_dir = project_path(directory)
        config = read_model_config(model_dir)
        required = max(settings["prefill_total_tokens"], max(settings["decode_cache_lengths"]) + 1)
        if required > config["max_position_embeddings"]:
            raise ValueError(f"{name} 的输入长度超过模型 max_position_embeddings。")
        output = project_path(settings["output_dir"]) / name
        if output.exists():
            raise FileExistsError(f"已有结果不覆盖：{output}")
        selected[name] = (model_dir, config)
    if not selected:
        raise ValueError("尚未配置本地模型目录；请填写 models 中至少一个模型的路径。")
    return selected


def gpu_environment(device, allow_busy):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请安装匹配驱动的 CUDA 版 PyTorch。")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if not getattr(properties, "uuid", None):
        raise RuntimeError("无法确定当前 CUDA 设备的 UUID，不能可靠检查 GPU 占用。")
    command = ["nvidia-smi", f"--id={properties.uuid}",
               "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("无法通过 nvidia-smi 检查目标 GPU 占用，停止测量。") from error
    processes = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    others = [row for row in processes if row and row[0].strip() != str(os.getpid())]
    if others and not allow_busy:
        raise RuntimeError(f"目标 GPU 存在其他计算进程，不运行 Profile：{others}")
    return {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": properties.name, "uuid": str(properties.uuid),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor], "other_compute_processes": others,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def allocate_cache(config, batch, capacity, device, dtype, prefix_length=0):
    shape = (batch, config["num_key_value_heads"], capacity, config["head_dim"])
    cache = (torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype))
    for tensor in cache:
        if prefix_length:
            tensor[:, :, :prefix_length].normal_()
    return cache


def benchmark_forward(layer, hidden, cache, offset, mask, settings, device):
    """重复覆盖当前 chunk/单 token 的 KV 槽，不改变历史前缀与有效上下文长度。"""
    with torch.inference_mode():
        for _ in range(settings["warmup"]):
            layer(hidden, cache, offset, mask)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(settings["repeats"])]
        ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
        for start, end in zip(starts, ends):
            start.record()
            layer(hidden, cache, offset, mask)
            end.record()
        torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return samples, baseline, torch.cuda.max_memory_allocated(device)


def profile_cases(layer, config, settings, device, dtype):
    """Prefill 沿整段输入逐块推进；Decode 独立扫描指定 batch 和历史长度。"""
    total = settings["prefill_total_tokens"]
    batch = settings["prefill_batch_size"]
    # 同一层的四种 chunk size 使用相同输入，各自从空 KV 开始。
    hidden = torch.randn(batch, total, config["hidden_size"], device=device, dtype=dtype)
    for chunk_size in settings["prefill_chunk_sizes"]:
        cache = allocate_cache(config, batch, total, device, dtype)
        for chunk_index, offset in enumerate(range(0, total, chunk_size)):
            length = min(chunk_size, total - offset)
            mask = causal_mask(offset, length, device)
            values = benchmark_forward(layer, hidden[:, offset:offset + length], cache, offset,
                                       mask, settings, device)
            yield {
                "phase": "prefill", "batch_size": batch, "prefill_total_tokens": total,
                "chunk_size": chunk_size, "chunk_index": chunk_index,
                "query_tokens": length, "cache_length": offset, "cache_capacity": total,
            }, values
            del mask
        del cache
    del hidden
    for cache_length in settings["decode_cache_lengths"]:
        for batch in settings["decode_batch_sizes"]:
            hidden = torch.randn(batch, 1, config["hidden_size"], device=device, dtype=dtype)
            cache = allocate_cache(config, batch, cache_length + 1, device, dtype, cache_length)
            values = benchmark_forward(layer, hidden, cache, cache_length, None, settings, device)
            yield {
                "phase": "decode", "batch_size": batch, "prefill_total_tokens": None,
                "chunk_size": None, "chunk_index": None, "query_tokens": 1,
                "cache_length": cache_length, "cache_capacity": cache_length + 1,
            }, values
            del hidden, cache


def statistics_row(name, layer_index, case, values, dtype):
    samples, baseline, peak = values
    return {
        "model": name, "layer_index": layer_index, "dtype": dtype, **case,
        "samples": len(samples), "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples), "stddev_ms": statistics.pstdev(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "baseline_allocated_mib": baseline / 1024**2, "peak_allocated_mib": peak / 1024**2,
        "peak_increment_mib": (peak - baseline) / 1024**2,
    }


def profile_model(name, model_dir, config, settings, device, environment):
    store = WeightStore(model_dir)
    output = project_path(settings["output_dir"]) / name
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": "real_weight_single_decoder_layer_synthetic_hidden_states",
        "model": name, "model_dir": str(model_dir), "model_config": config,
        "settings": settings, "environment": environment, "completed_cases": 0,
        "timing": "CUDA events; PyTorch SDPA auto backend; eager execution; no torch.compile",
        "included": ["RMSNorm", "QKV projections", "RoPE", "static KV writes", "attention", "MLP", "residuals"],
        "excluded": ["weight loading", "input/cache allocation and initialization", "mask creation",
                     "embeddings", "final norm", "LM head", "sampling", "cross-layer execution"],
        "input_scope": "Random layer inputs; Prefill KV computed from preceding chunks of the same input; Decode historical KV is synthetic.",
        "cache_policy": "Preallocated per case, in-place append; warmup and repeats overwrite only current slots, preserving the same prefix.",
        "memory_scope": "PyTorch allocated bytes for one resident layer, full input, reserved-capacity KV tensors, mask and temporaries; not full-model or nvidia-smi memory.",
        "power_measured": False,
    }
    try:
        dtype = DTYPES[settings["dtype"]]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("目标 GPU 不支持 BF16，请显式调整 dtype。")
        torch.manual_seed(settings["seed"])
        layer = DecoderLayer(config, dtype).to(device).eval().requires_grad_(False)
        with (output / "layers.csv").open("x", encoding="utf-8", newline="") as table, \
                (output / "samples.jsonl").open("x", encoding="utf-8") as raw:
            writer = None
            for layer_index in range(config["num_hidden_layers"]):
                store.load_layer(layer, layer_index)
                torch.cuda.synchronize(device)
                for case, values in profile_cases(layer, config, settings, device, dtype):
                    row = statistics_row(name, layer_index, case, values, settings["dtype"])
                    if writer is None:
                        writer = csv.DictWriter(table, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    raw.write(json.dumps({"model": name, "layer_index": layer_index, **case,
                                          "elapsed_ms": values[0]}, ensure_ascii=False) + "\n")
                    table.flush()
                    raw.flush()
                    metadata["completed_cases"] += 1
                print(f"{name}: layer {layer_index + 1}/{config['num_hidden_layers']} 完成", flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (output / "metadata.json").open("x", encoding="utf-8") as destination:
            json.dump(metadata, destination, ensure_ascii=False, indent=2)
    return output


def main():
    settings = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    selected = validate_settings(settings)
    device = torch.device(settings["device"])
    environment = gpu_environment(device, settings["allow_busy_gpu"])
    for name, (model_dir, config) in selected.items():
        output = profile_model(name, model_dir, config, settings, device, environment)
        print(f"已保存：{output}", flush=True)


if __name__ == "__main__":
    main()
