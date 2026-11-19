"""ShareGPT 驱动的真实逐层 Profile：chunked Prefill 与 KV 持续增长的 Decode。"""

import csv
import json
import os
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .decoder import DecoderLayer, RMSNorm, WeightStore, causal_mask, read_model_config
from .power import PowerMeter
from .workload import load_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs/profile.yaml"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def project_path(value):
    return (PROJECT_ROOT / value).resolve()


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Profile 配置必须为 YAML 映射。")
    for key in ("sample_requests", "max_input_tokens", "decode_steps", "decode_prefill_chunk_size", "repeats"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} 必须为正整数。")
    if type(settings.get("warmup")) is not int or settings["warmup"] < 0:
        raise ValueError("warmup 必须为非负整数。")
    for key in ("prefill_chunk_sizes", "decode_batch_sizes", "decode_context_lengths"):
        values = settings.get(key)
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(type(value) is not int or value < 1 for value in values)):
            raise ValueError(f"{key} 必须为非空、无重复的正整数列表。")
    if max(settings["decode_batch_sizes"]) > settings["sample_requests"]:
        raise ValueError("Decode batch size 不能超过每种上下文长度的请求池大小。")
    if max(settings["decode_context_lengths"]) > settings["max_input_tokens"]:
        raise ValueError("Decode 初始上下文长度不能超过 max_input_tokens。")
    for key in ("power_settle_s", "power_measure_s", "power_sample_interval_s"):
        value = settings.get(key)
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"{key} 必须为正有限数。")
    if settings["power_measure_s"] < 1 or settings["power_settle_s"] < 1:
        raise ValueError("NVML 功率采样和稳态准备窗口均至少为 1 秒。")
    if settings.get("dtype") not in DTYPES or torch.device(settings["device"]).type != "cuda":
        raise ValueError("需要浮点 dtype 和 CUDA device。")
    dataset = project_path(settings["dataset"])
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    models = settings.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("models 必须为模型名到本地目录的非空映射。")
    selected = {}
    for name, directory in models.items():
        if directory is None:
            continue
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("模型名称只能包含英文字母、数字、点、下划线和连字符。")
        model_dir = project_path(directory)
        config = read_model_config(model_dir)
        if max(settings["max_input_tokens"], max(settings["decode_context_lengths"]) + settings["decode_steps"]) > config["max_position_embeddings"]:
            raise ValueError(f"{name} 的输入与生成长度超过模型位置编码上限。")
        selected[name] = (model_dir, config)
    if not selected:
        raise ValueError("没有配置本地模型。")
    return selected


def gpu_environment(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if not uuid:
        raise RuntimeError("无法确定目标 GPU UUID，停止测量。")
    try:
        command = ["nvidia-smi", f"--id={uuid}", "--query-compute-apps=pid,process_name,used_gpu_memory",
                   "--format=csv,noheader,nounits"]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("无法检查目标 GPU 占用，停止测量。") from error
    others = [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True)
              if row and row[0].strip() != str(os.getpid())]
    if others:
        raise RuntimeError(f"目标 GPU 存在其他计算进程，不运行时延/功率测量：{others}")
    return {
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "device_type": properties.name, "uuid": str(uuid),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor], "other_compute_processes": others,
    }


class StreamingModel:
    """真实层间传递，CPU 保存权重，GPU 仅驻留当前层；所有层 KV 保留在 GPU。"""

    def __init__(self, model_dir, config, device, dtype):
        self.config, self.device, self.dtype = config, device, dtype
        store = WeightStore(model_dir)
        with torch.device("meta"):
            self.layer = DecoderLayer(config, dtype)
        self.layer.to_empty(device=device).eval().requires_grad_(False)
        self.weights = []
        for index in range(config["num_hidden_layers"]):
            tensors = {}
            for name, parameter in self.layer.named_parameters():
                module = name.split(".")[0]
                prefix = "self_attn." if module in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm") else (
                    "mlp." if module in ("gate_proj", "up_proj", "down_proj") else "")
                tensor = store.read_tensor(f"model.layers.{index}.{prefix}{name}")
                if tensor.shape != parameter.shape:
                    raise ValueError(f"第 {index} 层权重形状不匹配：{name}")
                tensors[name] = tensor.to(dtype=dtype)
            self.weights.append(tensors)
        self.embedding = store.read_tensor("model.embed_tokens.weight").to(dtype=dtype)
        self.head = self.embedding if config.get("tie_word_embeddings") else store.read_tensor("lm_head.weight").to(dtype=dtype)
        self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-6), dtype).to(device)
        self.norm.requires_grad_(False)
        with torch.inference_mode():
            self.norm.weight.copy_(store.read_tensor("model.norm.weight"))
        self.loaded_layer = None

    def activate(self, index):
        if self.loaded_layer == index:
            return
        with torch.inference_mode():
            for name, parameter in self.layer.named_parameters():
                parameter.copy_(self.weights[index][name])
        self.loaded_layer = index

    def new_cache(self, batch, capacity):
        shape = (batch, self.config["num_key_value_heads"], capacity, self.config["head_dim"])
        return [(torch.empty(shape, device=self.device, dtype=self.dtype),
                 torch.empty(shape, device=self.device, dtype=self.dtype))
                for _ in range(self.config["num_hidden_layers"])]

    def forward(self, input_ids, caches, offset, measure=None):
        hidden = torch.nn.functional.embedding(input_ids.cpu(), self.embedding).to(self.device)
        mask = causal_mask(offset, input_ids.shape[1], self.device)
        for index, cache in enumerate(caches):
            self.activate(index)
            operation = lambda: self.layer(hidden, cache, offset, mask)
            if measure is not None:
                measure(index, operation)
            hidden = operation()
        return hidden

    def next_token(self, hidden):
        # Embedding、最终 norm、LM head 与 greedy argmax 用于真实生成，但不计入层 Profile。
        head = self.head.to(self.device)
        logits = torch.nn.functional.linear(self.norm(hidden[:, -1:]), head)
        return logits[:, -1].argmax(-1, keepdim=True).cpu()


def benchmark_forward(forward, meter, device, settings):
    for _ in range(settings["warmup"]):
        forward()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(settings["repeats"])]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    for start, end in zip(starts, ends):
        start.record()
        forward()
        end.record()
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    peak = torch.cuda.max_memory_allocated(device)
    latency = statistics.fmean(samples)
    power = meter.measure(forward, device, latency, settings)
    return {"latency_ms": latency, **power, "peak_memory_bytes": peak,
            "baseline_memory_bytes": baseline, "latency_samples_ms": samples}


def append_record(destination, record, first):
    if not first:
        destination.write(",\n")
    json.dump(record, destination, ensure_ascii=False, allow_nan=False)
    destination.flush()


def run_prefill(model, requests, settings, record):
    for chunk_size in settings["prefill_chunk_sizes"]:
        for request in requests:
            tokens = request["input_ids"]
            caches = model.new_cache(1, len(tokens))
            for chunk_index, offset in enumerate(range(0, len(tokens), chunk_size)):
                ids = torch.tensor([tokens[offset:offset + chunk_size]], dtype=torch.long)
                case = {"request_id": request["request_id"], "dataset_index": request["dataset_index"],
                        "chunk_size": chunk_size, "chunk_index": chunk_index, "batch_size": 1,
                        "input_length": len(tokens), "context_length": offset, "query_tokens": ids.shape[1]}
                model.forward(ids, caches, offset, lambda index, operation: record(index, case, operation))
            del caches
            print(f"Prefill chunk={chunk_size}, request={request['dataset_index']} 完成", flush=True)


def run_decode(model, pools, settings, record):
    for context, requests in pools.items():
        for batch in settings["decode_batch_sizes"]:
            chosen = requests[:batch]
            ids = torch.tensor([request["input_ids"] for request in chosen], dtype=torch.long)
            caches = model.new_cache(batch, context + settings["decode_steps"])
            # 用真实请求、真实层间激活填满所有层的历史 KV，不将准备阶段计入 Decode。
            for offset in range(0, context, settings["decode_prefill_chunk_size"]):
                hidden = model.forward(ids[:, offset:offset + settings["decode_prefill_chunk_size"]], caches, offset)
            token = model.next_token(hidden)
            del hidden
            for step in range(settings["decode_steps"]):
                case = {"batch_size": batch, "initial_context_length": context,
                        "context_length": context + step, "step": step, "query_tokens": 1,
                        "request_ids": [request["request_id"] for request in chosen],
                        "dataset_indices": [request["dataset_index"] for request in chosen]}
                hidden = model.forward(token, caches, context + step,
                                       lambda index, operation: record(index, case, operation))
                token = model.next_token(hidden)
                del hidden
                print(f"Decode batch={batch}, context={context + step}, step={step + 1}/{settings['decode_steps']} 完成", flush=True)
            del caches


def output_paths(name, device_type, settings):
    device_name = re.sub(r"[^A-Za-z0-9.-]+", "_", device_type).strip("_")
    base = project_path(settings["output_dir"]) / f"{name}_{device_name}"
    return {phase: base.with_name(base.name + f"_{phase}.json") for phase in ("prefill", "decode")}


def profile_model(name, model_dir, config, settings, device, environment):
    paths = output_paths(name, environment["device_type"], settings)
    # 同目录的每阶段锁文件也用于标记硬中断后的不完整运行；不自动覆盖或删除。
    locks = {phase: path.with_suffix(".json.lock") for phase, path in paths.items()}
    for path in [*paths.values(), *locks.values()]:
        if path.exists():
            raise FileExistsError(f"已有结果或未完成运行不覆盖：{path}")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    acquired = []
    try:
        for path in locks.values():
            with path.open("x") as destination:
                destination.write(str(os.getpid()))
            acquired.append(path)
    except BaseException:
        for path in acquired:
            path.unlink()
        raise
    metadata = {
        "model_id": {"name": name, "structure": config}, "device_type": environment["device_type"],
        "environment": environment, "settings": settings, "model_dir": str(model_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing_scope": "single decoder layer eager SDPA; includes norms, RoPE, KV write, attention, MLP; excludes CPU weight copy, embedding, final norm/head/argmax, mask construction",
        "power_scope": "NVML total GPU board watts, no idle subtraction; separate steady replay of the same layer/case, not instantaneous per-token energy or full-model power",
        "memory_scope": "PyTorch process peak allocated GPU bytes during latency samples; all-layer preallocated KV + one resident decoder layer + live inputs/temporary tensors; excludes CPU weight storage; not per-layer-exclusive memory",
        "cache_policy": "all layers retain real KV; fixed-position replay overwrites current slots only; actual Decode advances one position after all layers; no EOS removal",
        "decode_semantics": "Prefill prepares the first generated token; each profiled step consumes the preceding generated token, appends KV and greedily generates the next token; context_length is before that append",
    }
    temporary = {phase: path.with_suffix(".json.partial") for phase, path in paths.items()}
    meter = None
    files = {}
    counts = {phase: 0 for phase in paths}
    status, failure = "failed", None
    try:
        prefill, decode, dataset = load_workload(project_path(settings["dataset"]), model_dir, settings)
        metadata["dataset"] = dataset
        if settings["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("目标 GPU 不支持 BF16。")
        meter = PowerMeter(environment["uuid"])
        model = StreamingModel(model_dir, config, device, DTYPES[settings["dtype"]])
        for phase, path in temporary.items():
            files[phase] = path.open("x", encoding="utf-8")
            files[phase].write('{"records":[\n')
        def record(phase, index, case, operation):
            values = benchmark_forward(operation, meter, device, settings)
            row = {"model_id": name, "device_type": environment["device_type"],
                   "model_layers": [index], **case, **values}
            append_record(files[phase], row, counts[phase] == 0)
            counts[phase] += 1
        with torch.inference_mode():
            run_prefill(model, prefill, settings, lambda *args: record("prefill", *args))
            run_decode(model, decode, settings, lambda *args: record("decode", *args))
        status = "complete"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata.update(status=status, error=failure, completed_records=counts,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        for phase, destination in files.items():
            destination.write('],"metadata":')
            json.dump(metadata | {"phase": phase}, destination, ensure_ascii=False, allow_nan=False)
            destination.write("}\n")
            destination.close()
            # 硬链接发布不覆盖，即使其他进程在检查后创建了目标文件也会失败。
            os.link(temporary[phase], paths[phase])
            temporary[phase].unlink()
        if meter is not None:
            meter.close()
        for path in acquired:
            path.unlink()
    return paths


def main():
    settings = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    selected = validate_settings(settings)
    device = torch.device(settings["device"])
    environment = gpu_environment(device)  # 忙 GPU 必须在分词、模型加载和输出创建之前拒绝。
    for name, (model_dir, config) in selected.items():
        for path in output_paths(name, environment["device_type"], settings).values():
            if path.exists():
                raise FileExistsError(f"已有结果不覆盖：{path}")
        paths = profile_model(name, model_dir, config, settings, device, environment)
        print(f"已保存：{paths}", flush=True)


if __name__ == "__main__":
    main()
