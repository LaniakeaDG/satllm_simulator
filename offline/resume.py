"""校验并复用已有 Profile；源结果只读，新结果保留逐条来源。"""

import json
import math
from pathlib import Path

from .workload import file_sha256


def prefill_cases(request, chunk_size):
    tokens = request["input_ids"]
    for chunk_index, offset in enumerate(range(0, len(tokens), chunk_size)):
        yield {"request_id": request["request_id"], "dataset_index": request["dataset_index"],
               "chunk_size": chunk_size, "chunk_index": chunk_index, "batch_size": 1,
               "input_length": len(tokens), "context_length": offset,
               "query_tokens": min(chunk_size, len(tokens) - offset)}


def decode_case(requests, batch, context, step):
    return {"batch_size": batch, "initial_context_length": context,
            "context_length": context + step, "step": step, "query_tokens": 1,
            "request_ids": [request["request_id"] for request in requests[:batch]],
            "dataset_indices": [request["dataset_index"] for request in requests[:batch]]}


def record_key(phase, layer, case):
    if phase == "prefill":
        return (phase, layer, case["dataset_index"], case["chunk_size"], case["chunk_index"])
    return (phase, layer, case["initial_context_length"], case["batch_size"], case["step"])


def expected_records(prefill, decode, settings):
    cases = [("prefill", case) for chunk in settings["prefill_chunk_sizes"]
             for request in prefill for case in prefill_cases(request, chunk)]
    cases += [("decode", decode_case(requests, batch, context, step))
              for context, requests in decode.items() for batch in settings["decode_batch_sizes"]
              for step in range(settings["decode_steps"])]
    return {record_key(phase, layer, case): case for phase, case in cases
            for layer in settings["model_layers"]}


def valid_values(row, settings):
    for field in ("latency_ms", "avg_power_w", "power_window_s"):
        value = row.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            return False
    samples = row.get("latency_samples_ms")
    if (not isinstance(samples, list) or len(samples) != settings["repeats"]
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in samples)):
        return False
    if not math.isclose(sum(samples) / len(samples), row["latency_ms"], rel_tol=1e-6):
        return False
    for field in ("peak_memory_bytes", "baseline_memory_bytes", "power_samples", "power_replay_block"):
        if type(row.get(field)) is not int or row[field] <= 0:
            return False
    return (row["peak_memory_bytes"] >= row["baseline_memory_bytes"]
            and row["power_samples"] >= 2
            and row["power_window_s"] >= settings["power_measure_s"])


def load_resume(paths, name, config, settings, environment, dataset, expected):
    """仅接受兼容且未报告 GPU 共享的来源；failed 不等于每条记录无效。"""
    retained, audit = {}, {"sources": [], "rejected_records": [], "reused_records": 0}
    operational = {"output_dir", "resume_from", "decode_prefill_batch_size"}
    for phase, path in paths.items():
        path = Path(path)
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if "schema_version" in data:
            data = data[phase]
        metadata = data["metadata"]
        if metadata.get("phase") != phase or metadata.get("model_id") != {"name": name, "structure": config}:
            raise ValueError(f"续测来源模型或阶段不匹配：{path}")
        previous = metadata["settings"]
        if ({key: value for key, value in previous.items() if key not in operational}
                != {key: value for key, value in settings.items() if key not in operational}):
            raise ValueError(f"续测来源的测量/输入参数不匹配：{path}")
        for field in ("uuid", "torch", "cuda", "dependencies", "device_type"):
            if metadata["environment"].get(field) != environment.get(field):
                raise ValueError(f"续测来源的环境 {field} 不匹配：{path}")
        if metadata.get("dataset") != dataset:
            raise ValueError(f"续测来源的数据摘要或请求选择不匹配：{path}")
        if (metadata.get("status") not in ("failed", "complete")
                or metadata["environment"].get("other_compute_processes")):
            raise ValueError(f"续测来源状态或启动占用不符合条件：{path}")
        error = metadata.get("error") or ""
        if "其他 GPU" in error or "other GPU" in error:
            raise ValueError(f"来源报告 GPU 干扰且没有逐记录时间线，不自动复用：{path}")
        info = {"path": str(path), "sha256": file_sha256(path), "status": metadata["status"],
                "error": metadata.get("error"), "records": len(data["records"]),
                "started_at_utc": metadata.get("started_at_utc"),
                "finished_at_utc": metadata.get("finished_at_utc"),
                "occupancy_evidence": "legacy power-window boundary checks only; brief intervening processes cannot be ruled out"}
        audit["sources"].append(info)
        for index, row in enumerate(data["records"]):
            try:
                layers = row["model_layers"]
                if len(layers) != 1 or type(layers[0]) is not int:
                    raise ValueError("invalid layer")
                key = record_key(phase, layers[0], row)
                case = expected[key]
                valid = (row["model_id"] == name and row["device_type"] == environment["device_type"]
                         and all(row.get(field) == value for field, value in case.items())
                         and valid_values(row, settings))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                audit["rejected_records"].append({"path": str(path), "record_index": index,
                                                  "reason": "invalid values or unexpected case"})
                continue
            if key in retained:
                raise ValueError(f"续测来源含重复 case：{path}, {key}")
            retained[key] = row | {"measurement_source": {
                "path": str(path), "sha256": info["sha256"], "record_index": index,
                "occupancy_evidence": info["occupancy_evidence"],
            }}
    audit["reused_records"] = len(retained)
    audit["missing_records"] = len(expected) - len(retained)
    return retained, audit
