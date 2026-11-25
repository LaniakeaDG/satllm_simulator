"""基于 A100 Vidur Profile 的节点计算时间仿真。"""

import csv
import json
from functools import lru_cache
from math import isfinite
from pathlib import Path
import sys
import warnings


_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_ROOT = _ROOT / "profilling_vidur"
_DEPENDENCIES = _PROFILE_ROOT / "dependencies"
if str(_DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(_DEPENDENCIES))

import joblib
import numpy as np
import sklearn


_DEVICE = "a100"
_MODEL = "meta-llama/Llama-2-7b-hf"
_DTYPE = "float16"
_SUPPORTED_TP = {1, 2, 4, 8}
_ATTENTION_COMPONENTS = (
    "input_layernorm",
    "attn_pre_proj",
    "attn_rope",
    "attn_post_proj",
)
_MLP_COMPONENTS = (
    "post_attention_layernorm",
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "add",
)


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} 必须是正整数")
    return value


def _nonnegative_int(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} 必须是非负整数")
    return value


class A100VidurLatencyModel:
    """查询 A100/Llama-2-7B 的单层组件随机森林，所有输出均为毫秒。"""

    def __init__(self, profile_root=_PROFILE_ROOT):
        self.profile_root = Path(profile_root)
        manifest_path = profile_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scope = manifest.get("compute_scope", {})
        if (scope.get("device") != _DEVICE or scope.get("model") != _MODEL
                or set(scope.get("tp", [])) != _SUPPORTED_TP):
            raise ValueError("Profile manifest 不是预期的 A100/Llama-2-7B/TP1,2,4,8 数据")
        recorded_version = manifest.get("versions", {}).get("sklearn")
        if recorded_version and recorded_version != sklearn.__version__:
            raise RuntimeError(
                f"Profile 使用 sklearn {recorded_version}，当前为 {sklearn.__version__}"
            )
        self.entries = {}
        for entry in manifest.get("models", []):
            tp = entry.get("filters", {}).get("num_tensor_parallel_workers")
            key = (tp, entry.get("component"))
            if key in self.entries:
                raise ValueError(f"Profile 模型条目重复: {key}")
            self.entries[key] = entry
        self.token_shapes, self.prefill_shapes, self.decode_shapes = self._load_domains(
            self.profile_root.resolve()
        )
        self._models = {}
        self._predictions = {}

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_domains(profile_root):
        token_shapes = set()
        with (profile_root / "mlp_measurement_combinations.csv").open(
                newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                token_shapes.add((int(row["num_tensor_parallel_workers"]), int(row["num_tokens"])))

        prefill_shapes = set()
        decode_shapes = set()
        with (profile_root / "attention_measurement_combinations.csv").open(
                newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                tp = int(row["num_tensor_parallel_workers"])
                batch = int(row["batch_size"])
                chunk = int(row["prefill_chunk_size"])
                context = int(row["kv_cache_size"])
                if chunk == 0:
                    decode_shapes.add((tp, batch, context))
                else:
                    prefill_shapes.add((tp, batch, chunk, context))
        return frozenset(token_shapes), frozenset(prefill_shapes), frozenset(decode_shapes)

    def _load_model(self, tp, component):
        key = (tp, component)
        if key in self._models:
            return self._models[key]
        entry = self.entries.get(key)
        if entry is None:
            raise ValueError(f"不存在 TP={tp} 的 {component} Profile")
        path = self.profile_root / entry["model_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Profile 模型不存在: {path}")
        model = joblib.load(path)
        self._models[key] = model
        return model

    def _predict_component(self, tp, component, features):
        model = self._load_model(tp, component)
        entry = self.entries[(tp, component)]
        ordered = tuple(features[name] for name in entry["features"])
        key = (tp, component, ordered)
        if key in self._predictions:
            return self._predictions[key]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            value = float(model.predict(np.asarray([ordered], dtype=float))[0])
        if not isfinite(value) or value < 0:
            raise RuntimeError(f"{component} 返回非法时延 {value}")
        self._predictions[key] = value
        return value

    def _validate_node_profile(self, task, node_profile, tp):
        if not isinstance(node_profile, dict):
            raise ValueError("节点 Profile 必须是字典")
        node = _nonnegative_int(task.get("node_id"), "task.node_id")
        if node_profile.get("node_id", node) != node:
            raise ValueError("节点 Profile 与任务 node_id 不一致")
        if node_profile.get("device_type", "").lower() != _DEVICE:
            raise ValueError("当前计算模型仅支持 device_type='a100'")
        if node_profile.get("model_id") != _MODEL:
            raise ValueError(f"当前计算模型仅支持 model_id='{_MODEL}'")
        if node_profile.get("dtype") != _DTYPE:
            raise ValueError("当前计算模型仅支持 dtype='float16'")
        if node_profile.get("tensor_parallel_size") != tp:
            raise ValueError("节点 Profile 的 tensor_parallel_size 与任务不一致")
        profile_id = node_profile.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("节点 Profile 必须提供非空 profile_id")

    def predict(self, task, node_profile):
        """预测一个 layer_compute 或 operation_compute 任务的纯 GPU 计算时延。"""
        if not isinstance(task, dict):
            raise ValueError("计算任务必须是字典")
        task_type = task.get("task_type")
        if task_type not in ("layer_compute", "operation_compute"):
            raise ValueError(f"计算模块不接受 task_type={task_type!r}")
        phase = task.get("phase")
        if phase not in ("prefill", "decode"):
            raise ValueError("phase 必须是 prefill 或 decode")
        batch = _positive_int(task.get("batch_size"), "batch_size")
        current = _positive_int(task.get("input_tokens"), "input_tokens")
        context = _nonnegative_int(task.get("context_tokens"), "context_tokens")
        if phase == "decode" and current != 1:
            raise ValueError("Decode Profile 只表示每序列新增一个 Token")

        if task_type == "layer_compute":
            start = _nonnegative_int(task.get("start_layer"), "start_layer")
            end = _positive_int(task.get("end_layer"), "end_layer")
            if not start < end <= 32:
                raise ValueError("层区间必须非空且位于 Llama-2-7B 的 [0, 32) 内")
            tp = 1
            layer_count = end - start
            operation = None
        else:
            layer = _nonnegative_int(task.get("layer_id"), "layer_id")
            if layer >= 32:
                raise ValueError("layer_id 必须位于 Llama-2-7B 的 [0, 32) 内")
            tp = _positive_int(task.get("tp_size"), "tp_size")
            rank = _nonnegative_int(task.get("rank"), "rank")
            if tp not in _SUPPORTED_TP or rank >= tp:
                raise ValueError("TP 仅支持 1、2、4、8，且 rank 必须小于 tp_size")
            operation = task.get("operation")
            if operation not in ("attention", "mlp"):
                raise ValueError("operation 必须是 attention 或 mlp")
            layer_count = 1
        self._validate_node_profile(task, node_profile, tp)

        past = 0 if phase == "prefill" else context
        num_tokens = batch * current
        phase_attention = "attn_prefill" if phase == "prefill" else "attn_decode"
        features = {
            "num_tokens": num_tokens,
            "batch_size": batch,
            "kv_cache_size": past,
            "prefill_chunk_size_squared": current ** 2,
        }

        components = {}
        selected = []
        if operation in (None, "attention"):
            selected.extend(_ATTENTION_COMPONENTS)
            selected.append(phase_attention)
        if operation in (None, "mlp"):
            selected.extend(_MLP_COMPONENTS)
        for component in selected:
            components[component] = self._predict_component(tp, component, features)

        token_measured = (tp, num_tokens) in self.token_shapes
        if phase == "prefill":
            attention_measured = (tp, batch, current, past) in self.prefill_shapes
        else:
            attention_measured = (tp, batch, past) in self.decode_shapes
        in_domain = token_measured and attention_measured
        notices = [
            "组件预测值相加是单层估计，不是校准后的端到端时延。",
            "KV cache save 缺少实测目标，未把原导出模型中的补零值纳入时延。",
        ]
        if phase == "prefill":
            notices.append(
                "策略任务的 context_tokens 表示完整提示词长度；完整 Prefill 按历史 KV=0 查询。"
            )
        if not token_measured:
            notices.append(f"num_tokens={num_tokens} 不在 TP={tp} 的通用组件实测组合中，已允许 RF 域外预测。")
        if not attention_measured:
            shape = ((batch, current, past) if phase == "prefill" else (batch, past))
            notices.append(f"{phase} Attention 形状 {shape} 不在 TP={tp} 的实测组合中，已允许 RF 域外预测。")

        per_layer = sum(components.values())
        return {
            "latency_ms": per_layer * layer_count,
            "per_layer_latency_ms": per_layer,
            "layer_count": layer_count,
            "components_ms": {name: value * layer_count for name, value in components.items()},
            "profile_id": node_profile["profile_id"],
            "model_id": _MODEL,
            "device_type": _DEVICE,
            "tensor_parallel_size": tp,
            "in_measured_domain": in_domain,
            "warnings": notices,
        }


class NodeComputeSimulator:
    """单计算资源、非抢占 FIFO 的节点计算调度器。"""

    def __init__(self, node_profiles, latency_model=None):
        if not isinstance(node_profiles, dict) or not node_profiles:
            raise ValueError("node_profiles 必须是按节点编号索引的非空字典")
        if any(type(node) is not int or node < 0 for node in node_profiles):
            raise ValueError("node_profiles 的键必须是非负整数 node_id")
        self.node_profiles = node_profiles
        self.latency_model = latency_model or A100VidurLatencyModel()
        self.available_time_ms = {node: 0.0 for node in node_profiles}
        self.records = []
        self._records_by_node = {node: [] for node in node_profiles}
        self._task_ids = set()

    def submit(self, task, ready_time_ms):
        """按调用顺序进入节点 FIFO，返回该任务的完整计算时间记录。"""
        if type(ready_time_ms) not in (int, float) or not isfinite(ready_time_ms) or ready_time_ms < 0:
            raise ValueError("ready_time_ms 必须是有限非负数")
        task_id = task.get("task_id") if isinstance(task, dict) else None
        request_id = task.get("request_id") if isinstance(task, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("任务必须提供非空 task_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("任务必须提供非空 request_id")
        if task_id in self._task_ids:
            raise ValueError(f"任务 {task_id} 已提交")
        node = task.get("node_id")
        if node not in self.node_profiles:
            raise ValueError(f"节点 {node} 没有 Profile 映射")

        estimate = self.latency_model.predict(task, self.node_profiles[node])
        ready = float(ready_time_ms)
        start = max(ready, self.available_time_ms[node])
        end = start + estimate["latency_ms"]
        record = {
            "task_id": task_id,
            "request_id": request_id,
            "node_id": node,
            "phase": task["phase"],
            "task_type": task["task_type"],
            "ready_time_ms": ready,
            "start_time_ms": start,
            "queue_delay_ms": start - ready,
            "compute_time_ms": estimate["latency_ms"],
            "end_time_ms": end,
            **{key: value for key, value in estimate.items() if key != "latency_ms"},
        }
        self.available_time_ms[node] = end
        self.records.append(record)
        self._records_by_node[node].append(record)
        self._task_ids.add(task_id)
        return record

    def node_states(self, time_ms):
        """返回给定时刻各节点正在执行及已就绪排队的任务。"""
        if type(time_ms) not in (int, float) or not isfinite(time_ms) or time_ms < 0:
            raise ValueError("time_ms 必须是有限非负数")
        now = float(time_ms)
        states = {}
        for node, available in self.available_time_ms.items():
            node_records = self._records_by_node[node]
            running = next((record["task_id"] for record in node_records
                            if record["start_time_ms"] <= now < record["end_time_ms"]), None)
            queued = [record["task_id"] for record in node_records
                      if record["ready_time_ms"] <= now < record["start_time_ms"]]
            states[node] = {
                "node_id": node,
                "available_time_ms": available,
                "running_task_id": running,
                "queued_task_ids": queued,
            }
        return states
