"""单跳无竞争传输时间模型，所有时间输入和输出均为毫秒。

策略层负责将多跳路径拆成独立的 transfer 任务并建立逐跳依赖。
本模块只计算单个任务的耗时，不维护时钟、拓扑、带宽占用或传输进度。
"""

from math import isfinite


def transfer_time_ms(task: dict, *, bandwidth_bps: float, delay_ms: float) -> float:
    """返回 8 * data_size / bandwidth_bps * 1000 + delay_ms。

    task 沿用策略层 TransferTask 格式，data_size 单位为 Byte。
    调用方提供 src → dst 单跳链路的带宽（bit/s）和传播时延（ms），
    并负责确认链路可用。每个任务独享给定带宽，不计排队或并发竞争；
    本次计算使用固定参数，不处理执行期间的链路变化或通信能耗。
    """
    if not isinstance(task, dict) or task.get("task_type") != "transfer":
        raise ValueError("传输模型只接受 task_type='transfer' 的任务字典")
    for field in ("task_id", "request_id"):
        value = task.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须是非空字符串")
    if task.get("phase") not in ("prefill", "decode"):
        raise ValueError("phase 必须是 prefill 或 decode")
    if task.get("data_type") not in ("kv", "activation", "token"):
        raise ValueError("data_type 必须是 kv、activation 或 token")
    for field in ("decode_step", "src", "dst", "data_size"):
        value = task.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} 必须是非负整数")
    if task["src"] == task["dst"]:
        raise ValueError("单跳传输的 src 和 dst 必须不同")
    if (type(bandwidth_bps) not in (int, float)
            or not isfinite(bandwidth_bps) or bandwidth_bps <= 0):
        raise ValueError("bandwidth_bps 必须是有限正数，单位为 bit/s")
    if (type(delay_ms) not in (int, float)
            or not isfinite(delay_ms) or delay_ms < 0):
        raise ValueError("delay_ms 必须是有限非负数，单位为 ms")

    duration_ms = 8 * task["data_size"] / bandwidth_bps * 1000 + delay_ms
    if not isfinite(duration_ms):
        raise ValueError("传输耗时超出有限数值范围")
    return duration_ms
