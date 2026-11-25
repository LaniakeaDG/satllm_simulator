"""任务图离散事件仿真。文件输入与输出由 run_simulation.py 负责。

时间统一为 ms。同刻先结算发送/执行完成，再更新拓扑和请求到达，最后启动就绪任务。
传播不占带宽；发送完成后的传播不受断链影响。未完成任务保留状态，不伪造结束时间。
"""

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from heapq import heappop, heappush
from math import isfinite
from pathlib import Path
import sys

if __package__:
    from .transmission import transfer_time_ms
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.transmission import transfer_time_ms


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} 必须是有限{'正' if positive else '非负'}数")
    return float(value)


def _file(value, name, base):
    if not isinstance(value, str) or not value:
        raise ValueError(f"请在 inputs.json 指定 {name}")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"{name} 必须指定具体文件，而非目录：{path}")
    return path


def _graph(plans, requests):
    arrivals = {}
    for request in requests:
        rid = request["request_id"]
        if not isinstance(rid, str) or not rid or rid in arrivals:
            raise ValueError("请求 request_id 必须为唯一非空字符串")
        arrivals[rid] = _number(request["arrival_time_ms"], "arrival_time_ms")
    tasks, dependencies = {}, {}
    planned = set()
    for plan in plans:
        rid = plan["request_id"]
        if rid not in arrivals or rid in planned:
            raise ValueError("每个请求必须对应一个策略计划，不能混合备选策略")
        planned.add(rid)
        local = {task["task_id"] for task in plan["tasks"]}
        if not local or set(plan["depends_on"]) != local:
            raise ValueError("任务图不能为空，depends_on 必须覆盖全部任务")
        for task in plan["tasks"]:
            tid = task["task_id"]
            if not isinstance(tid, str) or not tid or tid in tasks:
                raise ValueError("task_id 必须为全局唯一非空字符串")
            if task["request_id"] != rid:
                raise ValueError("任务与计划 request_id 不一致")
            if task["task_type"] not in ("layer_compute", "operation_compute", "transfer"):
                raise ValueError(f"不支持的任务类型：{task['task_type']}")
            preds = plan["depends_on"][tid]
            if not isinstance(preds, list) or len(set(preds)) != len(preds) or not set(preds) <= local:
                raise ValueError(f"任务 {tid} 的前驱无效")
            tasks[tid] = task
            dependencies[tid] = preds
    if planned != set(arrivals):
        raise ValueError("请求 trace 和策略计划的 request_id 集合必须一致")
    successors = defaultdict(list)
    counts = {tid: len(preds) for tid, preds in dependencies.items()}
    for tid, preds in dependencies.items():
        for pred in preds:
            successors[pred].append(tid)
    queue = deque(tid for tid, count in counts.items() if count == 0)
    visited = 0
    while queue:
        visited += 1
        for child in successors[queue.popleft()]:
            counts[child] -= 1
            if counts[child] == 0:
                queue.append(child)
    if visited != len(tasks):
        raise ValueError("任务图包含环")
    return tasks, dependencies, successors, arrivals


def _durations(tasks, profile_root, bandwidth_bps, latency_model):
    """计算和通信分支并行预估；Profile 在计算分支内串行查询，避免缓存竞争。"""
    def compute():
        compute_tasks = [t for t in tasks.values() if t["task_type"] != "transfer"]
        if not compute_tasks:
            return {}
        model = latency_model
        if model is None:
            if __package__:
                from .computation import A100VidurLatencyModel
            else:
                from execution.computation import A100VidurLatencyModel
            model = A100VidurLatencyModel(Path(profile_root))
        estimates = {}
        for task in compute_tasks:
            tp = task.get("tp_size", 1)
            profile = {
                "node_id": task["node_id"], "profile_id": f"a100-llama2-7b-fp16-tp{tp}",
                "device_type": "a100", "model_id": "meta-llama/Llama-2-7b-hf",
                "dtype": "float16", "tensor_parallel_size": tp,
            }
            estimate = model.predict(task, profile)
            estimates[task["task_id"]] = {
                **estimate, "latency_ms": _number(estimate["latency_ms"], "latency_ms"),
            }
        return estimates

    def transfer():
        # 传播时延只能在实际发送时确定；先预计算独享带宽下的纯发送时间。
        return {tid: transfer_time_ms(task, bandwidth_bps=bandwidth_bps, delay_ms=0)
                for tid, task in tasks.items() if task["task_type"] == "transfer"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        computation = pool.submit(compute)
        communication = pool.submit(transfer)
        return computation.result(), communication.result()


def simulate(plans, requests, link_events, *, profile_root, bandwidth_bps=200_000_000,
             end_time_ms, latency_model=None):
    """返回任务时间线。link_events 为含 time_s/sat_i/sat_j/delay_ms 的行序列。

    end_time_ms 是有效观测边界；边界后不推断链路持续存在。
    latency_model 可注入兼容 predict(task, node_profile) 的计算预测器。
    """
    bandwidth = _number(bandwidth_bps, "bandwidth_bps", positive=True)
    horizon = _number(end_time_ms, "end_time_ms")
    tasks, dependencies, successors, arrivals = _graph(plans, requests)
    estimates, send_times = _durations(tasks, profile_root, bandwidth, latency_model)
    records = {tid: {
        "task_id": tid, "request_id": task["request_id"], "task_type": task["task_type"],
        "status": "pending", "ready_time_ms": None, "start_time_ms": None,
        "end_time_ms": None, "queue_delay_ms": None,
        **({"node_id": task["node_id"], "compute_time_ms": estimates[tid]["latency_ms"],
            "profile": estimates[tid]} if tid in estimates else {
            "src": task["src"], "dst": task["dst"], "data_size": task["data_size"],
            "uncontended_send_time_ms": send_times[tid], "baseline_transfer_time_ms": None,
            "attempts": [], "retransmissions": 0,
        }),
    } for tid, task in tasks.items()}
    events, serial = [], 0

    def push(time, kind, payload):
        nonlocal serial
        serial += 1
        heappush(events, (time, serial, kind, payload))

    relevant = {tuple(sorted((t["src"], t["dst"])))
                for t in tasks.values() if t["task_type"] == "transfer"}
    for event in link_events:
        i, j = event["sat_i"], event["sat_j"]
        if type(i) is not int or type(j) is not int or min(i, j) < 0 or i == j:
            raise ValueError("链路端点必须是不同的非负整数")
        time = _number(event["time_s"], "time_s") * 1000
        delay = event["delay_ms"]
        if delay is not None:
            delay = _number(delay, "delay_ms")
        pair = tuple(sorted((i, j)))
        if pair in relevant and time <= horizon:
            push(time, "link", (pair, delay))
    for rid, time in arrivals.items():
        push(time, "arrival", rid)

    roots = defaultdict(list)
    remaining = {tid: len(preds) for tid, preds in dependencies.items()}
    for tid, count in remaining.items():
        if count == 0:
            roots[tasks[tid]["request_id"]].append(tid)
    links, arrived, ready = {}, set(), deque()
    queues = defaultdict(deque)
    busy = {}
    waiting = set()
    sending = {}  # task_id -> 剩余 bit；方向由任务 src/dst 标识。
    rank = {tid: index for index, tid in enumerate(tasks)}
    now = 0.0

    def finish(tid):
        record = records[tid]
        record.update(status="completed", end_time_ms=now)
        for child in successors[tid]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)

    def start_senders():
        for tid in sorted(waiting, key=rank.get):
            task = tasks[tid]
            pair = tuple(sorted((task["src"], task["dst"])))
            if pair not in links:
                continue
            record = records[tid]
            delay = links[pair]
            if record["start_time_ms"] is None:
                record["start_time_ms"] = now
                record["queue_delay_ms"] = now - record["ready_time_ms"]
            record["baseline_transfer_time_ms"] = send_times[tid] + delay
            record["attempts"].append({"start_time_ms": now, "send_end_time_ms": None,
                                       "delay_ms": delay, "status": "sending"})
            record["status"] = "sending"
            sending[tid] = 8 * task["data_size"]
            waiting.remove(tid)

    while True:
        while ready:
            tid = ready.popleft()
            record, task = records[tid], tasks[tid]
            if task["request_id"] not in arrived:
                continue
            record.update(ready_time_ms=now, status="queued")
            if task["task_type"] == "transfer":
                waiting.add(tid)
                record["status"] = "waiting_link"
            else:
                queues[task["node_id"]].append(tid)
        for node, queue in queues.items():
            if node not in busy and queue:
                tid = queue.popleft()
                busy[node] = tid
                records[tid].update(status="computing", start_time_ms=now,
                                    queue_delay_ms=now - records[tid]["ready_time_ms"])
                push(now + estimates[tid]["latency_ms"], "compute_done", tid)
        start_senders()
        directions = defaultdict(list)
        for tid in sending:
            task = tasks[tid]
            directions[(task["src"], task["dst"])].append(tid)
        # 同方向 n 个发送流均分 B：每流速率 B/(1000*n) bit/ms，全双工分别计数。
        rates = {tid: bandwidth / (1000 * len(group))
                 for group in directions.values() for tid in group}
        next_send = min((now + bits / rates[tid] for tid, bits in sending.items()), default=float("inf"))
        next_event = events[0][0] if events else float("inf")
        future = min(next_send, next_event)
        target = min(future, horizon)
        completed_sends = [tid for tid, bits in sending.items()
                           if now + bits / rates[tid] <= target]
        for tid in sending:
            sending[tid] = max(0.0, sending[tid] - rates[tid] * (target - now))
        now = target
        for tid in completed_sends:
            del sending[tid]
            record = records[tid]
            attempt = record["attempts"][-1]
            attempt.update(send_end_time_ms=now, status="sent")
            record["status"] = "propagating"
            push(now + attempt["delay_ms"], "transfer_done", tid)
        batch = []
        while events and events[0][0] <= now:
            batch.append(heappop(events))
        # 完成优先于同刻断链；链路变动完成后才释放新的任务。
        for _, _, kind, payload in batch:
            if kind == "compute_done":
                del busy[tasks[payload]["node_id"]]
                finish(payload)
            elif kind == "transfer_done":
                finish(payload)
        for _, _, kind, payload in batch:
            if kind == "arrival":
                arrived.add(payload)
                ready.extend(roots[payload])
            elif kind == "link":
                pair, delay = payload
                if delay is not None:
                    links[pair] = delay
                    continue
                links.pop(pair, None)
                for tid in list(sending):
                    task = tasks[tid]
                    if tuple(sorted((task["src"], task["dst"]))) != pair:
                        continue
                    del sending[tid]
                    record = records[tid]
                    record["attempts"][-1].update(status="interrupted", send_end_time_ms=now)
                    record["retransmissions"] += 1
                    record["status"] = "waiting_link"
                    waiting.add(tid)
        if now == horizon:
            # 结算边界上的完成事件，不再启动新任务。
            break
        if all(r["status"] == "completed" for r in records.values()):
            break
        if future == float("inf"):
            break
    for tid, record in records.items():
        start, end = record["start_time_ms"], record["end_time_ms"]
        record["execution_elapsed_ms"] = None if end is None else end - start
        if tasks[tid]["task_type"] == "transfer":
            record["remaining_bits"] = (
                sending[tid] if tid in sending else
                8 * tasks[tid]["data_size"] if record["status"] in ("pending", "waiting_link") else 0
            )
    return {
        "time_unit": "ms", "end_time_ms": now, "observation_end_time_ms": horizon,
        "bandwidth_bps": bandwidth,
        "completed": all(r["status"] == "completed" for r in records.values()),
        "tasks": list(records.values()),
    }

