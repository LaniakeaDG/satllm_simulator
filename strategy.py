"""卫星 LLM Serving 的任务计划。"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction
from functools import cache, lru_cache
from heapq import heappop, heappush
from math import isfinite, prod
from pathlib import Path
from typing import Literal, TypedDict, get_args


_FLOAT_ELEMENT_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
# offline.profile.StreamingModel.next_token 的 argmax 返回一个 torch.int64 索引。
_TOKEN_ID_BYTES = 8


class _RequestSource(TypedDict, total=False):
    source_sat_id: int


class Request(_RequestSource):
    request_id: str
    arrival_time: float
    input_length: int
    output_length: int
    service_requirements: object


Phase = Literal["prefill", "decode"]


class _ComputeFields(TypedDict):
    task_id: str
    request_id: str
    node_id: int
    phase: Phase
    batch_size: int
    input_tokens: int
    context_tokens: int


class LayerComputeTask(_ComputeFields):
    task_type: Literal["layer_compute"]
    start_layer: int
    end_layer: int


class OperationComputeTask(_ComputeFields):
    task_type: Literal["operation_compute"]
    layer_id: int
    operation: Literal["attention", "mlp"]
    tp_size: int
    rank: int


class TransferTask(TypedDict):
    task_id: str
    request_id: str
    task_type: Literal["transfer"]
    phase: Phase
    decode_step: int
    data_type: Literal["kv", "activation", "token"]
    src: int
    dst: int
    data_size: int


class CollectiveTask(TypedDict):
    """participants 按 Rank 排列。data_size 为单个逻辑张量的字节数。"""

    task_id: str
    request_id: str
    task_type: Literal["collective"]
    phase: Phase
    decode_step: int
    collective_type: Literal["all_reduce", "broadcast"]
    participants: list[int]
    root: int | None
    data_size: int
    layer_id: int | None


Strategy = Literal["local", "pd", "pp", "tp", "hybrid"]
ComputeTask = LayerComputeTask | OperationComputeTask
Task = ComputeTask | TransferTask | CollectiveTask


class Plan(TypedDict):
    request_id: str
    strategy: Strategy
    tasks: list[Task]
    depends_on: dict[str, list[str]]


def assemble_plan(strategy: Strategy, request: Request, *parts) -> Plan:
    """组装完整的 Local/PD 结果，或依次组装 Prefill 和 Decode 片段。"""
    if strategy not in get_args(Strategy):
        raise ValueError(f"未知策略 {strategy!r}")
    single_part = strategy in ("local", "pd")
    if len(parts) != (1 if single_part else 2):
        raise ValueError(f"{strategy} 需要{'一个完整规划' if single_part else '两个规划片段'}")
    if not parts[0]["tasks"]:
        raise ValueError("首个规划片段必须包含 Prefill")

    request_id = request["request_id"]
    prefix = f"{request_id}:{strategy}:" if single_part else ""
    tasks: list[Task] = []
    positions: dict[str, int] = {}
    depends_on: dict[str, list[str]] = {}
    for part in parts:
        if part["request_id"] != request_id:
            raise ValueError("规划片段的 request_id 与请求不一致")
        for task in part["tasks"]:
            if task["request_id"] != request_id:
                raise ValueError("任务的 request_id 与请求不一致")
            task_id = prefix + task["task_id"]
            if task_id in positions:
                raise ValueError(f"重复任务 ID {task_id}")
            positions[task_id] = len(tasks)
            item = deepcopy(task)
            item["task_id"] = task_id
            tasks.append(item)
        for key, predecessors in part["depends_on"].items():
            task_id = prefix + key
            if task_id in depends_on:
                raise ValueError(f"重复依赖键 {task_id}")
            depends_on[task_id] = [prefix + pred for pred in predecessors]

    if depends_on.keys() != positions.keys():
        raise ValueError("depends_on 的键集合与任务 ID 集合不一致")
    for task_id, predecessors in depends_on.items():
        for pred in predecessors:
            if pred not in positions:
                raise ValueError(f"{task_id} 的前驱 {pred} 不存在")
            if positions[pred] >= positions[task_id]:
                raise ValueError(f"{pred} 必须排在 {task_id} 之前")
    return Plan(request_id=request_id, strategy=strategy, tasks=tasks, depends_on=depends_on)


def plan_pd_request(request, prefill_node, decode_node, model_info=None):
    """在不同节点安排 Prefill 和 Decode，通过 KV 交接衔接两个阶段。"""
    return _plan_request(request, prefill_node, decode_node, model_info, pd_separated=True)


def _pd_sat_id(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} 必须是非负整数卫星编号")
    return value


def _pd_nonnegative(value, field):
    if (type(value) not in (int, float) or value < 0
            or (type(value) is float and not isfinite(value))):
        raise ValueError(f"{field} 必须是有限非负数")


@lru_cache(maxsize=1)
def _pd_trace_events(path, mtime_ns, size):
    """按路径、修改时间和文件大小缓存最近读取的一份已校验事件。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = ["time_s", "sat_i", "sat_j", "delay_ms"]
    with pq.ParquetFile(path) as trace:
        schema = trace.schema_arrow
        for name in fields:
            if schema.get_field_index(name) < 0:
                raise ValueError(f"链路事件字段缺失或重名: {name}")
            dtype = schema.field(name).type
            if pa.types.is_integer(dtype):
                continue
            if name in ("time_s", "delay_ms") and pa.types.is_floating(dtype):
                continue
            if name == "delay_ms" and pa.types.is_null(dtype):
                continue
            raise ValueError(f"链路事件字段类型错误: {name} ({dtype})")
        events = trace.read(columns=fields)

    ordered = []
    for row in events.sort_by([("time_s", "ascending")]).to_pylist():
        time_s, delay = row["time_s"], row["delay_ms"]
        _pd_nonnegative(time_s, "time_s")
        i = _pd_sat_id(row["sat_i"], "sat_i")
        j = _pd_sat_id(row["sat_j"], "sat_j")
        if i == j:
            raise ValueError("链路端点不能相同")
        if delay is not None:
            _pd_nonnegative(delay, "delay_ms")
        pair = (min(i, j), max(i, j))
        ordered.append((time_s, pair, delay))
    return tuple(ordered)


def _pd_trace_snapshot(path, decision_time_s):
    """独立回放至指定时刻。现有 trace 只在建链时记录传播时延。"""
    path = Path(path).resolve()
    stat = path.stat()
    active_links = {}
    for time_s, pair, delay in _pd_trace_events(path, stat.st_mtime_ns, stat.st_size):
        if time_s > decision_time_s:
            break
        if delay is None:
            active_links.pop(pair, None)
        else:
            active_links[pair] = delay
    return active_links


def _pd_snapshot_items(active_links):
    if not isinstance(active_links, dict):
        raise ValueError("active_links 必须是链路快照字典")
    for pair, delay in active_links.items():
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(type(node) is not int for node in pair) or not 0 <= pair[0] < pair[1]):
            raise ValueError("active_links 端点必须是严格升序的非负整数对")
        _pd_nonnegative(delay, "active_links 传播时延")
        yield pair, delay


def plan_pd_request_at(request, prefill_candidates, decode_candidates, *,
                       decision_time_s: float, model_info=None, link_events_path=None,
                       topology_provider: Callable[[float], dict[tuple[int, int], float]] | None = None):
    """用指定仿真秒数的 Provider 或 Parquet 快照，从请求来源选点并规划。"""
    _pd_nonnegative(decision_time_s, "decision_time_s")
    if (link_events_path is None) == (topology_provider is None):
        raise ValueError("必须且只能提供 link_events_path 和 topology_provider 中的一项")
    if topology_provider is None:
        active_links = _pd_trace_snapshot(link_events_path, decision_time_s)
    else:
        active_links = dict(_pd_snapshot_items(topology_provider(decision_time_s)))
    source = _pd_sat_id(request.get("source_sat_id"), "source_sat_id")
    nodes = select_pd_nodes(prefill_candidates, decode_candidates, active_links, source_sat_id=source)
    return None if nodes is None else plan_pd_request(request, *nodes, model_info=model_info)


def plan_local_request(request, node_id, model_info=None):
    """将单请求的 Prefill 和所有 Decode step 安排在同一节点。"""
    return _plan_request(request, node_id, node_id, model_info, pd_separated=False)


def _request_lengths(request):
    """校验统一 Request 或原有的 input_ids/output_tokens 请求字典。"""
    if not isinstance(request, dict):
        raise ValueError("request 必须是字典")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id 必须是非空字符串")
    if "input_ids" in request or "output_tokens" in request:
        input_ids = request.get("input_ids")
        if not isinstance(input_ids, (list, tuple)) or not input_ids:
            raise ValueError("input_ids 必须是非空 token 序列")
        if any(type(token) is not int or token < 0 for token in input_ids):
            raise ValueError("input_ids 中的 token 编号必须是非负整数")
        input_length = len(input_ids)
        output_tokens = request.get("output_tokens")
    else:
        if not Request.__required_keys__.issubset(request):
            raise ValueError("统一 Request 缺少必需字段")
        if type(request["arrival_time"]) not in (int, float):
            raise ValueError("arrival_time 必须是数值")
        input_length = request["input_length"]
        if type(input_length) is not int or input_length < 1:
            raise ValueError("input_length 必须是正整数")
        output_tokens = request["output_length"]
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError("输出长度必须是包含首 token 的正整数")
    return request_id, input_length, output_tokens


def _plan_request(request, prefill_node, decode_node, model_info, *, pd_separated):
    """全模型层区间为 [0, layers)，Decode 的 context_tokens 为追加 KV 前的长度。"""
    request_id, input_length, output_tokens = _request_lengths(request)
    if any(type(node) is not int or node < 0 for node in (prefill_node, decode_node)):
        raise ValueError("卫星节点编号必须是非负整数")
    if pd_separated and prefill_node == decode_node:
        raise ValueError("Prefill 和 Decode 节点必须不同")

    layers = model_info.get("num_hidden_layers") if isinstance(model_info, dict) else None
    if type(layers) is not int or layers <= 0:
        raise ValueError("model_info 必须提供正整数 num_hidden_layers")
    tasks: list[LayerComputeTask | TransferTask] = [LayerComputeTask(
        task_id="prefill", request_id=request_id, task_type="layer_compute", phase="prefill",
        node_id=prefill_node, batch_size=1, input_tokens=input_length,
        context_tokens=input_length, start_layer=0, end_layer=layers,
    )]
    depends_on = {"prefill": []}
    if pd_separated and "source_sat_id" in request:
        source = _pd_sat_id(request["source_sat_id"], "source_sat_id")
        if source != prefill_node:
            tasks.insert(0, TransferTask(
                task_id="request_transfer", request_id=request_id, task_type="transfer",
                phase="prefill", decode_step=0, data_type="token",
                src=source, dst=prefill_node, data_size=input_length * _TOKEN_ID_BYTES,
            ))
            depends_on["request_transfer"] = []
            depends_on["prefill"] = ["request_transfer"]
    predecessor = "prefill"
    if pd_separated and output_tokens > 1:
        # Decode 端同时接收提示词的 KV 和 Prefill 采样的首个 token。
        handoff_bytes = _kv_bytes(model_info, layers, input_length) + _TOKEN_ID_BYTES
        tasks.append(TransferTask(
            task_id="kv_transfer", request_id=request_id, task_type="transfer",
            phase="prefill", decode_step=0, data_type="kv",
            src=prefill_node, dst=decode_node, data_size=handoff_bytes,
        ))
        depends_on["kv_transfer"] = ["prefill"]
        predecessor = "kv_transfer"
    for step in range(1, output_tokens):
        task_id = f"decode_step_{step}"
        tasks.append(LayerComputeTask(
            task_id=task_id, request_id=request_id, task_type="layer_compute", phase="decode",
            node_id=decode_node, batch_size=1, input_tokens=1,
            context_tokens=input_length + step - 1, start_layer=0, end_layer=layers,
        ))
        depends_on[task_id] = [predecessor]
        predecessor = task_id
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}


def _pd_distances(graph, start):
    """Dijkstra 单源最短传播时延（ms）。"""
    distances = {start: 0}
    queue = [(0, start)]
    while queue:
        cost, node = heappop(queue)
        if cost != distances[node]:
            continue
        for neighbor, delay in graph.get(node, ()):
            total = cost + delay
            if neighbor not in distances or total < distances[neighbor]:
                distances[neighbor] = total
                heappush(queue, (total, neighbor))
    return distances


def select_pd_nodes(prefill_candidates, decode_candidates, active_links, source_sat_id=None):
    """有来源时最小化两段最短路之和；省略来源时沿用原直连规则。"""
    if source_sat_id is not None:
        source = _pd_sat_id(source_sat_id, "source_sat_id")
        graph = {}
        for (i, j), delay in _pd_snapshot_items(active_links):
            graph.setdefault(i, []).append((j, delay))
            graph.setdefault(j, []).append((i, delay))
        prefill_ids = {_pd_sat_id(node, "prefill_candidate") for node in prefill_candidates}
        decode_ids = {_pd_sat_id(node, "decode_candidate") for node in decode_candidates}
        from_source = _pd_distances(graph, source)
        best = None
        for prefill in prefill_ids.intersection(from_source):
            distances = from_source if prefill == source else _pd_distances(graph, prefill)
            for decode in decode_ids:
                if decode == prefill or decode not in distances:
                    continue
                candidate = (from_source[prefill] + distances[decode], prefill, decode)
                if best is None or candidate < best:
                    best = candidate
        return None if best is None else best[1:]

    best = None
    for prefill_node in prefill_candidates:
        for decode_node in decode_candidates:
            if prefill_node == decode_node:
                continue
            link = tuple(sorted((prefill_node, decode_node)))
            if link not in active_links:
                continue
            # 先比较传播时延（ms），再按 Prefill、Decode 编号打破平局。
            candidate = (active_links[link], prefill_node, decode_node)
            if best is None or candidate < best:
                best = candidate

    return None if best is None else best[1:]


def _kv_bytes(model_info, layers, tokens):
    """batch_size=1 时指定层数和已缓存 token 数所占的紧凑 K、V 字节数。"""
    for field in ("num_key_value_heads", "head_dim"):
        value = model_info.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} 必须是正整数")
    dtype = model_info.get("kv_cache_dtype")
    if not isinstance(dtype, str) or dtype not in _FLOAT_ELEMENT_BYTES:
        raise ValueError("kv_cache_dtype 必须是 float16、bfloat16 或 float32")
    return (2 * layers * tokens * model_info["num_key_value_heads"]
            * model_info["head_dim"] * _FLOAT_ELEMENT_BYTES[dtype])


def pp_stage_kv_bytes(model_info, request, stage) -> int:
    """计算 [start_layer, end_layer) 在 batch_size=1 下的紧凑 KV 峰值字节数。"""
    _, input_length, output_length = _request_lengths(request)
    if not isinstance(model_info, dict):
        raise ValueError("model_info 必须是字典")
    layers = model_info.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0:
        raise ValueError("num_hidden_layers 必须是正整数")
    if not isinstance(stage, dict):
        raise ValueError("Stage 必须是字典")
    start, end = stage.get("start_layer"), stage.get("end_layer")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= model_info["num_hidden_layers"]:
        raise ValueError("Stage 必须是模型范围内的非空左闭右开层区间")

    capacity = input_length + output_length - 1
    return _kv_bytes(model_info, end - start, capacity)


def pp_stage_fits_memory(model_info, request, stage, node) -> bool:
    """检查权重独立预算和权重加单请求 KV 的合计预算，其他占用由调用方预留。"""
    kv_bytes = pp_stage_kv_bytes(model_info, request, stage)
    weights = model_info.get("layer_weight_bytes")
    if (not isinstance(weights, (list, tuple)) or len(weights) != model_info["num_hidden_layers"]
            or any(type(w) is not int or w <= 0 for w in weights)):
        raise ValueError("layer_weight_bytes 必须为每层提供一个正整数字节数")
    if not isinstance(node, dict):
        raise ValueError("候选节点必须是字典")
    for field in ("weight_memory_budget_bytes", "total_memory_budget_bytes"):
        budget = node.get(field)
        if type(budget) is not int or budget < 0:
            raise ValueError(f"{field} 必须为非负整数字节数")

    weight_bytes = sum(weights[stage["start_layer"]:stage["end_layer"]])
    return (weight_bytes <= node["weight_memory_budget_bytes"]
            and weight_bytes + kv_bytes <= node["total_memory_budget_bytes"])


def _pp_inputs(model_info, candidate_nodes):
    """校验同一个 case 的逐层数据，按卫星编号固定搜索顺序。"""
    if not isinstance(model_info, dict):
        raise ValueError("model_info 必须是字典")
    count = model_info.get("num_hidden_layers")
    weights = model_info.get("layer_weight_bytes")
    if type(count) is not int or count <= 0:
        raise ValueError("num_hidden_layers 必须是正整数")
    if (not isinstance(weights, (list, tuple)) or len(weights) != count
            or any(type(w) is not int or w <= 0 for w in weights)):
        raise ValueError("layer_weight_bytes 必须为每层提供一个正整数字节数")
    case = model_info.get("profile_case")
    if not isinstance(case, str) or not case.strip():
        raise ValueError("profile_case 必须标识调用方选定的测量条件")
    profiles = model_info.get("layer_latency_ms_by_device")
    if not isinstance(profiles, dict):
        raise ValueError("缺少设备到逐层时延的映射 layer_latency_ms_by_device")
    prefixes = {}
    for device, values in profiles.items():
        if not isinstance(device, str) or not device.strip():
            raise ValueError("Profile 设备类型必须是非空字符串")
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(f"{device} 必须为每层提供一个 latency_ms")
        prefix = [0]
        for value in values:
            if (type(value) not in (float, int) or value < 0
                    or (type(value) is float and not isfinite(value))):
                raise ValueError("latency_ms 必须是有限非负数")
            # 按输入的十进制毫秒精确比较，避免前缀相减丢失小层代价。
            prefix.append(prefix[-1] + Fraction(str(value)))
        prefixes[device] = prefix

    if not isinstance(candidate_nodes, (list, tuple)):
        raise ValueError("candidate_nodes 必须是节点序列")
    nodes = []
    seen = set()
    for node in candidate_nodes:
        if not isinstance(node, dict):
            raise ValueError("候选节点必须是字典")
        sat = node.get("sat_id")
        budget = node.get("weight_memory_budget_bytes")
        device = node.get("device_type")
        if type(sat) is not int or sat < 0 or sat in seen:
            raise ValueError("sat_id 必须为唯一的非负整数")
        if type(budget) is not int or budget < 0:
            raise ValueError("weight_memory_budget_bytes 必须为非负整数")
        if not isinstance(device, str) or device not in prefixes:
            raise ValueError(f"卫星 {sat} 缺少该 profile_case 的 device_type 映射")
        seen.add(sat)
        nodes.append((sat, budget, device))
    return weights, sorted(nodes), prefixes


def build_pp_stages(model_info, candidate_nodes):
    """最小化同一 Profile case 下最大 Stage 的逐层 latency_ms 之和。

    model_info 提供 num_hidden_layers、layer_weight_bytes、profile_case 和
    layer_latency_ms_by_device。节点提供 sat_id、weight_memory_budget_bytes、
    device_type。输出按层顺序排列，每颗入选卫星承接一个非空连续区间。
    输入错误抛出 ValueError，输入有效但无可行部署时抛出 RuntimeError。
    """
    weights, nodes, costs = _pp_inputs(model_info, candidate_nodes)
    if not nodes:
        raise RuntimeError("没有候选卫星")
    max_budget = max(budget for _, budget, _ in nodes)
    for layer, weight in enumerate(weights):
        if weight > max_budget:
            raise RuntimeError(f"单层超出所有节点预算：layer {layer}")
    if sum(weights) > sum(budget for _, budget, _ in nodes):
        raise RuntimeError("模型层权重总量超过所有候选节点的总预算")
    min_weight = min(weights)
    nodes = [node for node in nodes if node[1] >= min_weight]
    weight_prefix = [0]
    for weight in weights:
        weight_prefix.append(weight_prefix[-1] + weight)

    # states[end][mask] 保存覆盖 [0, end) 的最小瓶颈和最后一段的回溯信息。
    count = len(weights)
    states = [{} for _ in range(count + 1)]
    states[0][0] = (0, 0, -1)
    for start in range(count):
        for mask, (bottleneck, _, _) in states[start].items():
            for index, (_, budget, device) in enumerate(nodes):
                bit = 1 << index
                if mask & bit:
                    continue
                next_mask = mask | bit
                prefix = costs[device]
                for end in range(start + 1, count + 1):
                    if weight_prefix[end] - weight_prefix[start] > budget:
                        break
                    score = max(bottleneck, prefix[end] - prefix[start])
                    previous = states[end].get(next_mask)
                    if previous is None or score < previous[0]:
                        states[end][next_mask] = (score, start, index)

    if not states[count]:
        raise RuntimeError("内存约束下无法形成完整连续划分")
    mask = min(states[count], key=lambda m: (states[count][m][0], m.bit_count(), m))
    stages = []
    end = count
    while end:
        _, start, index = states[end][mask]
        stages.append({"node_id": nodes[index][0], "start_layer": start, "end_layer": end})
        mask ^= 1 << index
        end = start
    stages.reverse()
    budgets = {sat: budget for sat, budget, _ in nodes}
    remaining = set(budgets)
    covered = 0
    for stage_id, stage in enumerate(stages):
        stage["stage_id"] = stage_id
        sat = stage["node_id"]
        start, end = stage["start_layer"], stage["end_layer"]
        if (sat not in remaining or start != covered or not start < end <= count
                or weight_prefix[end] - weight_prefix[start] > budgets[sat]):
            raise AssertionError("PP 划分回溯结果无效")
        remaining.remove(sat)
        covered = end
    if covered != count:
        raise AssertionError("PP 划分未覆盖全部模型层")
    return stages


def _pp_direct_edges(active_links):
    if not isinstance(active_links, dict):
        raise ValueError("active_links 必须显式提供无向链路快照字典")
    edges = set()
    for pair, delay in active_links.items():
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(type(sat) is not int or sat < 0 for sat in pair)
                or pair[0] == pair[1]):
            raise ValueError("链路端点必须是两个不同的非负整数")
        if (type(delay) not in (int, float) or delay < 0
                or (type(delay) is float and not isfinite(delay))):
            raise ValueError("传播时延必须是有限非负数")
        edges.add((min(pair), max(pair)))
    return edges


def build_pp_stages_for_request(model_info, candidate_nodes, *, request, active_links=None):
    """在单请求内存预算和指定快照的直连约束下，最小化最大 Stage 计算时延。

    缺失快照属于输入错误；空快照只允许单 Stage。返回与 build_pp_stages 相同的层划分。
    """
    edges = _pp_direct_edges(active_links)
    weights, nodes, costs = _pp_inputs(model_info, candidate_nodes)
    _, _, output_length = _request_lengths(request)
    if not nodes:
        pp_stage_kv_bytes(model_info, request, {"start_layer": 0, "end_layer": 1})
        raise RuntimeError("没有候选卫星")

    count = len(weights)
    resources = {node["sat_id"]: node for node in candidate_nodes}

    @cache
    def last_feasible_end(start, sat):
        # 固定节点和起点时，权重与 KV 均随终点单调增加。
        low, high = start, count
        while low < high:
            end = (low + high + 1) // 2
            stage = {"start_layer": start, "end_layer": end}
            if pp_stage_fits_memory(model_info, request, stage, resources[sat]):
                low = end
            else:
                high = end - 1
        return low

    # states[end][mask, first, last] 保存瓶颈、最后一段起点及前驱状态。
    states = [{} for _ in range(count + 1)]
    states[0][(0, -1, -1)] = (0, 0, None)
    for start in range(count):
        for key, (bottleneck, _, _) in states[start].items():
            mask, first, last = key
            for index, (sat, _, device) in enumerate(nodes):
                bit = 1 << index
                if mask & bit or (mask and (min(last, sat), max(last, sat)) not in edges):
                    continue
                end_limit = last_feasible_end(start, sat)
                if mask and output_length > 1 and (min(sat, first), max(sat, first)) not in edges:
                    # 缺返回边时仍可作为中间 Stage。
                    end_limit = min(end_limit, count - 1)
                next_key = (mask | bit, first if mask else sat, sat)
                prefix = costs[device]
                for end in range(start + 1, end_limit + 1):
                    score = max(bottleneck, prefix[end] - prefix[start])
                    previous = states[end].get(next_key)
                    if previous is None or score < previous[0]:
                        states[end][next_key] = (score, start, key)

    if not states[count]:
        raise RuntimeError("内存和直连链路约束下无法形成完整连续划分")
    key = min(states[count], key=lambda k: (states[count][k][0], k[0].bit_count(), k))
    stages = []
    end = count
    while end:
        _, start, predecessor = states[end][key]
        stages.append({"stage_id": key[0].bit_count() - 1, "node_id": key[2],
                       "start_layer": start, "end_layer": end})
        end, key = start, predecessor
    stages.reverse()
    return stages


def _validate_pp_stages(stages):
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("stages 必须是非空的 Stage 序列")

    used_nodes = set()
    covered = 0
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError("Stage 必须是字典")
        stage_id = stage.get("stage_id")
        node = stage.get("node_id")
        start, end = stage.get("start_layer"), stage.get("end_layer")
        if type(stage_id) is not int or stage_id != index:
            raise ValueError("stage_id 必须从 0 开始按顺序递增")
        if type(node) is not int or node < 0 or node in used_nodes:
            raise ValueError("Stage 的 node_id 必须是唯一的非负整数")
        if type(start) is not int or type(end) is not int or start != covered or end <= start:
            raise ValueError("Stage 必须从第 0 层开始按顺序覆盖连续非空层区间")
        used_nodes.add(node)
        covered = end


def _pp_activation_bytes(model_info, num_tokens):
    """单请求 batch_size=1，传输形状为 [1, num_tokens, hidden_size]。"""
    if not isinstance(model_info, dict):
        raise ValueError("Activation Transfer 需要 model_info 提供 hidden_size 和 activation_dtype")
    hidden_size = model_info.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size 必须是正整数")
    dtype = model_info.get("activation_dtype")
    if not isinstance(dtype, str) or dtype not in _FLOAT_ELEMENT_BYTES:
        raise ValueError("activation_dtype 必须是 float16、bfloat16 或 float32")
    return num_tokens * hidden_size * _FLOAT_ELEMENT_BYTES[dtype]


def plan_pp_prefill(request, stages, model_info=None):
    """按 build_pp_stages 的完整划分生成单请求的 Prefill 任务链。

    层区间为 [start_layer, end_layer)，最终边界由划分器对照模型校验。
    depends_on 记录直接前驱，末个计算任务完成即代表 Prefill 完成。
    """
    request_id, input_length, _ = _request_lengths(request)
    _validate_pp_stages(stages)
    activation_bytes = _pp_activation_bytes(model_info, input_length) if len(stages) > 1 else 0
    tasks: list[LayerComputeTask | TransferTask] = []
    depends_on: dict[str, list[str]] = {}
    for index, stage in enumerate(stages):
        node = stage["node_id"]
        prefill_id = f"{request_id}:prefill_{index}"
        predecessors = []
        if index > 0:
            transfer_id = f"{request_id}:activation_transfer_{index - 1}_{index}"
            depends_on[transfer_id] = [tasks[-1]["task_id"]]
            tasks.append(TransferTask(
                task_id=transfer_id, request_id=request_id, task_type="transfer",
                phase="prefill", decode_step=0, data_type="activation",
                src=stages[index - 1]["node_id"], dst=node, data_size=activation_bytes,
            ))
            predecessors = [transfer_id]
        depends_on[prefill_id] = predecessors
        tasks.append(LayerComputeTask(
            task_id=prefill_id, request_id=request_id, task_type="layer_compute", phase="prefill",
            node_id=node, batch_size=1, input_tokens=input_length,
            context_tokens=input_length,
            start_layer=stage["start_layer"], end_layer=stage["end_layer"],
        ))
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}


def plan_pp_decode(request, stages, model_info=None):
    """生成单请求逐轮经过全部 Stage 的 Decode 计划，直接接续 PP Prefill 末任务。"""
    request_id, input_length, output_length = _request_lengths(request)
    _validate_pp_stages(stages)
    activation_bytes = 0
    if len(stages) > 1 and output_length > 1:
        activation_bytes = _pp_activation_bytes(model_info, 1)
    tasks: list[LayerComputeTask | TransferTask] = []
    for step in range(1, output_length):
        prefix = f"{request_id}:decode_step_{step}"
        if len(stages) > 1:
            tasks.append(TransferTask(
                task_id=f"{prefix}_token_transfer", request_id=request_id,
                task_type="transfer",
                phase="decode", decode_step=step, data_type="token", src=stages[-1]["node_id"],
                dst=stages[0]["node_id"], data_size=_TOKEN_ID_BYTES,
            ))
        for index, stage in enumerate(stages):
            if index > 0:
                tasks.append(TransferTask(
                    task_id=f"{prefix}_activation_transfer_{index - 1}_{index}",
                    request_id=request_id, task_type="transfer",
                    phase="decode", decode_step=step, data_type="activation",
                    src=stages[index - 1]["node_id"], dst=stage["node_id"], data_size=activation_bytes,
                ))
            tasks.append(LayerComputeTask(
                task_id=f"{prefix}_stage_{index}", request_id=request_id, task_type="layer_compute", phase="decode",
                node_id=stage["node_id"], batch_size=1, input_tokens=1,
                context_tokens=input_length + step - 1,
                start_layer=stage["start_layer"], end_layer=stage["end_layer"],
            ))

    predecessor = f"{request_id}:prefill_{len(stages) - 1}"
    depends_on = {}
    for task in tasks:
        depends_on[task["task_id"]] = [predecessor]
        predecessor = task["task_id"]
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}


def plan_tp_prefill(request, tp_group, model_info):
    """在固定 TP Group 中逐层规划单请求 Prefill。"""
    return _plan_tp(request, tp_group, model_info, decode=False)


def plan_tp_decode(request, tp_group, model_info):
    """每轮由 Rank 0 广播上一个输出 Token，再执行各层 TP 计算。"""
    return _plan_tp(request, tp_group, model_info, decode=True)


def _tp_ranks(tp_group):
    if (not isinstance(tp_group, Sequence) or isinstance(tp_group, (str, bytes, bytearray))
            or len(tp_group) < 2):
        raise ValueError("tp_group 必须是包含至少两颗卫星的有序序列")
    ranks = list(tp_group)
    if any(type(node) is not int or node < 0 for node in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("TP 卫星编号必须是唯一的非负整数")
    return ranks


def _tp_group_connected(ranks, edges):
    neighbors = {sat: [] for sat in ranks}
    for a, b in edges:
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)
    reached = {ranks[0]}
    pending = [ranks[0]]
    while pending:
        for sat in neighbors[pending.pop()]:
            if sat not in reached:
                reached.add(sat)
                pending.append(sat)
    return len(reached) == len(ranks)


def _tp_tensor_layout(model):
    """当前 offline.decoder 支持的标准 dense 层，名称与 checkpoint 一致。"""
    family = model.get("model_type")
    if family not in ("qwen2", "qwen3"):
        raise ValueError("需要 model_type=qwen2 或 qwen3 以校验完整权重清单")
    if (model.get("quantization_config") or model.get("mlp_bias")
            or model.get("attention_bias", family == "qwen2") != (family == "qwen2")):
        raise ValueError("当前权重清单仅支持项目已有的未量化标准 Qwen dense 层")
    hidden, middle, dim = model["hidden_size"], model["intermediate_size"], model["head_dim"]
    q, kv = model["num_attention_heads"] * dim, model["num_key_value_heads"] * dim
    # PyTorch Linear.weight 的轴顺序为 [out_features, in_features]。
    layer = {
        "self_attn.q_proj.weight": ((q, hidden), 0),
        "self_attn.k_proj.weight": ((kv, hidden), 0),
        "self_attn.v_proj.weight": ((kv, hidden), 0),
        "self_attn.o_proj.weight": ((hidden, q), 1),
        "mlp.gate_proj.weight": ((middle, hidden), 0),
        "mlp.up_proj.weight": ((middle, hidden), 0),
        "mlp.down_proj.weight": ((hidden, middle), 1),
        "input_layernorm.weight": ((hidden,), None),
        "post_attention_layernorm.weight": ((hidden,), None),
    }
    if family == "qwen2":
        for projection, width in (("q", q), ("k", kv), ("v", kv)):
            layer[f"self_attn.{projection}_proj.bias"] = ((width,), 0)
    else:
        for prefix in ("q", "k"):
            layer[f"self_attn.{prefix}_norm.weight"] = ((dim,), None)
    layout = {}
    for i in range(model["num_hidden_layers"]):
        for suffix, (shape, axis) in layer.items():
            layout[f"model.layers.{i}.{suffix}"] = (i, shape, axis)
    layout["model.embed_tokens.weight"] = (None, (None, hidden), None)
    layout["model.norm.weight"] = (None, (hidden,), None)
    layout["lm_head.weight"] = (None, (None, hidden), None)
    return layout


def _tp_weight_tensors(model):
    """校验完整具名清单和存储别名；驻留位置由具体部署校验。"""
    layout = _tp_tensor_layout(model)
    manifest = model.get("weight_tensors")
    if not isinstance(manifest, (list, tuple)):
        raise ValueError("需要完整的 weight_tensors 具名张量清单")
    by_name = {}
    for tensor in manifest:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
            raise ValueError("weight_tensors 中每个张量必须提供名称")
        name = tensor["name"]
        if name in by_name or name not in layout:
            raise ValueError(f"重复或未建模的权重：{name}")
        by_name[name] = tensor
    missing = layout.keys() - by_name.keys()
    if missing:
        raise ValueError(f"缺少权重张量：{sorted(missing)}")

    descriptors = {}
    vocab = model.get("vocab_size")
    if vocab is not None and (type(vocab) is not int or vocab < 1):
        raise ValueError("vocab_size 必须是正整数")
    for name, (layer, expected_shape, expected_axis) in layout.items():
        tensor = by_name[name]
        required = ("layer_id", "shape", "storage_dtype", "split_axis", "storage_id")
        if any(field not in tensor for field in required):
            raise ValueError(f"{name} 缺少层、形状、存储精度、切分轴或共享存储标识")
        if type(tensor["layer_id"]) is not type(layer) or tensor["layer_id"] != layer:
            raise ValueError(f"{name} 的 layer_id 与名称不符")
        shape = tensor["shape"]
        if (not isinstance(shape, (list, tuple)) or len(shape) != len(expected_shape)
                or any(type(v) is not int or v <= 0 for v in shape)
                or any(e is not None and v != e for v, e in zip(shape, expected_shape))):
            raise ValueError(f"{name} 的 shape 与模型投影维度不符")
        if expected_shape[0] is None:
            if vocab is not None and vocab != shape[0]:
                raise ValueError(f"{name} 的词表维度不符")
            vocab = shape[0]
        axis, dtype = tensor["split_axis"], tensor["storage_dtype"]
        if type(axis) is not type(expected_axis) or axis != expected_axis:
            raise ValueError(f"{name} 的 split_axis 与切分规则不符")
        if not isinstance(dtype, str) or dtype not in _FLOAT_ELEMENT_BYTES:
            raise ValueError(f"{name} 缺少支持的部署存储 dtype")
        storage = tensor["storage_id"]
        if not isinstance(storage, str) or not storage.strip():
            raise ValueError(f"{name} 缺少明确的 storage_id")
        descriptor = (tuple(shape), dtype, axis)
        if descriptors.setdefault(storage, descriptor) != descriptor:
            raise ValueError(f"共享存储 {storage} 的形状、dtype 或切分布局冲突")
        yield tensor, prod(shape) * _FLOAT_ELEMENT_BYTES[dtype]


def _tp_weight_bytes(model, size, *, can_split):
    resident = [set() for _ in range(size)]
    layers = model["num_hidden_layers"]
    totals = [{"layer_shard_bytes": [0] * layers, "layer_replicated_bytes": [0] * layers,
               "outside_weight_bytes": 0} for _ in range(size)]
    all_ranks = set(range(size))
    for tensor, nbytes in _tp_weight_tensors(model):
        name, layer = tensor["name"], tensor["layer_id"]
        axis, storage = tensor["split_axis"], tensor["storage_id"]
        placement = tensor.get("resident_ranks")
        if (not isinstance(placement, (list, tuple)) or not placement
                or any(type(r) is not int or r not in all_ranks for r in placement)
                or len(set(placement)) != len(placement)):
            raise ValueError(f"{name} 需要明确且有效的 resident_ranks")
        if layer is not None and set(placement) != all_ranks:
            raise ValueError(f"{name} 的层内权重或均匀分片必须覆盖全部 Rank")
        if not can_split:
            continue
        if axis is not None:
            nbytes //= size
        for rank in placement:
            if storage in resident[rank]:
                continue
            resident[rank].add(storage)
            if layer is None:
                totals[rank]["outside_weight_bytes"] += nbytes
            else:
                field = "layer_replicated_bytes" if axis is None else "layer_shard_bytes"
                totals[rank][field][layer] += nbytes
    return totals


def _validate_tp_dimensions(model_info):
    if not isinstance(model_info, dict):
        raise ValueError("model_info 必须是字典")
    _pp_activation_bytes(model_info, 1)
    for field in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                  "intermediate_size", "head_dim"):
        value = model_info.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} 必须是正整数")
    if model_info["hidden_size"] != model_info["num_attention_heads"] * model_info["head_dim"]:
        raise ValueError("hidden_size 必须等于 num_attention_heads * head_dim")


def _validate_node_budgets(ranks, node_resources):
    if not isinstance(node_resources, dict) or any(type(s) is not int or s < 0 for s in node_resources):
        raise ValueError("node_resources 必须按卫星编号提供预算")
    for sat in ranks:
        node = node_resources.get(sat)
        for field in ("weight_memory_budget_bytes", "total_memory_budget_bytes"):
            if not isinstance(node, dict) or type(node.get(field)) is not int or node[field] < 0:
                raise ValueError(f"卫星 {sat} 缺少合法的 {field}")


def _tp_split_reasons(model_info, size):
    reasons = []
    for field in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if model_info[field] % size:
            reasons.append(f"{field} 不能被 TP Group Size 整除")
    if model_info["num_attention_heads"] % model_info["num_key_value_heads"]:
        reasons.append("num_attention_heads 不能被 num_key_value_heads 整除")
    return reasons


def assess_tp_group(request, tp_group, model_info, node_resources, *, active_links=None):
    """检查固定 TP Group 的均匀切分、逐 Rank 内存预算及组内快照连通性。"""
    ranks = _tp_ranks(tp_group)
    connected = _tp_group_connected(ranks, _pp_direct_edges(active_links))
    _validate_tp_dimensions(model_info)
    full_kv = pp_stage_kv_bytes(model_info, request, {
        "start_layer": 0, "end_layer": model_info["num_hidden_layers"],
    })
    _validate_node_budgets(ranks, node_resources)
    size = len(ranks)
    reasons = _tp_split_reasons(model_info, size)
    weights = _tp_weight_bytes(model_info, size, can_split=not reasons)
    reports = []
    if not reasons:
        kv_bytes = full_kv // size
        for rank, sat in enumerate(ranks):
            parts = weights[rank]
            weight = (sum(parts["layer_shard_bytes"]) + sum(parts["layer_replicated_bytes"])
                      + parts["outside_weight_bytes"])
            node = node_resources[sat]
            weight_fits = weight <= node["weight_memory_budget_bytes"]
            total_fits = weight + kv_bytes <= node["total_memory_budget_bytes"]
            if not weight_fits:
                reasons.append(f"Rank {rank} 卫星 {sat} 超出权重预算")
            if not total_fits:
                reasons.append(f"Rank {rank} 卫星 {sat} 权重与 KV 合计超出总预算")
            reports.append({"rank": rank, "node_id": sat, "weight_bytes": weight,
                            "kv_bytes": kv_bytes, "weight_fits": weight_fits,
                            "total_fits": total_fits, **parts})
    if not connected:
        reasons.append("指定快照中的 TP Group 诱导子图不连通")
    return {"feasible": not reasons, "reasons": reasons, "participants": ranks,
            "connected": connected, "ranks": reports}


def _hybrid_stages(stages, layer_count):
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError("Hybrid stages 必须是非空有序序列")
    used = set()
    covered = 0
    result = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or type(stage.get("stage_id")) is not int or stage["stage_id"] != index:
            raise ValueError("Hybrid stage_id 必须从 0 连续递增")
        start, end = stage.get("start_layer"), stage.get("end_layer")
        if (type(start) is not int or type(end) is not int or start != covered
                or not start < end <= layer_count):
            raise ValueError(f"Stage {index} 的层区间必须连续、非空且位于模型范围内")
        group = _tp_ranks(stage.get("tp_group"))
        if not used.isdisjoint(group):
            raise ValueError(f"Stage {index} 的卫星与其他 Stage 重叠")
        used.update(group)
        result.append({"stage_id": index, "start_layer": start, "end_layer": end, "tp_group": group})
        covered = end
    if covered != layer_count:
        raise ValueError("Hybrid stages 必须完整覆盖模型层")
    return result


def _hybrid_weight_bytes(model_info, stages):
    # Stage 已按全局层号完整划分。层外张量的组和驻留位置另行指定。
    by_layer = [stage for stage in stages for _ in range(stage["start_layer"], stage["end_layer"])]
    totals = {sat: {"shard_bytes": 0, "replicated_bytes": 0, "outside_weight_bytes": 0}
              for stage in stages for sat in stage["tp_group"]}
    resident = {sat: set() for sat in totals}
    for tensor, nbytes in _tp_weight_tensors(model_info):
        layer, name = tensor["layer_id"], tensor["name"]
        if layer is not None:
            stage = by_layer[layer]
            expected = stage["tp_group"]
            field = "shard_bytes" if tensor["split_axis"] is not None else "replicated_bytes"
        else:
            embedding = name == "model.embed_tokens.weight"
            stage = stages[0] if embedding else stages[-1]
            expected = stage["tp_group"] if embedding else stage["tp_group"][:1]
            field = "outside_weight_bytes"
        placement = tensor.get("resident_nodes")
        if (not isinstance(placement, (list, tuple)) or len(placement) != len(expected)
                or any(type(sat) is not int for sat in placement) or set(placement) != set(expected)):
            raise ValueError(f"{name} 的 resident_nodes 须与规定驻留卫星 {expected} 一致")
        if not stage["split_valid"]:
            continue
        if tensor["split_axis"] is not None:
            nbytes //= len(stage["tp_group"])
        for sat in expected:
            storage = tensor["storage_id"]
            if storage not in resident[sat]:
                totals[sat][field] += nbytes
                resident[sat].add(storage)
    return totals


def assess_hybrid_deployment(request, stages, model_info, node_resources, *, active_links=None):
    """核对固定 PP 层区间与 TP Group 的权重驻留、单请求峰值 KV 和当前链路快照。"""
    _, _, output_length = _request_lengths(request)
    edges = _pp_direct_edges(active_links)
    _validate_tp_dimensions(model_info)
    stages = _hybrid_stages(stages, model_info["num_hidden_layers"])
    _validate_node_budgets([sat for stage in stages for sat in stage["tp_group"]], node_resources)
    reasons = []
    for stage in stages:
        group = stage["tp_group"]
        split_reasons = _tp_split_reasons(model_info, len(group))
        stage["split_valid"] = not split_reasons
        stage["connected"] = _tp_group_connected(group, edges)
        stage_id = stage["stage_id"]
        reasons.extend(f"Stage {stage_id}: {reason}" for reason in split_reasons)
        if not stage["connected"]:
            reasons.append(f"Stage {stage_id} 的组内诱导子图不连通")

    roots = [stage["tp_group"][0] for stage in stages]
    pairs = [("activation", src, dst) for src, dst in zip(roots, roots[1:])]
    if len(roots) > 1 and output_length > 1:
        pairs.append(("token", roots[-1], roots[0]))
    required_links = []
    for purpose, src, dst in pairs:
        available = (min(src, dst), max(src, dst)) in edges
        required_links.append({"purpose": purpose, "src": src, "dst": dst, "available": available})
        if not available:
            reasons.append(f"{purpose} 缺少根节点直连边 {src} → {dst}")

    weights = _hybrid_weight_bytes(model_info, stages)
    nodes = []
    for stage in stages:
        stage_kv = pp_stage_kv_bytes(model_info, request, stage)
        if not stage["split_valid"]:
            continue
        group = stage["tp_group"]
        kv = stage_kv // len(group)
        for rank, sat in enumerate(group):
            parts = weights[sat]
            weight = sum(parts.values())
            budget = node_resources[sat]
            weight_fits = weight <= budget["weight_memory_budget_bytes"]
            total_fits = weight + kv <= budget["total_memory_budget_bytes"]
            if not weight_fits:
                reasons.append(f"卫星 {sat} 超出权重预算")
            if not total_fits:
                reasons.append(f"卫星 {sat} 权重与 KV 合计超出总预算")
            nodes.append({"node_id": sat, "stage_id": stage["stage_id"], "rank": rank,
                          **parts, "weight_bytes": weight, "kv_bytes": kv,
                          "weight_fits": weight_fits, "total_fits": total_fits})
    return {"feasible": not reasons, "reasons": reasons, "stages": stages,
            "nodes": nodes, "required_links": required_links}


def plan_hybrid_prefill(request, stages, model_info):
    """按固定部署生成 Prefill 任务；每个 Stage 边界由根节点直传并在接收组广播。"""
    request_id, input_length, _ = _request_lengths(request)
    activation_bytes = _pp_activation_bytes(model_info, input_length)
    layers = model_info.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ValueError("num_hidden_layers 必须是正整数")
    stages = _hybrid_stages(stages, layers)

    tasks: list[OperationComputeTask | TransferTask | CollectiveTask] = []
    depends_on: dict[str, list[str]] = {}
    predecessors = []
    for stage in stages:
        index = stage["stage_id"]
        group = stage["tp_group"]
        prefix = f"{request_id}:hybrid_prefill_stage_{index}"
        if index > 0:
            sender = stages[index - 1]
            transfer_id = f"{prefix}_activation_transfer"
            tasks.append(TransferTask(
                task_id=transfer_id, request_id=request_id, task_type="transfer",
                phase="prefill", decode_step=0, data_type="activation",
                src=sender["tp_group"][0], dst=group[0], data_size=activation_bytes,
            ))
            depends_on[transfer_id] = predecessors
            broadcast_id = f"{prefix}_activation_broadcast"
            tasks.append(CollectiveTask(
                task_id=broadcast_id, request_id=request_id, task_type="collective",
                phase="prefill", decode_step=0,
                collective_type="broadcast", participants=group.copy(), root=group[0],
                layer_id=sender["end_layer"] - 1, data_size=activation_bytes,
            ))
            depends_on[broadcast_id] = [transfer_id]
            predecessors = [broadcast_id]

        for layer in range(stage["start_layer"], stage["end_layer"]):
            for operation in ("attention", "mlp"):
                block = f"{prefix}_layer_{layer}_{operation}"
                local_ids = []
                for rank, sat in enumerate(group):
                    task_id = f"{block}_rank_{rank}"
                    tasks.append(OperationComputeTask(
                        task_id=task_id, request_id=request_id, task_type="operation_compute",
                        node_id=sat, batch_size=1, input_tokens=input_length,
                        context_tokens=input_length,
                        phase="prefill", layer_id=layer, operation=operation,
                        tp_size=len(group), rank=rank,
                    ))
                    depends_on[task_id] = predecessors.copy()
                    local_ids.append(task_id)
                reduce_id = f"{block}_all_reduce"
                tasks.append(CollectiveTask(
                    task_id=reduce_id, request_id=request_id, task_type="collective",
                    phase="prefill", decode_step=0,
                    collective_type="all_reduce", participants=group.copy(), root=None,
                    layer_id=layer, data_size=activation_bytes,
                ))
                depends_on[reduce_id] = local_ids
                predecessors = [reduce_id]
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}


def plan_hybrid_decode(request, stages, model_info):
    """逐轮返回 Token，按全局层顺序执行组内 TP 计算并交接 Stage 激活。"""
    request_id, input_length, output_length = _request_lengths(request)
    activation_bytes = _pp_activation_bytes(model_info, 1)
    layers = model_info.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ValueError("num_hidden_layers 必须是正整数")
    stages = _hybrid_stages(stages, layers)
    last_stage = stages[-1]
    predecessor = (f"{request_id}:hybrid_prefill_stage_{last_stage['stage_id']}"
                   f"_layer_{layers - 1}_mlp_all_reduce")
    tasks: list[OperationComputeTask | TransferTask | CollectiveTask] = []
    depends_on: dict[str, list[str]] = {}
    for step in range(1, output_length):
        for index, stage in enumerate(stages):
            group = stage["tp_group"]
            prefix = f"{request_id}:hybrid_decode_step_{step}_stage_{index}"
            # 首组接收末组采样的 Token；其余组接收前一组的完整隐藏状态。
            if index == 0:
                sender = last_stage
                payload, size, layer_id = "token", _TOKEN_ID_BYTES, None
            else:
                sender = stages[index - 1]
                payload, size, layer_id = "activation", activation_bytes, sender["end_layer"] - 1
            if len(stages) > 1:
                transfer_id = f"{prefix}_{payload}_transfer"
                tasks.append(TransferTask(
                    task_id=transfer_id, request_id=request_id, task_type="transfer",
                    phase="decode", decode_step=step, data_type=payload,
                    src=sender["tp_group"][0], dst=group[0], data_size=size,
                ))
                depends_on[transfer_id] = [predecessor]
                predecessor = transfer_id
            broadcast_id = f"{prefix}_{payload}_broadcast"
            tasks.append(CollectiveTask(
                task_id=broadcast_id, request_id=request_id, task_type="collective",
                phase="decode", decode_step=step,
                collective_type="broadcast", participants=group.copy(), root=group[0],
                layer_id=layer_id, data_size=size,
            ))
            depends_on[broadcast_id] = [predecessor]
            predecessor = broadcast_id
            for layer in range(stage["start_layer"], stage["end_layer"]):
                for operation in ("attention", "mlp"):
                    block = f"{prefix}_layer_{layer}_{operation}"
                    local_ids = []
                    for rank, sat in enumerate(group):
                        task_id = f"{block}_rank_{rank}"
                        tasks.append(OperationComputeTask(
                            task_id=task_id, request_id=request_id, task_type="operation_compute",
                            node_id=sat, batch_size=1, input_tokens=1,
                            context_tokens=input_length + step - 1,
                            phase="decode", layer_id=layer, operation=operation,
                            tp_size=len(group), rank=rank,
                        ))
                        depends_on[task_id] = [predecessor]
                        local_ids.append(task_id)
                    reduce_id = f"{block}_all_reduce"
                    tasks.append(CollectiveTask(
                        task_id=reduce_id, request_id=request_id, task_type="collective",
                        phase="decode", decode_step=step,
                        collective_type="all_reduce", participants=group.copy(), root=None,
                        layer_id=layer, data_size=activation_bytes,
                    ))
                    depends_on[reduce_id] = local_ids
                    predecessor = reduce_id
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}


def _plan_tp(request, tp_group, model_info, *, decode):
    request_id, input_length, output_length = _request_lengths(request)
    ranks = _tp_ranks(tp_group)
    if not isinstance(model_info, dict):
        raise ValueError("model_info 必须是字典")
    layers = model_info.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ValueError("num_hidden_layers 必须是正整数")
    input_tokens = 1 if decode else input_length
    activation_bytes = _pp_activation_bytes(model_info, input_tokens)
    tasks: list[OperationComputeTask | CollectiveTask] = []
    depends_on: dict[str, list[str]] = {}
    prefill_id = f"{request_id}:tp_prefill"
    predecessors = [f"{prefill_id}_layer_{layers - 1}_mlp_all_reduce"] if decode else []
    # step 0 为 Prefill；Decode 从 step 1 开始，消费前一阶段产生的 Token。
    steps = range(1, output_length) if decode else range(1)
    phase: Phase = "decode" if decode else "prefill"
    for step in steps:
        prefix = prefill_id
        context_tokens = input_length
        if step:
            prefix = f"{request_id}:tp_decode_step_{step}"
            context_tokens += step - 1
            broadcast = f"{prefix}_broadcast"
            tasks.append(CollectiveTask(
                task_id=broadcast, request_id=request_id, task_type="collective",
                phase=phase, decode_step=step,
                collective_type="broadcast", participants=ranks.copy(),
                data_size=_TOKEN_ID_BYTES, layer_id=None, root=ranks[0],
            ))
            depends_on[broadcast] = predecessors
            predecessors = [broadcast]
        for layer in range(layers):
            for operation in ("attention", "mlp"):
                block = f"{prefix}_layer_{layer}_{operation}"
                local_ids = []
                for rank, node in enumerate(ranks):
                    task_id = f"{block}_rank_{rank}"
                    tasks.append(OperationComputeTask(
                        task_id=task_id, request_id=request_id, task_type="operation_compute",
                        node_id=node, batch_size=1, input_tokens=input_tokens,
                        context_tokens=context_tokens,
                        phase=phase, layer_id=layer, operation=operation,
                        tp_size=len(ranks), rank=rank,
                    ))
                    depends_on[task_id] = predecessors.copy()
                    local_ids.append(task_id)
                reduce_id = f"{block}_all_reduce"
                tasks.append(CollectiveTask(
                    task_id=reduce_id, request_id=request_id, task_type="collective",
                    phase=phase, decode_step=step,
                    collective_type="all_reduce", participants=ranks.copy(),
                    data_size=activation_bytes, layer_id=layer, root=None,
                ))
                depends_on[reduce_id] = local_ids
                predecessors = [reduce_id]
    return {"request_id": request_id, "tasks": tasks, "depends_on": depends_on}
