# 卫星 LLM Serving 仿真平台设计

本文描述平台架构、模块职责与使用方法。策略层实现进度与验收范围见第 5 节。项目范围与实现原则见 [AGENTS.md](AGENTS.md)。

## 1. 总体架构

平台采用离线建模与在线离散事件仿真相结合的架构，分为四层：

| 层次 | 职责 | 主要输出 |
| --- | --- | --- |
| 离线输入层 | 生成卫星链路数据和 LLM 硬件性能 Profile，独立持久化并复用。 | Link Trace、Profile Database。 |
| 策略层 | 根据请求负载、动态拓扑和 Serving 方式安排部署、计算与传输。 | 时隙级计划计算流与传输流。 |
| 执行仿真层 | 在统一离散事件时钟下推进计算、传输和资源状态。 | 实际完成事件、资源使用与能耗记录。 |
| 性能评估层 | 汇总请求完成情况与资源使用数据，比较不同星座和策略。 | 系统性能指标。 |

整体数据流：

```text
轨道配置 / TLE → 链路生成 → Link Trace ─────────┐
                                             │
模型与设备配置 → 性能建模 → Profile Database ──┤
                                             ↓
请求负载 + Serving 方式 → 策略层 → 计划计算流 / 传输流
                            ↑                ↓
                            └── 运行状态 ← 执行仿真层
                                             ↓
                                         性能评估层
```

链路数据与硬件 Profile 可以分别生成、独立保存，不要求每次在线实验重新建模。

## 2. 统一时间与并发执行

整个系统由单一离散事件时钟驱动。网络传输、节点计算和电量变化可以同时发生，不按模块串行推进。

- 策略在时隙边界安排任务，执行层决定任务的实际开始与完成时间。
- 时隙不是任务完成边界；未完成的计算任务和传输流延续到后续时隙。
- 任务完成或链路状态变化时更新运行状态，并供策略层后续决策使用。
- 链路离线采样间隔与在线调度时隙是不同概念，不默认将二者等同。

共享运行状态包括：

- 请求执行进度与完成状态；
- 链路占用及正在传输的数据流；
- 节点任务队列、Batch 状态及 KV Cache 占用；
- 节点计算与通信能耗、电池剩余电量（SOC）。

同一时刻不同类型事件的处理顺序及策略调用的详细时机需在执行层接口中明确，不在此处预设。

## 3. 离线输入层：卫星轨道与链路状态

### 3.1 模块职责

生成不同星座构型下的动态网络拓扑，持久化链路建立、断开事件及传播时延。在线链路传输模块读取离线结果，不重新传播轨道。

本模块不执行路由、带宽分配、实际数据传输、计算调度或能耗计算，也不在事件文件中包含带宽。

### 3.2 轨道输入与位置传播

支持参数化 Walker Delta、Walker Star 和真实 TLE 数据。

**Walker 输入**

输入包括构型、总卫星数 T、轨道面数 P、相位因子 F、倾角和高度。要求 T、P 为正整数、T 能被 P 整除，且 `0 ≤ F < P`。采用圆轨道，每面 T/P 颗卫星，轨道面 p、面内槽位 s 均从 0 开始：

```text
Delta: RAAN_p = 360° * p / P
Star:  RAAN_p = 180° * p / P
两者:  M_p,s = (360° * s / (T/P) + 360° * F * p / T) mod 360°
```

本项目的两种构型均采用 `360° * F / T` 的相邻面相位偏移。外部工具对 Walker Star 的 F 可能采用不同定义，不能仅按 T/P/F 数字认定几何等价。

Walker 在仿真起始时刻初始化，统一使用 SGP4 和 WGS84 引力模型；不额外建模阻力，`bstar`、`ndot`、`nddot` 均为 0。

**TLE 输入**

- 指定 TLE 文件，支持无名称的两行格式和带名称的三行格式。
- 配置按轨道面显式列出 NORAD ID，仅读取列出的卫星。
- 外层列表定义轨道面顺序，面内列表定义固定的前后邻居顺序。
- 运行时不自动聚类或筛选壳层；更换 TLE 或大幅改变起点时需重新核对映射。

两类输入均使用 SGP4 返回的地心 TEME 三维位置，距离单位为 km。球形地球下的星间距离和遮挡计算不需要转换为地固坐标。

### 3.3 配置与时间约定

一个 YAML 配置文件同时保存通用参数和命名轨道预设，以 `selected` 选择单个案例：

```yaml
selected: starlink_delta_24
start_time_utc: "2026-01-01T00:00:00Z"
duration_s: 600
sample_step_s: 1
max_link_distance_km: 1500
output_dir: data/link_traces

presets:
  starlink_delta_24:
    source: walker
    pattern: delta
    total_satellites: 1584
    planes: 24
    phasing: 1
    altitude_km: 550
    inclination_deg: 53
```

- 默认时长 600 秒，采样间隔 1 秒；60 分钟实验使用 3600 秒。
- 起点必须显式带时区，统一归一化为 UTC。
- 预设内部的 `start_time_utc` 优先于顶层值，未设置时沿用顶层值。
- 相对时间从 0 开始，采样区间为 `[0, duration_s)`；默认采样时刻为 0、1、…、599 秒。
- Walker 默认固定起点为 `2026-01-01T00:00:00Z`。只考虑星间几何、不涉及日照和地面站时，10–60 分钟实验无需特意选择日期，比较实验保持起点一致。
- TLE 起点必须接近数据历元，短仿真时长不能弥补数据过旧的误差。
- 相对路径以项目根目录为基准，输出位于 `output_dir/selected/`。

完整预设与显式 NORAD 映射由 [configs/orbit.yaml](configs/orbit.yaml) 保存。

### 3.4 +Grid 最近邻连接

每颗卫星的目标邻居为同面前一颗、后一颗，以及左右相邻轨道面各一颗卫星。

**同面链路**

- 按初始顺序固定前后环形邻居：Walker 使用槽位顺序，TLE 使用配置列表顺序。
- 邻居不满足链路可用性条件时，不改连其他同面卫星。

**跨面链路**

- 每个采样时刻重新选择邻居。
- 先排除超过最大距离或被地球遮挡的候选，再在每侧相邻面中寻找最近卫星。
- 双方互选才建立无向边；距离并列时按面内列表顺序选择。

所有构型均按轨道面环形相邻，首尾面也相邻。每颗卫星最多四个不同邻居，可用邻居不足时不补足；重合位置不建立零长度链路。

该规则是项目的网络模型，不保证与商用星座的真实终端能力及链路规则一致。

### 3.5 链路可用性与传播时延

采用半径 6371 km 的球形地球，不增加大气裕量；默认最大链路距离为 1500 km。

设卫星位置为 p、q，且 d = q - p，线段上距离地心最近的点为：

```text
u = clip(-dot(p, d) / dot(d, d), 0, 1)
closest = p + u * d
```

对于非零长度链路，同时满足以下条件时可用：

- 卫星间距离不超过最大链路距离；
- 最近点到地心的距离不小于地球半径。

传播速度采用真空光速 299792.458 km/s：

```text
delay_ms = distance_km / 299792.458 * 1000
```

不使用有效光速折减系数，也不将传播时延等同于包含排队和带宽竞争的实际传输时间。

### 3.6 链路事件与输出契约

每个采样时刻比较当前与前一帧的链路集合：新增链路记录建立事件，消失链路记录断开事件，状态不变不输出。只有时延变化、链路仍可用时不输出新事件。

首帧将全部可用链路记录为建立事件，结束时不人为关闭链路。结果仅在 manifest 标记的 `[0, duration_s)` 内有效。下游在链路存续期间沿用最近一次建立事件的时延，重新建立时更新。

**link_events.parquet**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `time_s` | float64，非空 | 相对 UTC 起点的秒数。 |
| `sat_i` | int64，非空 | 0-based 链路端点。 |
| `sat_j` | int64，非空 | 另一端点，满足 `sat_i < sat_j`。 |
| `delay_ms` | 可空 float64 | 数值表示建立及当时传播时延，真实 NULL 表示断开。 |

事件按时间、端点排序；同一时刻的事件表示同一次拓扑更新。不添加事件类型字段、带宽字段或完整拓扑快照。

**manifest.yaml**

记录完整配置、所选轨道参数、实际 UTC 起点、时长、采样间隔、有效区间、传播器、坐标系、引力模型、物理常数、链路规则、卫星数和事件数。

卫星映射包含 ID、名称、轨道面和槽位；TLE 模式增加 NORAD ID、源文件路径和 SHA-256 摘要。manifest 顶层及其中的 `config.start_time_utc` 均表示实际使用的起点。

### 3.7 Walker 预设与建模依据

以下预设是理想化轨道模型，不能视为商用系统当前在轨状态的精确复现。

| 预设 | 构型 | T / P / F | 高度 km | 倾角 ° | 理想同面距离 km | 相位依据与限制 |
| --- | --- | --- | --- | --- | --- | --- |
| `starlink_delta_24`（默认） | Delta | 1584 / 24 / 1 | 550 | 53 | 659 | 历史设计几何下的论文模型。 |
| `starlink_delta_72` | Delta | 1584 / 72 / 1 | 550 | 53 | 1970 | 论文模型，同面邻居超出默认距离限制。 |
| `iridium_star` | Star | 66 / 6 / 2 | 780 | 86.4 | 4029 | Brahe 的 360° 相位定义，同面邻居超出默认距离限制。 |
| `oneweb_star` | Star | 648 / 18 / 9 | 1200 | 87.4 | 1320 | F=9 为项目公式下的显式假设，面间偏移 5°；来源的相位公式未核实。 |
| `kuiper_delta` | Delta | 1296 / 36 / 1 | 610 | 42 | 1217 | 壳层几何有公开依据，F=1 为显式假设，面间偏移约 0.277778°。 |

理想同面距离按 `2 * (6371 + altitude_km) * sin(pi / satellites_per_plane)` 估算，不是传播后的测量结果。满足同面距离约束不代表跨面可用或全网连通。

OneWeb 一代未采用星间链路，为其添加 +Grid 属于假设网络。OneWeb、Kuiper 的相位假设必须在配置和元数据中明确标记，不宣称为运营方确认的参数。

### 3.8 Dove TLE 预设

`dove_tle` 使用 [Dove.tle](../starrynet_analyze/tle/Dove.tle) 的 197 颗卫星，起点为 `2022-04-23T12:00:00Z`，对应文件中 2022-04-23 的历元，不表示当前实时数据。

- 采用按倾角及 RAAN 划分的 10 个近似轨道面，映射来源为已有 `sn_constellation_analysis.detect_planes` 分面方法。
- 每面星数依次为 `[6, 57, 17, 65, 1, 24, 3, 7, 1, 16]`，保留单星面。
- 轨道面按 RAAN 升序，首尾按项目规则相邻。
- 各星在同一起点的纬度幅角按 `[0, 2π)` 升序定义面内顺序，同角度时按 NORAD ID 排序；不混用各自 TLE 历元的平均近点角。
- 配置保存固定的 NORAD 列表，运行时不执行分面算法。

该分组是几何近似，不是运营方确认的分面，+Grid 也不代表 Dove 的实际通信拓扑。

## 4. 离线输入层：硬件性能 Profile

### 4.1 设计思路

Profile 使用本地模型和真实文本产生层输入，分别描述 Chunked Prefill 与 KV Cache 持续增长的 Decode。当前 Qwen2.5-7B-Instruct 选择首层、中间层和末层 `[0, 14, 27]` 作为同构 decoder layer 的代表；其余层仍执行一次以维持真实层间激活和各层 KV，不对未测层伪造结果。

权重保存在 CPU，GPU 只驻留当前 decoder layer；所有层的 KV Cache 保留在 GPU。逐层计时不包含权重传输、Embedding、最终归一化和 LM Head。结果记录单层时延、整卡平均功率和进程峰值 GPU 已分配内存。

### 4.2 关键参数与实现路径

- 模型：本地 Qwen2.5-7B-Instruct，BF16，28 个 dense decoder layers。
- 数据：本地 ShareGPT，经模型聊天模板分词，不拼接请求；Prefill 取第一条长度至少 2048 token 的请求并裁为 2048 token，Decode 为 Batch Size 128 保留足量请求池。
- Prefill：Batch Size 1，Chunk Size 为 `64 / 128 / 256 / 512`；每个 Chunk 保留其自然 `context_length`，历史 KV 与 RoPE 位置跨 Chunk 连续。
- Decode：Batch Size 为 `1 / 32 / 64 / 128`，初始上下文为 `128 / 512 / 1024 / 2048`，每个组合连续 64 步且 KV 每步增长。
- 测量：每个 case 预热 5 次、CUDA Event 采样 30 次；NVML 对固定 case 做稳态重复采样。
- 输出：Prefill 和 Decode 分别保存 JSON，包含 `model_id`、`device_type`、`model_layers`、`latency_ms`、`avg_power_w` 和 `peak_memory_bytes`；已有结果不覆盖。

实现流程：读取本地模型与数据 → 构造真实层输入和 KV → 顺序执行全部层 → 测量代表层 → 保存 JSON。固定入口为 `python -m offline.profile`，参数见 [configs/profile.yaml](configs/profile.yaml)。

## 5. 策略层

**实现进度**：Local、PD、PP、TP 和 Hybrid 已完成第一版单请求规划及统一 Plan 输出。验收覆盖三个任务类型的字段、两个请求分别运行五种策略时的 ID 与可变数据隔离、拓扑顺序、All-Reduce 的多前驱依赖，以及单 Token 输出、单 Stage 和准入失败分支。从项目根目录运行 `python -B -m unittest discover -s tests -v` 可复验。

策略层根据请求负载、动态网络拓扑和预设的 LLM Serving 方式，确定请求的部署与执行方案。

| 执行方式 | 定义 |
| --- | --- |
| 本地推理（Local / Co-located） | Prefill、Decode 在同一节点执行。 |
| PD 分离（PD Disaggregation） | Prefill、Decode 分配到不同节点，通过传输 KV Cache 衔接。 |
| 流水线并行（PP） | 模型划分为多个阶段，部署在不同节点并传输中间激活。 |
| 张量并行（TP） | 模型算子分布到多个节点，通过通信完成协同计算。 |
| 混合并行（Hybrid） | 组合不同并行方式。 |

策略负责节点选择、模型部署、任务分配、Batch 组织和执行顺序，输出每个时隙计划执行的计算流与传输流（Slot-level Data Flows），不直接计算实际执行耗时。

### 5.1 单请求 PP 调用流程

从项目根目录运行以下 Python 示例。权重、预算和时延为可手算的测试数据，实际实验应提供对应模型、设备和测量条件的数据。

```python
from strategy import Request, assemble_plan, build_pp_stages_for_request, plan_pp_decode, plan_pp_prefill

request = Request(request_id="r1", arrival_time=0, input_length=3, output_length=3,
                  service_requirements=None)
model = {
    "num_hidden_layers": 4, "layer_weight_bytes": [10, 10, 10, 10],
    "profile_case": "synthetic-b1",
    "layer_latency_ms_by_device": {"A": [1, 1, 9, 9], "B": [9, 9, 1, 9], "C": [9, 9, 9, 1]},
    "num_key_value_heads": 1, "head_dim": 2, "kv_cache_dtype": "bfloat16",
    "hidden_size": 8, "activation_dtype": "float32",
}
nodes = [
    {"sat_id": 42, "device_type": "A", "weight_memory_budget_bytes": 20, "total_memory_budget_bytes": 100},
    {"sat_id": 3, "device_type": "B", "weight_memory_budget_bytes": 10, "total_memory_budget_bytes": 50},
    {"sat_id": 19, "device_type": "C", "weight_memory_budget_bytes": 10, "total_memory_budget_bytes": 50},
]
links = {(3, 42): 2.5, (3, 19): 1.8, (19, 42): 3.0}
stages = build_pp_stages_for_request(model, nodes, request=request, active_links=links)
prefill = plan_pp_prefill(request, stages, model)
decode = plan_pp_decode(request, stages, model)
plan = assemble_plan("pp", request, prefill, decode)
```

示例划分依次为卫星 42 的 `[0,2)`、卫星 3 的 `[2,3)`、卫星 19 的 `[3,4)`。Prefill 有 5 个任务，随后两轮 Decode 各有 6 个任务。第一轮的 Token Transfer 从 19 返回 42，其直接前驱是 `r1:prefill_2`。Prefill 和每轮 Decode 的 Activation 都按 42 → 3 → 19 传递，每段负载分别为 96 和 32 字节，每次返回 Token 为 8 字节。

### 5.2 输入与输出

五种策略的单请求正式输出均包含 `request_id`、`strategy`、按拓扑顺序排列的 `tasks`，以及记录每项任务直接前驱的 `depends_on`。Local 和 PD 传入一个完整规划，组装时将短 ID 改为 `{request_id}:{strategy}:{原 task_id}`，依赖同步改名。

```python
from strategy import assemble_plan, plan_local_request, plan_pd_request_at

local = assemble_plan("local", request, plan_local_request(request, node_id))
pd_part = plan_pd_request_at(request, prefill_candidates, decode_candidates,
                             decision_time_s=now_s, topology_provider=query_links)
pd = None if pd_part is None else assemble_plan("pd", request, pd_part)
```

示例中的 `now_s` 是执行层发起本次决策时的仿真秒数，与离线链路的 `time_s` 共用起点。`query_links` 是调用方注入的函数，未来可传入 transmission 的绑定方法。它须返回该时刻完整的有效链路字典，键为严格升序的卫星编号对，值为传播时延（ms）。查询失败应抛出异常；PD 校验快照后按原规则选点。有效快照中没有直连候选对时返回 `None`。已知部署节点的调用方仍可直接使用 `plan_pd_request(request, prefill_node, decode_node)`。

PP、TP 和 Hybrid 依次传入 Prefill 和 Decode 规划结果，即使 Decode 没有任务也传入该片段。组装保留原任务 ID，遇到请求 ID 不一致、重复任务或依赖键、依赖不完整或顺序错误时抛出 `ValueError`。五种策略在同一合成请求下的完整输出见 [Plan 示例](examples/strategy_plans.json)。

调用方应使用同一策略、请求参数和部署生成所有片段。片段只携带请求 ID、任务和依赖。组装函数检查图结构，无法据此认证片段来源或 Decode 轮数。例如，将一对有效的 TP 片段标记为 PP，或给需要 Decode 的请求传入结构合法的空 Decode，均可能通过组装。

请求级部署检查发生在规划之前。PP 划分失败会抛出异常；TP 和 Hybrid 调用方须检查 `assessment["feasible"]`，只在通过时生成片段并组装。单独调用 `assemble_plan` 不会重新评估部署。Local 直接使用指定节点，PD 可使用现有 `select_pd_nodes` 选点。

| 输入 | 字段及含义 |
| --- | --- |
| 请求 | 非空字符串 `request_id`、数值 `arrival_time`、正整数 `input_length` 和 `output_length`、`service_requirements`。输出长度包含 Prefill 产生的首个 Token；当前搜索使用请求长度。 |
| 模型与 Profile | `num_hidden_layers` 给出模型层数，`layer_weight_bytes` 为各层权重字节数。`profile_case` 标识选定的测量条件，`layer_latency_ms_by_device` 按设备类型提供该条件下每一层的计算时延（ms）。 |
| KV 与 Activation | `num_key_value_heads`、`head_dim`、`kv_cache_dtype` 用于 KV 准入；发生 Activation Transfer 时另需 `hidden_size` 和 `activation_dtype`。推理批大小为 1。 |
| 候选节点 | 唯一的非负整数 `sat_id`，与 Profile 匹配的 `device_type`，独立权重上限 `weight_memory_budget_bytes`，以及扣除其他固定预留后供逐层权重与该请求 KV 共用的 `total_memory_budget_bytes`。 |
| 链路快照 | `active_links` 的键为无向端点对，值为有限非负传播时延（ms）。搜索只检查边是否存在。空字典仅允许单 Stage；缺失快照或传入 `None` 会报错。 |

`build_pp_stages(model_info, candidate_nodes)` 按权重预算和指定 Profile 划分；需要请求 KV 与快照准入时，使用示例中的 `build_pp_stages_for_request()`。两者均最小化最大 Stage 的逐层计算时延之和，同瓶颈时优先较少 Stage。输出均为有序字典列表，字段为 `stage_id, node_id, start_layer, end_layer`。两种任务规划器都可直接消费该列表。非法资源输入抛出 `ValueError`，合法输入无可行完整划分时抛出 `RuntimeError`。

每个阶段返回 `request_id, tasks, depends_on`。计算和传输任务均含 `task_id, request_id, task_type`。计算任务另含 `node_id, start_layer, end_layer, batch_size, input_tokens, context_tokens, num_layers`，其中层区间左闭右开，`num_layers` 保留为 `None`。传输任务另含 `src, dst, data_size`，单位为字节。`depends_on` 将任务 ID 映射到直接前驱 ID 列表，首个 Prefill 对应空列表。Decode 的第一个前驱来自同一请求的 Prefill 计划，因此应传入同一份完整 Stage 划分。

多 Stage 的每对相邻节点都需要直连边。当 `output_length > 1` 时，末节点和首节点还需直连，每轮 Decode 开始前返回一个 Token。单 Stage 只有计算任务。`output_length=1` 时无需返回边，Decode 的任务列表和依赖映射均为空。

### 5.3 字节口径与资源边界

| 对象 | 单请求计算口径 |
| --- | --- |
| Stage 的峰值 KV | `pp_stage_kv_bytes(model, request, stage)` 返回 `2 × Stage 层数 × (input_length + output_length - 1) × num_key_value_heads × head_dim × KV 元素字节数`。K 和 V 各一份，Prefill 与后续 Decode 共用。 |
| Stage 内存准入 | `pp_stage_fits_memory(model, request, stage, node)` 返回布尔值。区间权重须满足独立权重上限，权重加峰值 KV 须满足合计上限，等于上限时通过。 |
| Activation Transfer | 每次为 `token 数 × hidden_size × Activation 元素字节数`。Prefill 取完整输入长度，Decode 每轮取 1。KV 和 Activation 独立指定 dtype；两者均支持 float16、bfloat16（2 字节）和 float32（4 字节）。 |
| Token Transfer | 每次为一个生成的 int64 Token ID，共 8 字节，与隐藏维度和浮点 dtype 无关。 |

内存准入仅覆盖已提供的逐层权重和紧凑 KV。Embedding、最终 Norm、LM Head 等未计入逐层权重的驻留内存，以及激活缓冲、算子临时工作区和分配器开销，需要调用方另行核算并预留。传输字节数只描述负载，不能据此推断峰值显存或完成时间。

精确划分搜索的规模随候选卫星数量指数增长。结果适用于给定请求、Profile case 和直连快照；带宽、协议开销、路由、链路后续变化、多请求竞争和执行耗时均在本次规划范围之外。手工截短 Stage 列表时，任务规划器也无法独立识别缺失的尾层，完整覆盖应由划分器保证。

### 5.4 固定 TP Group 的单请求规划

`plan_tp_prefill(request, tp_group, model_info)` 和 `plan_tp_decode(request, tp_group, model_info)` 返回 `request_id, tasks, depends_on`。两阶段使用同一请求、同一有序卫星组和同一模型信息。组中至少需要两颗不同的卫星，首节点是 Rank 0。模型须提供正整数 `num_hidden_layers`、`hidden_size`，以及 `activation_dtype`（float16、bfloat16 或 float32）。

每层分别对 Attention 和 MLP 做列并行与行并行的局部计算抽象。每一段中，各 Rank 的计算任务共享同一个前置条件，可以并行执行。随后的求和 All-Reduce 等待本段所有 Rank 完成，归约结果再供下一段或下一层使用。每个 All-Reduce 对应一个 `CollectiveTask`，其 `participants` 保留 Rank 顺序，`layer_id` 从 0 起算。`data_size` 记录逻辑张量的字节数，Prefill 为输入长度 × 隐藏维度 × 元素字节数，每轮 Decode 为隐藏维度 × 元素字节数。

Rank 0 负责抽象的 Token Sampling。每轮 Decode 以一次 8 字节 int64 Token 广播开始，`layer_id=None`、`root=tp_group[0]`（卫星编号）。All-Reduce 的 `root=None`。第一轮广播依赖 Prefill 最后一层的 MLP All-Reduce，其余广播依赖上一轮最后一层的 MLP All-Reduce。输出长度为 1 时 Decode 计划为空。

### 5.5 单请求 TP 的部署准入与完整任务计划

调用方准备统一 `request`、含完整权重清单的 `model_info`、有序卫星列表 `tp_group`、各卫星预算 `node_resources` 和同一时刻的链路快照 `active_links`。先评估该请求能否部署，再用通过准入的 Rank 顺序生成两个阶段的任务。

```python
from strategy import assemble_plan, assess_tp_group, plan_tp_prefill, plan_tp_decode

assessment = assess_tp_group(request, tp_group, model_info, node_resources,
                             active_links=active_links)
plan = None
if assessment["feasible"]:
    ranks = assessment["participants"]
    prefill = plan_tp_prefill(request, ranks, model_info)
    decode = plan_tp_decode(request, ranks, model_info)
    plan = assemble_plan("tp", request, prefill, decode)
```

准入失败时，`plan` 为 `None`，原因在 `assessment["reasons"]` 中。准入成功时，合并后的 `depends_on` 可直接解析首轮 Broadcast 对 Prefill 末层 MLP All-Reduce 的引用。任务列表按拓扑顺序排列，同一段各 Rank 的局部计算可并行开始；每次 All-Reduce 等待该段全部 Rank。若 `output_length=1`，合并结果仅含 Prefill。

验收用例使用两层、17 个输入 Token、`hidden_size=16` 和 float32 Activation。四颗卫星按 `[42, 3, 19, 7]` 排列，组内链式连通。当 `output_length=3` 时，Prefill 有 20 个任务，两轮 Decode 各有 21 个，合计 62 个。Prefill 和 Decode 的每次 All-Reduce 逻辑负载分别为 1088 和 64 字节；每轮 Decode 开始时由卫星 42 广播一个 8 字节 Token ID。

在 5.4 的模型字段上增加 `model_type`、`num_attention_heads`、`num_key_value_heads`、`head_dim`、`intermediate_size`、`kv_cache_dtype` 和 `weight_tensors`。当前权重清单按项目支持的标准未量化 dense Qwen2/2.5（`qwen2`）和 Qwen3（`qwen3`）参数名核对。TP 准入要求 `hidden_size == num_attention_heads * head_dim`。Query 头、KV 头和 MLP 中间通道须分别能被组大小 P 整除，Query 头数也须能被 KV 头数整除。每个 Rank 参与所有层并保留完整的 head 维度。

`weight_tensors` 必须由调用方按实际部署补齐，每项字段如下。例如两 Rank 部署中的一条 Q 投影记录可以是：

```python
{"name": "model.layers.0.self_attn.q_proj.weight", "layer_id": 0, "shape": [16, 16],
 "storage_dtype": "float16", "split_axis": 0, "resident_ranks": [0, 1],
 "storage_id": "layer0.q.weight"}
```

- `name` 是完整 checkpoint 参数名；`layer_id` 为从 0 起算的层号，层外为 `None`。`shape` 填切分前形状。每个张量显式提供部署存储精度 `float16`、`bfloat16` 或 `float32`。清单须含全部层和三项层外权重，不能只提供层权重总数或 Profile 显存峰值。
- PyTorch Linear 权重布局为 `[out, in]`。Q/K/V 和 gate/up 沿轴 0 切分，o/down 沿轴 1 切分。Qwen2 的 Q/K/V bias 随输出分片。两处层内 RMSNorm 及 Qwen3 的 Q/K Norm 在每个 Rank 复制，显式填 `split_axis=None`。
- `resident_ranks` 是实际保存该张量的 **Rank 索引**。层内张量覆盖所有 Rank。层外 `model.embed_tokens.weight`、`model.norm.weight`、`lm_head.weight` 分别指定保存整份权重的 Rank，显式填 `split_axis=None`。本阶段的切分范围是 decoder 层内的 Attention 和 MLP。
- 同一物理权重在多个名字下共享时使用相同 `storage_id`，独立副本使用不同 ID。仅支持形状、精度、切分布局相同的整张量别名。每个 Rank 单独去重；明细按层号、层内参数顺序、最后层外权重的顺序归属，与输入清单的顺序无关。

`node_resources` 为按卫星编号索引的字典，例如 `{42: {"weight_memory_budget_bytes": 1480, "total_memory_budget_bytes": 1816}, ...}`。每 Rank 的权重总量分别计入本地分片、完整复制参数和层外驻留权重。成功均分 KV 头后，单请求峰值 KV 等于覆盖全部层的紧凑 KV 字节数除以 P，容量沿用 `input_length + output_length - 1`。权重须满足权重预算，权重加 KV 须满足总预算。激活缓冲、集合通信工作区与框架开销由调用方在总预算中预留。

`active_links` 与 PP 使用相同的无向快照格式。准入要求仅由组内成员和成员之间的边构成的图连通，链式连接可行。空快照有效但不连通，缺失或 `None` 属于输入错误。Prefill 的 All-Reduce 与 Decode 的 Broadcast 采用同一连通条件。

返回字典的 `participants` 保留 Rank 顺序，`connected` 表示快照连通性，`feasible` 和 `reasons` 给出整体结论。`ranks` 中逐项含 `rank, node_id, weight_bytes, kv_bytes, weight_fits, total_fits`，以及按层列出的本地 `layer_shard_bytes, layer_replicated_bytes` 和层外 `outside_weight_bytes`。均分条件失败时返回空的 `ranks`；预算不足或拓扑不连通时仍可查看有效的内存明细。缺失或矛盾的输入抛出 `ValueError`。这些结果仅针对声明的部署和快照，不代表整卡实测峰值或通信完成时间。

### 5.6 固定 Hybrid 部署准入

`assess_hybrid_deployment(request, stages, model_info, node_resources, active_links=links)` 检查一个请求的固定 PP 层划分和各段的 TP Group。下面的部署覆盖三层模型，两个 Stage 分别使用两颗和四颗卫星。

```python
stages = [
    {"stage_id": 0, "start_layer": 0, "end_layer": 2, "tp_group": [42, 3]},
    {"stage_id": 1, "start_layer": 2, "end_layer": 3, "tp_group": [19, 7, 25, 11]},
]
```

层区间须按序完整覆盖模型，各组的成员互不重叠。组内列表顺序决定本 Stage 的 Rank，首个节点是根。模型和节点预算沿用 5.5 的契约；具名权重清单的驻留字段使用物理卫星 ID 列表 `resident_nodes`。层内张量必须驻留于所属组全体成员；Embedding 驻留于首组全体成员；最终 Norm 和 LM Head 驻留于末组根节点。清单必须完整且与声明的放置一致。内部 TP 投影按本组大小均分，复制权重按整份计费，兼容的共享存储按物理卫星独立去重。

每颗卫星的峰值 KV 等于其 Stage 的完整 KV 字节数除以本组大小。每颗卫星同时检查权重预算和权重加 KV 的合计预算；激活缓冲、Collective 工作区和框架占用由调用方另行预留。快照上需满足每个组内部连通和相邻根直连；多个 Stage 且请求需要 Decode 时，末根还需直连首根。相反方向可以共用一条无向边。

结果含 `feasible, reasons, stages, nodes, required_links`。`stages` 按输入顺序报告 `split_valid` 和 `connected`；`nodes` 按 Stage、Rank 顺序列出权重分片、复制及层外权重、峰值 KV 和两项预算结论。切分不成立的 Stage 不输出节点字节数，其他可核算 Stage 仍给出明细。`required_links` 列出跨组交付的用途（`activation` 或 `token`）、方向和当前直连可用性；组内只报告连通性。缺失快照、清单矛盾或输入结构非法抛出 `ValueError`。合法输入下的切分、预算或连通条件失败通过 `feasible=False` 和具体原因报告。本阶段输出部署评估。

### 5.7 单请求 Hybrid Prefill

```python
assessment = assess_hybrid_deployment(request, stages, model_info, node_resources,
                                      active_links=active_links)
if assessment["feasible"]:
    prefill = plan_hybrid_prefill(request, assessment["stages"], model_info)
```

任务计划包含 `request_id, tasks, depends_on`。首组所有 Rank 从相同输入和本地 Embedding 开始。每个 Stage 只遍历自己的全局层区间，每层按 Attention、MLP 两段计算；各段的 Rank 可并行，随后由一个 All-Reduce 等待该段所有 Rank。Stage 边界依次生成一次根到根 Activation Transfer 和一次接收组 Broadcast，后者的 `root` 为接收根卫星编号，`layer_id` 为发送组末层。接收组的首层 Attention 都等待这个 Broadcast。

组内归约、跨组直传和接收组广播各自记录完整逻辑负载 `input_length × hidden_size × activation 元素字节数`。末组末层的 MLP All-Reduce 是整份计划的完成任务，末组 Rank 0 的采样沿用抽象语义。

### 5.8 单请求 Hybrid 完整调用流程

沿用 5.6 的 Stage 和模型具名权重清单。下面的示例输入 17 个 Token，生成 3 个 Token。只有通过准入的部署才生成任务计划。

```python
from strategy import assemble_plan, assess_hybrid_deployment, plan_hybrid_prefill, plan_hybrid_decode

request = {"request_id": "hybrid", "arrival_time": 0, "input_length": 17,
           "output_length": 3, "service_requirements": None}
node_resources = {sat: {"weight_memory_budget_bytes": 5000, "total_memory_budget_bytes": 9000}
                  for stage in stages for sat in stage["tp_group"]}
active_links = {(3, 42): 1, (7, 19): 1, (7, 25): 1, (11, 25): 1, (19, 42): 1}
assessment = assess_hybrid_deployment(request, stages, model_info, node_resources,
                                      active_links=active_links)
plan = None
if assessment["feasible"]:
    admitted = assessment["stages"]
    prefill = plan_hybrid_prefill(request, admitted, model_info)
    decode = plan_hybrid_decode(request, admitted, model_info)
    plan = assemble_plan("hybrid", request, prefill, decode)
else:
    print(assessment["reasons"])
```

单 Stage 时，Rank 0 本地采样，直接向本组广播 8 字节 Token ID，省去本地 Token Transfer。多 Stage 时，末组 Rank 0 先将 Token ID 传给首组 Rank 0，再由首组 Rank 0 广播。首轮的第一个通信任务依赖 Prefill 末层的 MLP All-Reduce，后续轮次依赖上一轮的末层 MLP All-Reduce。

每轮消费一个 Token，`context_tokens` 为本轮追加前的上下文长度。各 Stage 内的 All-Reduce、各边界的 Activation Transfer 和接收组 Broadcast 均记录一个 Token 的隐藏状态字节数。最后一轮以末层 MLP All-Reduce 结束。`output_length=1` 时只有 Prefill 任务。上述两 Stage 部署通过准入时产生 24 个 Prefill 任务和两轮各 26 个 Decode 任务，共 76 个。完整测试输入见 [Hybrid 集成测试](tests/test_hybrid_integration.py)。

## 6. 执行仿真层

**实现进度**：当前为模块骨架。策略层交付单请求任务、逻辑通信负载及依赖关系。以下各节描述待实现的执行职责，包括仿真时钟、时隙调度、路由、带宽分配以及实际开始与完成时间。

### 6.1 链路传输

读取 Link Trace，按事件维护动态拓扑，结合链路带宽及并发传输竞争推进传输。职责包括：

- 链路可达性判断与多跳传输；
- 带宽分配；
- 传输进度与完成时间计算；
- 通信能耗计算。

带宽由本模块配置，不来自离线链路事件文件。路由、带宽共享、断链后未完成传输的处理和通信能耗模型需另行明确。

### 6.2 节点计算

根据任务与硬件 Profile 推进 Prefill、Decode 等计算，考虑 Batch Size、输入长度、上下文长度、设备类型和模型配置。职责包括：

- 节点任务队列与 Batch 状态；
- 执行进度与完成时间计算；
- KV Cache 内存占用；
- 计算能耗。

队列规则、Batch 更新、KV Cache 生命周期和计算能耗模型需另行明确。

### 6.3 仿真调度与运行状态

统一管理事件队列、仿真时钟和共享运行状态，协调策略、传输与计算模块。任务完成或拓扑变化后更新状态；未完成任务跨时隙保留，不在时隙边界重复开始。

电池容量、初始 SOC、充放电模型及能量约束需另行明确。

## 7. 性能评估层

汇总任务完成记录与资源使用数据，从五个维度评估不同星座构型和 LLM Serving 策略。

五个维度的具体定义、指标公式、统计口径和输出格式尚待提供，不以常见指标代替原设计要求。

## 8. 整体项目代码框架

以下目录树描述完整目标结构。未确定数据契约的模块只保留职责骨架。

```text
satllm_simulator/
├── CLAUDE.md                 # 项目范围、实现原则与本地开发约束
├── DESIGN.md                 # 整体规划、模型与接口约定、代码框架
├── README.md                 # 使用方法、实现状态与运行结果
├── requirements.txt          # 链路生成依赖
├── requirements-profile.txt  # GPU Profile 独立依赖
├── configs/
│   ├── orbit.yaml            # 链路通用参数与 Walker / TLE 预设
│   └── profile.yaml          # 本地模型、GPU 与 Prefill / Decode 测量条件
├── offline/
│   ├── orbit.py              # Walker / TLE 初始化与轨道传播
│   ├── topology.py           # +Grid、可用性与传播时延
│   ├── generate_links.py     # 离线链路数据生成入口
│   ├── decoder.py            # Decoder 算子、因果 mask 与真实权重读取
│   ├── workload.py           # ShareGPT 选择、本地聊天模板分词与来源摘要
│   ├── power.py              # NVML 稳态板卡功率采样
│   └── profile.py            # 真实层间执行、连续 Decode、逐层测量与 JSON 输出
├── strategy/              # Serving 方式的部署与任务计划
├── execution/
│   ├── simulator.py          # 统一时钟、事件队列与共享运行状态
│   ├── transmission.py       # 多跳传输、带宽竞争与通信能耗
│   └── computation.py        # 节点队列、Batch、KV Cache 与计算能耗
├── metrics.py                # 性能指标汇总
├── run_simulation.py         # 在线仿真实验入口
└── data/
    ├── link_traces/
    │   └── <preset>/
    │       ├── link_events.parquet
    │       └── manifest.yaml
    └── profiles/             # 硬件性能数据
```

### 8.1 模块边界

| 模块 | 输入 | 输出与职责边界 |
| --- | --- | --- |
| `offline/orbit.py` | Walker 参数或 TLE、起点与相对时刻。 | 卫星映射、轨道面分组及 TEME 位置；不计算链路。 |
| `offline/topology.py` | 卫星位置、轨道面分组、最大距离。 | 可用无向边及传播时延；不做文件 I/O。 |
| `offline/generate_links.py` | 轨道配置。 | 串联传播、拓扑计算、事件差分，写出事件和 manifest。 |
| `offline/decoder.py` | 支持的模型配置、单层权重、输入与 KV。 | 单层算子、KV 写入与因果注意力，不执行完整模型或文件输出。 |
| `offline/workload.py` | ShareGPT、本地 tokenizer 与取样规则。 | 真实 token 请求及可复现来源摘要。 |
| `offline/power.py` | 固定层输入调用、NVML 与采样窗口。 | 稳态平均板卡功率，不推测缺失数据。 |
| `offline/profile.py` | 本地模型、真实请求、CUDA 与测量条件。 | 真实层间传递与增长 KV，逐层测量时延/功率/显存，保存 JSON；不定义在线插值。 |
| `strategy.py` | 请求负载、拓扑、运行状态和 Serving 方式。 | 计划计算流与传输流，不计算实际完成耗时。 |
| `execution/transmission.py` | 计划传输流、链路事件、带宽和运行状态。 | 传输进度、完成事件及通信资源与能耗更新。 |
| `execution/computation.py` | 计划计算流、Profile 和节点状态。 | 计算进度、完成事件及内存与能耗更新。 |
| `execution/simulator.py` | 策略与执行模块、实验条件及离线数据。 | 统一推进事件与运行状态，收集执行记录。 |
| `metrics.py` | 完成记录与资源使用数据。 | 按约定口径汇总性能指标。 |
| `run_simulation.py` | 在线实验配置与输入数据。 | 组织策略、执行仿真和指标汇总。 |

表中在线模块只描述逻辑输入输出，不预先定义尚未提供的字段或函数签名。

### 8.2 模块协作

离线链路生成：

```text
配置 → 轨道初始化 → [位置传播 → 拓扑计算 → 事件差分] → 持久化
```

离线硬件 Profile：

```text
模型 / GPU / 输入条件 → 配置校验与 GPU 占用检查
  → ShareGPT 分词 / 选择 → [真实层间执行 Prefill chunks / 连续 Decode steps]
  → 逐层 CUDA 计时 / NVML 稳态采样 → JSON 时间 / 功率 / 显存 / 原始样本
```

在线实验：

```text
run_simulation
  → 读取 Link Trace、Profile 和实验条件
  → simulator 管理时钟、事件与状态
      → strategy 生成计划
      → computation 与 transmission 并发推进
      → 完成 / 拓扑事件更新状态
  → metrics 汇总记录
```

离线数据生成与在线执行解耦；在线模块之间的共享状态由仿真调度统一管理，避免各自维护独立时钟。

## 9. 待明确的设计项

以下内容在原始需求中缺少具体定义，需在对应模块业务实现前补充：

- 计划计算流、传输流的字段和依赖关系；
- 原“六个模块输入与输出”详细表；
- Profile 的在线查询与插值规则，以及其他模型/设备的扩展实验配置；
- 请求负载与在线实验配置格式、调度时隙及事件优先级；
- 各 Serving 方式的具体调度算法；
- 路由、带宽共享及断链处理规则；
- 计算队列、Batch 与 KV Cache 管理规则；
- 功率、能耗、电池和 SOC 模型；
- 五个性能评估维度、指标公式及统计口径。

## 10. 建模参考资料

### 10.1 仓库参考代码

| 参考 | 可借鉴内容 | 与本项目的区别 |
| --- | --- | --- |
| `../starrynet_analyze/dove_24h_analysis/dove_24h_topo.py` | TLE 传播与链路集合差分。 | 全局互选 KNN，不是 +Grid；时间、编号和光速约定不同。 |
| `../starrynet_analyze/extract_dove_90.py` | 链路寿命汇总。 | 汇总 CSV 不代替完整建立、断开事件及传播时延。 |
| `../starrynet_analyze/starlink_90min_analysis/starlink_90min_topo.py` | 球形地球遮挡几何。 | 跨面固定槽位映射，不是逐采样最近邻。 |
| `../starrynet_analyze/starrynet/sn_observer.py` | 参数化卫星构造。 | 不继承其余无关功能。 |
| `../tools/profile_qwen25_32b_layers.py` | 逐层真实权重加载、CUDA events 和原始样本保存。 | 原实现固定 32B，Prefill 不跨块维护 KV；本项目读取模型维度并顺序测量 chunked Prefill。 |

参考代码中的编号、距离、遮挡裕量和有效光速不构成本项目的隐含默认值。

### 10.2 公开星座与相位资料

- [Iridium 官方公告](https://investor.iridium.com/2022-09-08-Iridium-Announces-Ninth-SpaceX-Launch)：66 颗运行卫星、6 面、每面 11 颗。
- [Brahe：Walker Constellations](https://docs.brahe.space/latest/learn/orbits/walker.html)：Iridium-like 780 km、86.4°、66/6/2 示例及 360° 相位公式。
- [MathWorks：walkerStar](https://www.mathworks.com/help/aerotbx/ug/satellitescenario.walkerstar.html)：采用 `180° * F / T`，与项目约定不同。
- [Feng 等，Mercator Projection Superposition，§5.2.3、表 8 与表 9](https://www.mdpi.com/2076-3417/15/19/10610)：550 km、53° 下的 1584/24/1 和 1584/72/1 建模配置。
- [ESA / GMV，LEOPARD Final Presentation，第 13 页](https://navisp.esa.int/uploads/files/project_documents/%5BNAVISP-EL1-019%20LEOPARD%5DFinal%20Presentation%20Slides.pdf)：OneWeb-like 648/18/9、87.4°、1200 km，但不能据此确认与项目的相位公式一致。
- [IETF：卫星互联网星座性能基准草案](https://datatracker.ietf.org/doc/html/draft-lai-bmwg-sic-benchmarking/)：Kuiper K2 的 610 km、42°、36 面、每面 36 星；不是运营方参数认证。
- [Satellite Selection for In-Band Coexistence of Dense LEO Networks，第 7 页 §IV、表 II](https://wireless.ee.ucla.edu/pdf/pub/satellite-selection.pdf)：Kuiper 壳层几何及 Walker Delta 建模，未给出可核实的 F。
- [ITU：卫星网络技术介绍](https://www.itu.int/hub/2020/10/satellite-innovation-space-technology-to-reach-the-unconnected/)：OneWeb 一代不采用星间链路。
- [IETF：A code to describe satellite constellations，版本 02](https://datatracker.ietf.org/doc/html/draft-piraux-space-constellation-code-02)：部分商业星座示例相位是推测值，不作为真实部署参数认证。
