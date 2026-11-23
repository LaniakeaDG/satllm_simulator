# 卫星 LLM Serving 仿真平台设计

本文描述平台的总体架构、模块职责、数据约定和整体代码框架，不记录开发过程、实现进度或运行验证结果。项目范围与实现原则见 [CLAUDE.md](CLAUDE.md)，运行说明见 [README.md](README.md)。

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

策略层根据请求负载、动态网络拓扑和预设的 LLM Serving 方式，确定请求的部署与执行方案。

| 执行方式 | 定义 |
| --- | --- |
| 本地推理（Local / Co-located） | Prefill、Decode 在同一节点执行。 |
| PD 分离（PD Disaggregation） | Prefill、Decode 分配到不同节点，通过传输 KV Cache 衔接。 |
| 流水线并行（PP） | 模型划分为多个阶段，部署在不同节点并传输中间激活。 |
| 张量并行（TP） | 模型算子分布到多个节点，通过通信完成协同计算。 |
| 混合并行（Hybrid） | 组合不同并行方式。 |

策略负责节点选择、模型部署、任务分配、Batch 组织和执行顺序，输出每个时隙计划执行的计算流与传输流（Slot-level Data Flows），不直接计算实际执行耗时。

计算流与传输流的字段、具体策略算法及不同并行方式的通信依赖需另行明确。

## 6. 执行仿真层

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
