# 卫星 LLM Serving 仿真平台

总体规划与完整代码框架见 [DESIGN.md](DESIGN.md)，项目范围和实现原则见 [CLAUDE.md](CLAUDE.md)。

当前具备业务实现的是离线卫星轨道与链路事件生成，以及真实权重逐层 GPU Profile。Profile 输入与模型路径仍待配置，尚未生成正式 GPU 测量结果。策略、在线传输、节点计算、统一仿真调度和指标统计仅保留职责骨架，不提供模拟结果。在线入口在接口与模型明确前会显式报未实现错误。

## 运行

需要 Python 3.10 或更新版本。在项目目录下运行：

```bash
cd /home/zdg/OrbitServe/satllm_simulator
python -m pip install -r requirements.txt
python -m offline.generate_links
```

当前机器可使用 `/home/zdg/.conda/envs/orbitserve/bin/python`。

入口固定读取 [configs/orbit.yaml](configs/orbit.yaml)，无需命令行参数。配置中的相对路径均相对于项目目录。

## 时间设置

Walker 默认使用固定起点 `2026-01-01T00:00:00Z`，以该时刻初始化轨道相位，便于重复实验。目前只计算星间距离与球形地球遮挡，不涉及日照、地面站或真实观测对齐，因此 10–60 分钟仿真无需特意选择日期。更换绝对日期不应用于改变负载或链路实验条件；比较实验保持起点一致即可。

- 10 分钟：`duration_s: 600`。
- 60 分钟：`duration_s: 3600`。
- 默认 `sample_step_s: 1`，10 分钟的采样时刻为 0–599 秒，不包含 600 秒。
- 起点必须带时区，建议统一使用 `Z` 表示 UTC，不使用运行当天时间作为隐式起点。
- 预设可设置自己的 `start_time_utc`，优先于顶层时间；未设置时沿用顶层时间。manifest 记录实际使用的 UTC 起点。
- **TLE 模式不同**：起点应接近所选 TLE 的历元，不能为了方便沿用 Walker 示例起点。Dove 预设已设置 `2022-04-23T12:00:00Z`；修改 Dove 起点应修改该预设内部的字段。短仿真时长不能弥补轨道数据过旧带来的误差。

## 输入配置

`selected` 从同一 YAML 的 `presets` 中选择一个案例。默认配置是 Starlink-like Walker Delta `1584/24/1`，高度 550 km、倾角 53°。

可选预设共六种：`starlink_delta_24`、`starlink_delta_72`、`iridium_star`、`oneweb_star`、`kuiper_delta` 和 `dove_tle`，均已实际运行验证。

**相位假设**：OneWeb 使用项目公式下的 `F=9`（面间偏移 5°）；Kuiper 使用 `F=1`（约 0.277778°）。这两个值是显式的理想化仿真输入，不是已核实的商用在轨相位，也不宣称复现来源文稿中未确认的相位约定。配置的 `note` 会写入 manifest；需要其他相位时直接修改 `phasing`。来源与限制见设计文档第 3.7 节及第 10.2 节。

Walker Delta 的 RAAN 跨 360°，Star 跨 180°；**两者均采用 `360° * F / T` 的相邻面相位偏移**。高度以 6371 km 球形地球为基准，SGP4 使用 WGS84 引力模型，无额外阻力。

TLE 预设的字段结构为：

```yaml
source: tle
tle_file: 相对项目目录的TLE文件路径
planes:
  - [第一轨道面的NORAD_ID，按面内前后顺序排列]
  - [第二轨道面的NORAD_ID，按面内前后顺序排列]
```

上面只展示结构，需替换为实际文件路径和整数 NORAD ID；将该映射作为一个命名预设加入 `presets`，再修改 `selected`。支持无名称的两行 TLE 或带名称的三行 TLE，仅读取显式列出的卫星，不自动分面或筛选壳层。外层列表顺序定义轨道面相邻关系。

### Dove 实际输入

将 `selected` 改为 `dove_tle` 即可读取已有的 [Dove.tle](../starrynet_analyze/tle/Dove.tle)，不需要另外手写分面。配置已包含全部 197 颗卫星的 NORAD ID，对应 10 个不均匀轨道面，每面星数为 `[6, 57, 17, 65, 1, 24, 3, 7, 1, 16]`。

分面复现已有分析程序的聚类结果并固化为列表；面内按预设起点的纬度幅角排序，运行时不聚类。它是近似几何分组，+Grid 是假设网络，不代表 Dove 实际通信拓扑。分组依据见 [DESIGN.md](DESIGN.md) 第 3.8 节。更换 TLE 或大幅改变起点时应重新核对列表。

该文件所有 TLE 历元均在 **2022-04-23**，不是当前实时数据。Dove 起点独立设置为当天 `12:00:00Z`，无需修改其他 Walker 案例的起点。

## 链路规则

- 同面前后邻居固定，环形连接；不可用时不改连其他同面卫星。
- 跨面每个采样时刻，在距离与无遮挡条件均满足的候选中寻找最近邻；双方互选才连边。
- 首尾轨道面也相邻；所有边均为无向边，每颗卫星最多四个不同邻居。
- 默认最大距离 1500 km，地球遮挡半径 6371 km，不加大气裕量；传播速度为 299792.458 km/s。
- 最近邻距离相同时按轨道面内列表顺序选择。重合位置不建立零长度链路。
- 不建模带宽、链路切换开销或真实星座的终端约束。

## 输出

默认生成在 `data/link_traces/starlink_delta_24/`：

| 文件 | 内容 |
| --- | --- |
| `link_events.parquet` | 按时间和端点排序的链路建立、断开事件。 |
| `manifest.yaml` | 完整配置、卫星映射、轨道面与槽位、起始时间、采样步长、物理常数；TLE 模式附源文件摘要。 |

事件表仅有四列：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `time_s` | float64 | 相对起点的秒数。 |
| `sat_i` | int64 | 0-based 卫星 ID。 |
| `sat_j` | int64 | 另一端点，始终满足 `sat_i < sat_j`。 |
| `delay_ms` | 可空 float64 | 数值表示建立及当时的传播时延，NULL 表示断开。 |

首帧记录所有可用链路的建立事件；时延单独变化不输出事件，存续期间保持建立时的时延；结束时不人为断开链路。数据只在 manifest 标记的 `[0, duration_s)` 区间内有效。NULL 是 Parquet 真正的空值，不是字符串或 NaN。

已有结果不会覆盖。再次运行时修改 `output_dir`，例如将 60 分钟案例保存到 `data/link_traces_60min`。生成过程先写临时文件，成功后再发布这两个输出文件。

### 已生成与验证的案例

以下结果均为 600 秒、1 秒采样、1500 km 上限，已保存在 `data/link_traces/<预设名>/`；更改 `selected` 后重新运行时，请同时改用新的 `output_dir`。

| 预设 | 卫星数 | 事件总数 | 其中断开数 |
| --- | --- | --- | --- |
| `starlink_delta_24` | 1584 | 3168 | 0 |
| `starlink_delta_72` | 1584 | 1584 | 0 |
| `iridium_star` | 66 | 11 | 5 |
| `oneweb_star` | 648 | 1732 | 218 |
| `kuiper_delta` | 1296 | 2592 | 0 |
| `dove_tle` | 197 | 253 | 44 |

新增的五个案例均已逐秒回放核对，并另在临时目录完成 3600 秒入口运行：

| 预设 | 60 分钟事件总数 | 其中断开数 |
| --- | --- | --- |
| `starlink_delta_24` | 3168 | 0 |
| `starlink_delta_72` | 1584 | 0 |
| `iridium_star` | 54 | 26 |
| `oneweb_star` | 4108 | 1406 |
| `kuiper_delta` | 2592 | 0 |
| `dove_tle` | 600 | 217 |

Dove 的长时案例也已逐秒对照拓扑；其他长时案例检查完整运行与事件增删顺序。长时输出由临时目录自动清除，不保留。另以 1000 km 上限运行默认构型 240 秒，产生 2602 条事件，其中 132 条为真实 NULL 断开事件，逐秒回放与重新计算一致。

其他检查覆盖两行/三行 TLE 的 197 星位置一致性、显式 NORAD 映射、预设起点优先级与时区归一化、空事件表、非整数采样间隔和已有输出不覆盖。以上均为离线链路模块的验证，不代表在线仿真模块已经实现。

**1500 km 限制的影响**：Starlink 72 面和 Iridium 的固定同面邻居均超出上限，不能建立同面链路。Iridium 初始有 62 颗孤立卫星；Dove 初始有 15 颗孤立卫星。没有为这些案例自动放宽距离或补连邻居，不能默认假设全网连通。

## GPU 逐层 Profile

参考根项目 [profile_qwen25_32b_layers.py](../tools/profile_qwen25_32b_layers.py)，按层加载本地真实权重，不需要把完整 7B / 14B / 32B 模型同时放入 GPU。模型维度读取各自 `config.json`，支持标准 dense Qwen2 / Qwen2.5，以及带 Q/K Norm 的 dense Qwen3。MoE、量化、滑动窗口、额外 RoPE scaling 和 `rope_parameters` 格式不支持，遇到会明确拒绝。

### 配置与启动

Profile 依赖单独安装，不影响链路模块：

```bash
cd /home/zdg/OrbitServe/satllm_simulator
python -m pip install -r requirements-profile.txt
# 先填写 configs/profile.yaml 中的待定项，再执行：
python -m offline.profile
```

PyTorch 必须为与本机驱动匹配的 CUDA 版本；不依赖 Transformers，不自动下载模型。权重支持 `model.safetensors` 或 `model.safetensors.index.json` 加分片文件。这里的 Qwen 7B 指受支持架构中的 Qwen2 / Qwen2.5-7B，不包括初代 `model_type: qwen`。

固定配置为 [configs/profile.yaml](configs/profile.yaml)，需要明确：

1. `models`：把需要测量的模型名称对应值填为本地目录；`null` 的模型跳过。预留 7B、14B、32B 名称不代表这些权重已存在，也不是最终模型集合。
2. `input_mode`：目前仅实现可显式选择的 `synthetic_hidden_states`，沿用随机层输入微基准；真实文本输入方案仍待确定，不能直接把文本路径填入此字段。
3. `prefill_total_tokens`：完整输入长度，必须显式填写。`prefill_batch_size` 默认 1，可单独配置。
4. `decode_cache_lengths`：Decode 每一步之前的历史 KV 长度列表，必须显式填写，可扫描多个上下文长度。

上述待定项默认留空，因此**直接执行会报“输入数据待定”，不会启动 GPU 测量或生成正式结果**。默认 BF16、预热 5 次、重复 30 次，设备为 `cuda:0`。目标 GPU 有其他计算进程或无法可靠检查占用时默认拒绝运行，不会停止其他进程。

### 测量语义

| 阶段 | 固定扫描条件 | KV 与计时方式 |
| --- | --- | --- |
| Chunked Prefill | chunk size 为 **64、128、256、512 token**。 | 每种大小从空 KV 开始，顺序处理完整输入；每块记录历史长度和实际 token 数，末尾不足块不补成完整块。 |
| Decode | Batch Size 为 **1、32、64、128**，每请求每步 **1 token**。 | 对指定历史 KV 长度测量单步；历史 KV 为随机张量，不是整模型 Prefill 的输出。 |

Prefill 四种 chunk size 使用同一层的相同随机输入。层内使用真实权重，历史 KV 来自该层之前的 chunk，但不同层的随机输入不是完整模型逐层传递的激活。重复测量覆盖当前 KV 槽，不增长同一 case 的上下文。

使用预分配 KV、PyTorch eager 和 SDPA 自动后端，以 CUDA events 记录时间。计时包含层内算子、RoPE 和 KV 写入；不包含权重加载、输入/KV 分配初始化、mask 构造、embedding、最终 norm、LM head、采样或跨层传递。因此不是 vLLM 等 Serving 引擎性能，也不是端到端推理延迟；**不测功率，不输出推测功率或能耗**。

### 输出与验证边界

输出到 `data/profiles/<模型名称>/`，已有目录一律不覆盖；需要重跑时使用新的 `output_dir`。

- `layers.csv`：逐层、逐块 / Decode case 条件，样本数，均值/中位数/标准差/最小/最大耗时（ms），常驻基线/峰值/峰值增量显存（MiB）。
- `samples.jsonl`：每个 case 的原始耗时样本。
- `metadata.json`：模型与运行配置、GPU 环境、测量范围、输入和 KV 假设、完成状态及失败原因。

显存是 PyTorch allocated memory，包含单层权重、完整层输入、预分配 KV 容量、mask 和临时张量，不是整模型占用或 `nvidia-smi` 数字。只把 `metadata.json` 中 `status: complete` 的运行当作完整结果；失败、进程强制退出可能保留部分文件，不应直接作为仿真输入。

当前验证覆盖 Python 语法、CPU 小尺寸 Qwen2 / Qwen3 整段与分块/逐 token 结果一致、因果隔离、历史 KV 不被重复采样改动、四种 chunk size 的尾块及 Decode batch/context 扫描、未配置入口拒绝和已有结果拒绝。CSV / 原始样本 / 完成与失败元数据的一致性，以及目标 GPU UUID 选择与占用检查失败时拒绝执行，通过临时替代计时器、CUDA 和依赖检查验证，仅证明流程而非性能。尚未验证真实 safetensors 权重加载和真实 GPU CUDA-event 测量；本次未安装 safetensors、未下载模型、未生成正式 Profile。

## 代码与实现状态

| 文件 | 实现状态 |
| --- | --- |
| [offline/orbit.py](offline/orbit.py) | Walker / TLE 初始化与 SGP4 传播。 |
| [offline/topology.py](offline/topology.py) | +Grid、链路可用性与传播时延。 |
| [offline/generate_links.py](offline/generate_links.py) | 配置、采样、事件差分与持久化。 |
| [offline/decoder.py](offline/decoder.py) | Qwen2 / Qwen2.5 / Qwen3 dense 单层算子、因果 mask 与真实权重加载。 |
| [offline/profile.py](offline/profile.py) | 多模型逐层 Chunked Prefill / Decode GPU 测量；输入与本地模型路径待配置。 |
| [strategy.py](strategy.py) | 骨架：等待任务字段及策略规则。 |
| [execution/simulator.py](execution/simulator.py) | 骨架：等待统一事件与状态接口。 |
| [execution/transmission.py](execution/transmission.py) | 骨架：等待路由、共享带宽及能耗规则。 |
| [execution/computation.py](execution/computation.py) | 骨架：等待队列、Batch、KV Cache 及能耗规则。 |
| [metrics.py](metrics.py) | 骨架：等待五个评估维度及指标公式。 |
| [run_simulation.py](run_simulation.py) | 在线入口骨架，直接执行会显式抛出 `NotImplementedError`。 |

离线链路生成只保存当前帧位置和前一帧链路集合，事件分批写入，不缓存全部时刻位置，也不保存完整拓扑快照。GPU Profile 依赖与配置单独维护；尚未实现的在线模块不引入占位仿真结果。
