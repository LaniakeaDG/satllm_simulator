# Vidur 原数据与计算预测模型

## 1. A100 / Llama-2-7B 原数据实际测量了什么

来源为 `data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/{mlp,attention}.csv`，不是本机新 profiling。`a100` 是原目录标记；CSV 未记录显存容量、GPU 具体型号/时钟、驱动等，不能据此认定为 A100 80GB。

### 固定参数与精度

- 两文件实际 TP 均为 **1、2、4、8**；hidden=4096、Q heads=32、KV heads=32，head_dim=128。MLP 还记录 FFN=11008、gated MLP=True、vocab_size=32768。已按原训练 loader 的全部架构条件过滤：本文件 1044 行均匹配，没有混入其他架构参与训练。
- Attention 记录 block_size=16、max_model_len=4096，backend 唯一值为 **`AttentionBackend.FLASH_ATTENTION`**。注意源码入口默认 FLASHINFER，不等于这份 CSV 的实际 backend。
- 两份 CSV **都没有 dtype 列**。复制源码 `profiling/mlp/mlp_wrapper.py` 将模型转为 `torch.float16`，`profiling/attention/main.py` 设置 `dtype=torch.float16`，即代码路径为 FP16，但 CSV 没有逐行独立记录/认证精度。
- 下述 context 指 `kv_cache_size`，即每序列已有 KV token 数；prefill 总长度为 context+chunk，decode 每次新增一个 token。batch 为序列数，不是 token 数。

### 精确参数集合与组合覆盖

区间写法 `a..b / s` 表示含端点、步长为 s 的离散集合，不是所有整数。

| 测量 | 每个 TP 的精确取值集合 | 实际组合 |
|---|---|---|
| MLP/投影等计算 | num_tokens = `{1,2,4}` ∪ `8..1024 / 8` ∪ `1040..2048 / 16` ∪ `2080..4096 / 32`，共259值 | 每个TP259形状、261行；2048和4096各有两次不同计时观测，没有batch/context轴 |
| Prefill | batch=1；chunk = `64..128 / 16` ∪ `160..1024 / 32` ∪ `1088..4096 / 64`（81值）；context = `0..2048 / 32` ∪ `2112..4032 / 64`（80值） | **不是完整笛卡尔积**：81×80=6480候选，仅保留 **chunk+context≤4096** 的4206形状；2274种不满足长度约束 |
| Decode | chunk=0；batch=`1..128 / 1`；context = `32..2048 / 32` ∪ `2112..4032 / 64`（79值） | 经逐TP集合比较，**确为完整128×79=10112形状**；本文件不存在额外batch×context缺口，不能因源码有内存筛选就推断这里不完整 |

每个 TP 的集合一致。原始行不等于唯一形状：同形状可能有不同计时观测，原 loader 仅去掉**所有列完全相同**的行，不对形状求均值或任选一行。

| TP | MLP 原行 / 全行去重 / 形状 | Prefill 原行 / 全行去重 / 形状 | Decode 原行 / 全行去重 / 形状 | Attention 全行去重合计 |
|---|---|---|---|---|
| 1 | 261 / 261 / 259 | 4418 / 4410 / 4206 | 10240 / 10240 / 10112 | 14650 |
| 2 | 261 / 261 / 259 | 4418 / 4403 / 4206 | 10240 / 10236 / 10112 | 14639 |
| 4 | 261 / 261 / 259 | 4418 / 4378 / 4206 | 10240 / 10234 / 10112 | 14612 |
| 8 | 261 / 261 / 259 | 4418 / 4355 / 4206 | 10240 / 10234 / 10112 | 14589 |

MLP 总计1044原行、1044全行去重记录、1036个TP内唯一形状，Attention 总计58632原行、58490全行去重记录、57272个TP内唯一形状。准确逐组合清单由训练脚本同步生成：

- [mlp_measurement_combinations.csv](mlp_measurement_combinations.csv)：1036行，每个TP×num_tokens一行。
- [attention_measurement_combinations.csv](attention_measurement_combinations.csv)：57272行，每个TP×batch×chunk×KV×block一行。

两份清单均有 `raw_rows` 与 `deduplicated_rows`，区分原始重复计数和训练loader全行去重后的观测数；清单行数表示形状数，不是训练样本数。

### 真实指标与缺失指标

所有实际时延列为 `time_stats.<组件>.{min,max,mean,median,std}`，单位 **ms**（源码 CUDA timer / tracer 将计时转换为毫秒）。它们是重复微基准的统计量，不是逐请求端到端延迟或吞吐量。

- **MLP CSV：10组、50个时延统计列**：`emb`、`input_layernorm`、`attn_pre_proj`、`attn_rope`、`attn_post_proj`、`post_attention_layernorm`、`mlp_up_proj`、`mlp_act`、`mlp_down_proj`、`add`。9个组件用于本次训练，`emb` 有原始测量但未建模型。
- **Attention CSV：4组、20个时延统计列**：`attn_input_reshape`、`attn_decode`、`attn_output_reshape`、`attn_prefill`。本次只用对应阶段的 prefill/decode median；reshape 有原始指标但未单独建模型。阶段不适用列可能为空，不能解释为全部阶段均测得全部指标。
- **没有功率、能耗、显存占用指标**；block_size/KV/context 是输入参数，不是显存测量。
- **`attn_kv_cache_save` 所有时延列均缺失，未测量**。原版 loader 为其 median 补零；保留的4个兼容预测器因此是全零目标，不是“实测零开销”。当前 norm/add 的目标列均实际存在且非零，不是缺测补零。

## 2. 每个预测模型对应的配置

仅保留 **TP=1/2/4/8 × 每TP12个组件 = 48个模型**，位于 `models/compute/a100/Llama-2-7b-hf/tp*/`。固定结构为上述 hidden/FFN/heads/gating/vocab；TP模型已经对应其分片配置，不是把TP1输出简单除以TP。Llama-2-7B 原配置有32层，但这里预测的是**单层组件微基准耗时**，不是32层全模型时延。

全部输出为 `time_stats.<组件>.median` 的 ms，输入列顺序如下。训练行数按 TP1/2/4/8 顺序列出，范围只描述边界，准确组合以第1节和CSV为准。

| 组件（每TP各一个） | 有序输入 | 训练数据范围 / 行数（TP1/2/4/8） |
|---|---|---|
| `attn_pre_proj` | `[num_tokens]` | MLP 259值；各261行 |
| `attn_post_proj` | `[num_tokens]` | 同上 |
| `mlp_up_proj` | `[num_tokens]` | 同上 |
| `mlp_down_proj` | `[num_tokens]` | 同上 |
| `mlp_act` | `[num_tokens]` | 同上 |
| `input_layernorm` | `[num_tokens]` | 同上 |
| `post_attention_layernorm` | `[num_tokens]` | 同上 |
| `attn_rope` | `[num_tokens]` | 同上 |
| `add` | `[num_tokens]` | 同上 |
| `attn_kv_cache_save` | `[num_tokens]` = max(chunk,batch) | 全attention；14650 / 14639 / 14612 / 14589行；**补零目标，非实测指标** |
| `attn_prefill` | `[kv_cache_size, prefill_chunk_size_squared]` | 4206种受约束形状；4410 / 4403 / 4378 / 4355行 |
| `attn_decode` | `[batch_size, kv_cache_size]` | 10112种完整网格形状；10240 / 10236 / 10234 / 10234行 |

`num_tokens`、KV、chunk单位为tokens，batch为requests，chunk²为token²。KV保存输入1..4096但并非每个整数都有样本；prefill平方输入4096..16777216；decode batch1..128、KV32..4032。不要拿这些上下限构造任意未测组合。

### 训练设置、缓存与验证

- 使用已有 sklearn 1.7.2 的 `RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_split=2, random_state=42, n_jobs=4)`，其余默认值；固定为原RF网格首候选。
- 原训练做 `n_estimators=[250,500,750] × max_depth=[8,16,32] × min_samples_split=[2,5,10]` 共27组、10折GridSearchCV；这里**不做CV、holdout或调参**，不宣称复现其最优选择。
- 原训练入口有ray/fasteners导入阻碍；脚本以AST执行未改动的原计算/attention加载、过滤与派生方法。保留形状重复的不同观测；拒绝剩余非有限输入/目标及负目标，本次剔除0行。
- 缓存身份使用源CSV、特征/目标/过滤条件、RF参数、sklearn版本、实际训练数据及原loader方法哈希，不因exporter文案或网络删除而全量重训。沿用已生成48个计算模型的文件名；从旧manifest迁移后记录新身份，重复运行复用同一批文件。
- 已重载全部48个现成模型并对 **126376条组件训练输入** 验证预测有限且非负，48个cache hit，未重训。此验证是序列化/输入健全性检查，**不是泛化精度或物理验证**。随机森林不适合域外外推。
- `manifest.json` 逐模型记录路径、配置、输入顺序/单位、目标、训练行数、源数据哈希与重载验证。原始CSV及完整复制源码保持不变；原网络CSV仅作来源存档，**不训练、不导出通信预测器**，已删除先前20个通信joblib及空的`models/network`目录。CPU overhead原始数据缺失，未虚构补齐。

固定CPU入口，无CLI、无下载或新增依赖，不执行GPU profiling：

```bash
/home/zdg/.conda/envs/orbitserve/bin/python /home/zdg/OrbitServe/satllm_simulator/profilling_vidur/original_vidur/train_models.py
```

只加载可信本地joblib，使用manifest记录的依赖版本。组件耗时不能直接无条件相加为全模型端到端时延；层数、TP collective、流水线、调度及CPU开销等仍需另行建模，本目录不提供在线仿真接口。
