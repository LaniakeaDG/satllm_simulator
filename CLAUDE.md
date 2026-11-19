# 项目本地约束

## 项目范围

本目录为卫星 LLM Serving 仿真平台。整体规划与模块框架以 [DESIGN.md](DESIGN.md) 为准，使用方法与运行状态记录在 [README.md](README.md)。

平台包含离线链路生成、硬件性能 Profile、Serving 策略、统一离散事件执行仿真和性能评估。只实现设计明确要求的功能，不提前加入通用扩展框架或无关工具。

## 实现原则

- 代码尽可能简洁明了，减少嵌套、条件分支和不必要的抽象。
- 优先使用少量函数、普通字典和 NumPy 数组，不引入无必要的基类、工厂、插件注册或配置框架。
- 未明确的建模决策、数据字段和统计口径由用户确认，不以实现默认值代替设计决策。
- 未确定接口的模块只搭建职责骨架，不编造函数签名或可产生伪结果的占位算法；未完成的入口应明确报错。
- 不将运行成功等同于模型已获物理或来源验证。仿真假设与已核实参数须区分，并写入配置和元数据。
- 离线链路生成与在线执行解耦；在线传输、计算和资源状态共享同一仿真时钟，未完成任务可跨时隙延续。
- 已有代码、TLE 和实验数据保持不变，除非任务明确要求修改；生成输出不覆盖已有结果。
- 默认不添加测试框架或测试文件；任务需要验证时，优先使用临时目录和临时检查脚本，报告实际验证范围。

## 代码与运行约定

- Python 3.10 或更新版本，链路依赖记录在 [requirements.txt](requirements.txt)，GPU Profile 独立依赖记录在 [requirements-profile.txt](requirements-profile.txt)。
- 脚本默认不提供 CLI 参数，使用固定配置入口；配置中的相对路径以项目根目录为基准。
- 离线链路入口为 `python -m offline.generate_links`，固定读取 [configs/orbit.yaml](configs/orbit.yaml)。
- GPU Profile 入口为 `python -m offline.profile`，固定读取 [configs/profile.yaml](configs/profile.yaml)；输入方案和模型目录未明确前保留 null，不自动下载模型或生成正式 Profile。
- 在线实验配置、Profile 在线查询/插值与功率接口在契约明确后再定义，不提前创建含猜测字段的配置文件。
- 开发机器可使用 `/home/zdg/.conda/envs/orbitserve/bin/python`。

## 文档分工

- DESIGN.md：只保留规划设计、模型假设、数据约定、模块边界、整体代码框架与参考依据。
- README.md：运行方法、实现状态、使用限制与验证结果。
- 本文件：项目范围、实现原则与本地开发约束，不放进设计正文。

不得向 DESIGN.md 追加开发过程、已完成清单、临时排查记录、验证计数或工具执行记录。
