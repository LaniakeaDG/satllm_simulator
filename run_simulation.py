"""在线仿真入口，固定读取 execution/inputs.json。"""

import json
from pathlib import Path

from execution.simulator import _file, _number, simulate


INPUT_PATH = Path(__file__).resolve().parent / "execution" / "inputs.json"


def main():
    config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    base = INPUT_PATH.parent
    # 先检查全部输入，避免在未指定请求 trace 时加载大模型或链路数据。
    request_path = _file(config.get("request_trace_path"), "request_trace_path", base)
    link_path = _file(config.get("link_events_path"), "link_events_path", base)
    strategy_path = _file(config.get("strategy_path"), "strategy_path", base)
    profile_root = Path(config["profile_root"])
    if not profile_root.is_absolute():
        profile_root = base / profile_root
    if not (profile_root / "manifest.json").is_file():
        raise ValueError("profile_root 必须是含 manifest.json 的 Vidur 目录")
    horizon = config.get("end_time_ms")
    if horizon is None:
        import yaml
        manifest_path = link_path.with_name("manifest.yaml")
        if not manifest_path.is_file():
            raise ValueError("无链路 manifest.yaml，请显式指定 end_time_ms")
        horizon = _number(yaml.safe_load(manifest_path.read_text())["duration_s"], "duration_s") * 1000
    import pyarrow.parquet as pq
    events = pq.read_table(link_path, columns=["time_s", "sat_i", "sat_j", "delay_ms"]).to_pylist()
    result = simulate(
        json.loads(strategy_path.read_text()), json.loads(request_path.read_text()), events,
        profile_root=profile_root, bandwidth_bps=config["bandwidth_bps"], end_time_ms=horizon,
    )
    output = Path(config["output_path"])
    if not output.is_absolute():
        output = base / output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"完成 {sum(t['status'] == 'completed' for t in result['tasks'])}/{len(result['tasks'])} 个任务：{output}")


if __name__ == "__main__":
    main()
