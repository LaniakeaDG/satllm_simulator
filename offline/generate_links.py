"""从 configs/orbit.yaml 生成离线链路事件和元数据。"""

from datetime import datetime, timezone
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sgp4.conveniences import jday_datetime

from .orbit import EARTH_RADIUS_KM, build_walker, load_tle, positions_at
from .topology import LIGHT_SPEED_KM_S, active_links

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_DIR / "configs" / "orbit.yaml"
EVENT_SCHEMA = pa.schema([
    pa.field("time_s", pa.float64(), nullable=False),
    pa.field("sat_i", pa.int64(), nullable=False),
    pa.field("sat_j", pa.int64(), nullable=False),
    pa.field("delay_ms", pa.float64()),
])
DEFAULTS = {
    "start_time_utc": "2026-01-01T00:00:00Z",
    "duration_s": 600,
    "sample_step_s": 1,
    "max_link_distance_km": 1500,
    "output_dir": "data/link_traces",
}


def generate_links(config):
    """运行一个配置案例，返回输出目录；已有结果不覆盖。"""
    config = DEFAULTS | config
    selected = config["selected"]
    if not isinstance(selected, str) or not selected or selected in (".", "..") or Path(selected).name != selected:
        raise ValueError("selected 必须是单个预设名称，不能包含路径。")
    for key in ("duration_s", "sample_step_s", "max_link_distance_km"):
        if not isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} 必须是正的有限数值。")
    orbit = dict(config["presets"][selected])
    start_time = orbit.get("start_time_utc", config["start_time_utc"])
    start = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("start_time_utc 必须显式带时区，例如 2026-01-01T00:00:00Z。")
    start = start.astimezone(timezone.utc)
    config["start_time_utc"] = start.isoformat().replace("+00:00", "Z")
    duration = float(config["duration_s"])
    step = float(config["sample_step_s"])
    max_distance = float(config["max_link_distance_km"])
    output_dir = PROJECT_DIR / config["output_dir"] / selected
    if any((output_dir / name).exists() for name in ("link_events.parquet", "manifest.yaml")):
        raise FileExistsError(f"结果已存在，不覆盖：{output_dir}；请修改 output_dir。")

    tle_metadata = {}
    if orbit["source"] == "walker":
        satellites, planes, metadata = build_walker(orbit, start)
    elif orbit["source"] == "tle":
        tle_file = (PROJECT_DIR / orbit["tle_file"]).resolve()
        orbit["tle_file"] = str(tle_file)
        satellites, planes, metadata = load_tle(orbit)
        tle_metadata = {"tle_file": str(tle_file), "tle_sha256": sha256(tle_file.read_bytes()).hexdigest()}
    else:
        raise ValueError("source 必须是 walker 或 tle。")

    output_dir.mkdir(parents=True, exist_ok=True)
    previous, pending = set(), []
    event_count = 0
    start_jd = jday_datetime(start)
    with TemporaryDirectory(prefix=".generating-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        with pq.ParquetWriter(temporary / "link_events.parquet", EVENT_SCHEMA) as writer:
            for index in range(ceil(duration / step)):
                time_s = index * step
                if time_s >= duration:
                    break
                positions = positions_at(satellites, start_jd, time_s)
                current = active_links(positions, planes, max_distance)
                changed = current.keys() ^ previous
                pending.extend({
                    "time_s": time_s, "sat_i": pair[0], "sat_j": pair[1],
                    "delay_ms": current.get(pair),
                } for pair in sorted(changed))
                event_count += len(changed)
                previous = set(current)
                if len(pending) >= 10000:
                    writer.write_table(pa.Table.from_pylist(pending, schema=EVENT_SCHEMA))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=EVENT_SCHEMA))

        manifest = {
            "config": config,
            "orbit": orbit,
            "start_time_utc": config["start_time_utc"],
            "duration_s": duration,
            "sample_step_s": step,
            "time_interval": "[0, duration_s)",
            "coordinate_frame": "TEME",
            "propagator": "SGP4",
            "gravity_model": "WGS84",
            "earth_radius_km": EARTH_RADIUS_KM,
            "propagation_speed_km_s": LIGHT_SPEED_KM_S,
            "max_link_distance_km": max_distance,
            "topology": {
                "intra_plane": "fixed_ring",
                "inter_plane": "mutual_nearest_available_each_sample",
                "plane_wrap": True,
                "walker_phase_formula": "360 * slot / (T/P) + 360 * F * plane / T",
                "initial_links_as_additions": True,
                "close_links_at_end": False,
                "delay_updates": "on_link_establishment_only",
            },
            "satellite_count": len(metadata),
            "event_count": event_count,
            "satellites": metadata,
            **tle_metadata,
        }
        (temporary / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8",
        )
        for name in ("link_events.parquet", "manifest.yaml"):
            (temporary / name).replace(output_dir / name)

    print(f"{selected}: {len(metadata)} 颗卫星，{event_count} 条链路事件，{duration:g} 秒")
    print(f"输出目录：{output_dir}")
    return output_dir


def main():
    with CONFIG_FILE.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    generate_links(config)


if __name__ == "__main__":
    main()
