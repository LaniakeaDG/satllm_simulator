"""Walker / TLE 轨道初始化，以及 SGP4 的 TEME 位置传播。"""

from pathlib import Path

import numpy as np
from sgp4.api import SGP4_ERRORS, Satrec, SatrecArray, WGS84
from sgp4.conveniences import jday_datetime
from sgp4.earth_gravity import wgs84

EARTH_RADIUS_KM = 6371.0


def build_walker(config, start_time):
    """返回批量传播器、按轨道面分组的 ID，以及卫星元数据。"""
    total = config["total_satellites"]
    plane_count = config["planes"]
    phasing = config["phasing"]
    if phasing is None:
        raise ValueError("该预设的相位因子尚未确认，请先补充 phasing。")
    if not all(isinstance(value, int) for value in (total, plane_count, phasing)):
        raise ValueError("total_satellites、planes、phasing 必须是整数。")
    if total < 1 or plane_count < 1 or total % plane_count:
        raise ValueError("卫星数和轨道面数必须为正，且总卫星数能被轨道面数整除。")
    if not 0 <= phasing < plane_count:
        raise ValueError("相位因子必须满足 0 <= phasing < planes。")
    if config["pattern"] not in ("delta", "star"):
        raise ValueError("Walker pattern 必须是 delta 或 star。")
    altitude = float(config["altitude_km"])
    inclination = float(config["inclination_deg"])
    if not np.isfinite(altitude) or altitude <= 0 or not 0 <= inclination <= 180:
        raise ValueError("高度必须为正数，倾角必须在 [0, 180] 度内。")

    jd, fraction = jday_datetime(start_time)
    epoch = (jd - 2433281.5) + fraction  # SGP4: 自 1949-12-31 起的天数
    mean_motion = np.sqrt(wgs84.mu / (EARTH_RADIUS_KM + altitude) ** 3) * 60
    per_plane = total // plane_count
    raan_span = 360 if config["pattern"] == "delta" else 180
    satellites, planes, metadata = [], [], []
    for plane in range(plane_count):
        ids = np.arange(plane * per_plane, (plane + 1) * per_plane)
        planes.append(ids)
        for slot, sat_id in enumerate(ids):
            anomaly = (360 * slot / per_plane + 360 * phasing * plane / total) % 360
            satellite = Satrec()
            satellite.sgp4init(
                WGS84, "i", int(sat_id) + 1, epoch,
                0.0, 0.0, 0.0,  # bstar、ndot、nddot：不额外建模阻力
                0.0, 0.0, np.deg2rad(inclination),
                np.deg2rad(anomaly), mean_motion,
                np.deg2rad(raan_span * plane / plane_count),
            )
            satellites.append(satellite)
            metadata.append({
                "sat_id": int(sat_id), "name": f"P{plane}-S{slot}",
                "plane": plane, "slot": slot,
            })
    return SatrecArray(satellites), planes, metadata


def load_tle(config):
    """读取两行或带名称的三行 TLE；仅使用配置显式列出的卫星。"""
    lines = iter(line.strip() for line in Path(config["tle_file"]).read_text().splitlines()
                 if line.strip())
    records = {}
    for first in lines:
        name = None
        if not first.startswith("1 "):
            name, first = first.removeprefix("0 ").strip(), next(lines, "")
        second = next(lines, "")
        if not first.startswith("1 ") or not second.startswith("2 "):
            raise ValueError("TLE 必须由连续的第 1、2 行组成，可在前面附加名称。")
        if first[2:7] != second[2:7]:
            raise ValueError("同一组 TLE 的两行 NORAD ID 不一致。")
        satellite = Satrec.twoline2rv(first, second, WGS84)
        norad_id = satellite.satnum
        if norad_id in records:
            raise ValueError(f"TLE 中重复的 NORAD ID：{norad_id}")
        records[norad_id] = (satellite, name or str(norad_id))

    groups = config["planes"]
    if not groups or any(not group for group in groups):
        raise ValueError("TLE planes 必须包含非空的 NORAD ID 列表。")
    norad_ids = [norad_id for group in groups for norad_id in group]
    if len(norad_ids) != len(set(norad_ids)):
        raise ValueError("同一颗 TLE 卫星不能重复出现在轨道面列表中。")
    missing = set(norad_ids) - records.keys()
    if missing:
        raise ValueError(f"配置中的 NORAD ID 不在 TLE 中：{sorted(missing)}")

    satellites, planes, metadata = [], [], []
    for plane, group in enumerate(groups):
        ids = []
        for slot, norad_id in enumerate(group):
            satellite, name = records[norad_id]
            sat_id = len(satellites)
            satellites.append(satellite)
            ids.append(sat_id)
            metadata.append({
                "sat_id": sat_id, "name": name, "norad_id": norad_id,
                "plane": plane, "slot": slot,
            })
        planes.append(np.asarray(ids, dtype=int))
    return SatrecArray(satellites), planes, metadata


def positions_at(satellites, start_jd, time_s):
    """返回相对起点 time_s 秒时的 (卫星数, 3) TEME 位置，单位 km。"""
    days, fraction = divmod(start_jd[1] + time_s / 86400, 1)
    errors, positions, _ = satellites.sgp4(
        np.array([start_jd[0] + days]), np.array([fraction]),
    )
    failed = np.flatnonzero(errors[:, 0])
    if len(failed):
        sat_id = int(failed[0])
        reason = SGP4_ERRORS[int(errors[sat_id, 0])]
        raise ValueError(f"t={time_s:g}s，卫星 {sat_id} 传播失败：{reason}")
    return positions[:, 0, :]
