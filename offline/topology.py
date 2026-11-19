"""+Grid 拓扑：固定同面邻居、动态跨面互选最近邻。"""

import numpy as np

from .orbit import EARTH_RADIUS_KM

LIGHT_SPEED_KM_S = 299792.458


def _link_distances(first, second, max_distance_km):
    """支持数组广播；不可用链路的距离设为 inf。"""
    segment = second - first
    squared = np.sum(segment * segment, axis=-1)
    projection = -np.sum(first * segment, axis=-1) / np.maximum(squared, 1e-30)
    closest = first + np.clip(projection, 0, 1)[..., None] * segment
    clear = np.sum(closest * closest, axis=-1) >= EARTH_RADIUS_KM ** 2
    valid = clear & (squared > 0) & (squared <= max_distance_km ** 2)
    return np.where(valid, np.sqrt(squared), np.inf)


def active_links(positions, planes, max_distance_km):
    """返回 {(较小卫星 ID, 较大卫星 ID): 传播时延 ms}。"""
    links = {}
    for ids in planes:
        neighbors = np.roll(ids, -1)
        distances = _link_distances(positions[ids], positions[neighbors], max_distance_km)
        for first, second, distance in zip(ids, neighbors, distances):
            if np.isfinite(distance):
                pair = tuple(sorted((int(first), int(second))))
                links[pair] = float(distance / LIGHT_SPEED_KM_S * 1000)

    if len(planes) < 2:
        return links
    for left, right in zip(planes, planes[1:] + planes[:1]):
        distances = _link_distances(
            positions[left, None, :], positions[None, right, :], max_distance_km,
        )
        nearest_right = distances.argmin(axis=1)
        nearest_left = distances.argmin(axis=0)
        rows = np.arange(len(left))
        mutual = (nearest_left[nearest_right] == rows) & np.isfinite(distances[rows, nearest_right])
        for row in rows[mutual]:
            column = nearest_right[row]
            pair = tuple(sorted((int(left[row]), int(right[column]))))
            links[pair] = float(distances[row, column] / LIGHT_SPEED_KM_S * 1000)
    return links
