"""NVML 板卡功率：对固定层输入稳态重复执行，时间加权采样，不推测缺失数据。"""

import time

import pynvml
import torch


class PowerMeter:
    def __init__(self, uuid):
        pynvml.nvmlInit()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            self.read_watts()
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def read_watts(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def close(self):
        pynvml.nvmlShutdown()

    def measure(self, forward, device, latency_ms, settings):
        # 每个提交批次约 20ms，避免无限排队；同步之后读取板卡遥测。
        block = min(256, max(1, int(settings["power_sample_interval_s"] * 1000 / max(latency_ms, 0.001))))
        with torch.inference_mode():
            for duration in (settings["power_settle_s"], settings["power_measure_s"]):
                samples = [(time.monotonic(), self.read_watts())]
                deadline = samples[0][0] + duration
                while samples[-1][0] < deadline:
                    for _ in range(block):
                        forward()
                    torch.cuda.synchronize(device)
                    samples.append((time.monotonic(), self.read_watts()))
        # 仅保留第二段；第一段用于摆脱此前其他层/CPU 间隙的功率历史。
        elapsed = samples[-1][0] - samples[0][0]
        energy = sum((right[0] - left[0]) * (left[1] + right[1]) / 2
                     for left, right in zip(samples, samples[1:]))
        return {"avg_power_w": energy / elapsed, "power_samples": len(samples),
                "power_window_s": elapsed, "power_replay_block": block}
