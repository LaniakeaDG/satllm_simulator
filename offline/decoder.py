"""Qwen2 / Qwen2.5 / Qwen3 dense decoder 的独立逐层微基准实现。

沿用原逐层 Profile 的 RMSNorm、RoPE、GQA 和 SwiGLU 计算流程。
只支持标准全注意力与未量化 safetensors 权重，不运行模型仓库中的代码。
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def read_model_config(model_dir):
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in ("qwen2", "qwen3"):
        raise ValueError("仅支持 Qwen2 / Qwen2.5 / Qwen3 dense；不支持 MoE 或其他架构。")
    if config.get("quantization_config") or config.get("use_sliding_window"):
        raise ValueError("不支持量化权重或滑动窗口注意力。")
    scaling = config.get("rope_scaling") or {}
    if scaling and scaling.get("rope_type", scaling.get("type")) != "default":
        raise ValueError("不支持额外 RoPE scaling，不能将其静默当作标准 RoPE。")
    if config.get("hidden_act", "silu") != "silu" or config.get("partial_rotary_factor", 1) != 1:
        raise ValueError("仅支持 SiLU 和完整维度 RoPE。")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "max_position_embeddings"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"模型配置 {key} 必须为正整数。")
    if any(kind != "full_attention" for kind in config.get("layer_types", [])):
        raise ValueError("仅支持全部 decoder layer 使用标准全注意力。")
    if config.get("mlp_bias", False):
        raise ValueError("不支持带 bias 的 MLP。")
    expected_bias = config["model_type"] == "qwen2"
    if config.get("attention_bias", expected_bias) != expected_bias:
        raise ValueError("仅支持标准 Qwen2 的 QKV bias 或标准 Qwen3 的无 bias 注意力。")
    if config.get("rope_parameters"):
        raise ValueError("暂不支持 rope_parameters 格式，请使用含 rope_theta 的标准配置。")
    for key, default in (("rope_theta", 10000.0), ("rms_norm_eps", 1e-6)):
        value = config.get(key, default)
        if not isinstance(value, (float, int)) or not 0 < value < float("inf"):
            raise ValueError(f"模型配置 {key} 必须为正有限数。")
    heads = config["num_attention_heads"]
    default_head_dim = 128 if config["model_type"] == "qwen3" else config["hidden_size"] // heads
    head_dim = config.get("head_dim", default_head_dim)
    if (type(head_dim) is not int or head_dim <= 0 or head_dim % 2
            or heads % config["num_key_value_heads"]):
        raise ValueError("head_dim 必须为正偶数，attention heads 必须整除 KV heads。")
    if config["model_type"] == "qwen2" and "head_dim" not in config and config["hidden_size"] % heads:
        raise ValueError("未提供 head_dim 时，hidden_size 必须能被 attention heads 整除。")
    config["head_dim"] = head_dim
    return config


class RMSNorm(nn.Module):
    def __init__(self, size, eps, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden):
        normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(hidden.dtype)


def apply_rotary(tensor, cosine, sine):
    first, second = tensor.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return tensor * cosine + rotated * sine


class DecoderLayer(nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.theta = config.get("rope_theta", 10000.0)
        eps = config.get("rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hidden, eps, dtype)
        self.post_attention_layernorm = RMSNorm(hidden, eps, dtype)
        qkv_bias = config.get("attention_bias", config["model_type"] == "qwen2")
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        if config["model_type"] == "qwen3":
            self.q_norm = RMSNorm(self.head_dim, eps, dtype)
            self.k_norm = RMSNorm(self.head_dim, eps, dtype)

    def forward(self, hidden, cache, offset, attention_mask=None):
        residual = hidden
        normalized = self.input_layernorm(hidden)
        batch, length, _ = hidden.shape
        query = self.q_proj(normalized).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normalized).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if hasattr(self, "q_norm"):
            query, key = self.q_norm(query), self.k_norm(key)

        positions = torch.arange(offset, offset + length, device=hidden.device, dtype=torch.float32)
        frequencies = self.theta ** (-torch.arange(0, self.head_dim, 2, device=hidden.device,
                                                   dtype=torch.float32) / self.head_dim)
        angles = torch.outer(positions, frequencies)
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        cosine, sine = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
        query, key = apply_rotary(query, cosine, sine), apply_rotary(key, cosine, sine)

        end = offset + length
        cache[0][:, :, offset:end].copy_(key)
        cache[1][:, :, offset:end].copy_(value)
        attention = F.scaled_dot_product_attention(
            query, cache[0][:, :, :end], cache[1][:, :, :end], attn_mask=attention_mask,
            is_causal=offset == 0 and length > 1,
            enable_gqa=self.heads != self.kv_heads,
        )
        attention = attention.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        hidden = residual + self.o_proj(attention)
        normalized = self.post_attention_layernorm(hidden)
        return hidden + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))


def causal_mask(offset, length, device):
    """有前缀的多 token chunk 需要右对齐因果 mask，不能直接用 is_causal=True。"""
    if offset == 0 or length == 1:
        return None
    keys = torch.arange(offset + length, device=device)
    queries = offset + torch.arange(length, device=device)
    return keys[None, :] <= queries[:, None]


class WeightStore:
    """只读单文件或索引分片 safetensors，逐层读取真实参数。"""

    def __init__(self, model_dir):
        from safetensors import safe_open

        self.model_dir = Path(model_dir).resolve()
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            with safe_open(self.model_dir / "model.safetensors", framework="pt", device="cpu") as source:
                self.weight_map = {name: "model.safetensors" for name in source.keys()}
        for shard in set(self.weight_map.values()):
            relative = Path(shard)
            if relative.is_absolute() or ".." in relative.parts or not (self.model_dir / relative).is_file():
                raise ValueError(f"权重分片不存在或名称越出模型目录：{shard}")
            # 本地 Hugging Face snapshot 的合法分片可以链接到同一缓存的 blobs/。

    def read_tensor(self, name):
        from safetensors import safe_open

        with safe_open(self.model_dir / self.weight_map[name], framework="pt", device="cpu") as source:
            tensor = source.get_tensor(name)
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"仅支持未量化浮点权重：{name}，{tensor.dtype}")
        return tensor

    def load_layer(self, layer, layer_index):
        with torch.inference_mode():
            for name, parameter in layer.named_parameters():
                module = name.split(".")[0]
                prefix = ""
                if module in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                    prefix = "self_attn."
                elif module in ("gate_proj", "up_proj", "down_proj"):
                    prefix = "mlp."
                checkpoint_name = f"model.layers.{layer_index}.{prefix}{name}"
                tensor = self.read_tensor(checkpoint_name)
                if tensor.shape != parameter.shape or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise ValueError(f"权重形状或精度不支持：{checkpoint_name}，{tensor.shape}，{tensor.dtype}")
                parameter.copy_(tensor)
