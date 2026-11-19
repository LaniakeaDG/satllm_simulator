"""从 ShareGPT 选择首个用户消息；仅使用本地 tokenizer，不执行模型仓库代码。"""

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_requests(records, tokenizer, count, max_tokens, context_lengths):
    """Prefill 取前 count 条；每个 Decode 长度独立取前 count 条足够长的请求。"""
    prefill = []
    decode = {length: [] for length in context_lengths}
    for index, record in enumerate(records):
        message = next((turn.get("value") for turn in record.get("conversations", [])
                        if turn.get("from") in ("human", "user") and isinstance(turn.get("value"), str)
                        and turn["value"].strip()), None)
        if message is None:
            continue
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=True, add_generation_prompt=True,
        )
        if not tokens:
            continue
        request = {"dataset_index": index, "request_id": str(record.get("id", index)),
                   "original_tokens": len(tokens), "input_ids": tokens[:max_tokens]}
        if len(prefill) < count:
            prefill.append(request)
        for length, selected in decode.items():
            if len(tokens) >= length and len(selected) < count:
                selected.append({**request, "input_ids": tokens[:length]})
        if len(prefill) == count and all(len(selected) == count for selected in decode.values()):
            return prefill, decode
    available = {length: len(selected) for length, selected in decode.items()}
    raise ValueError(f"有效 ShareGPT 请求不足：Prefill={len(prefill)}/{count}，Decode={available}")


def request_metadata(request):
    tokens = request["input_ids"]
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    return {key: value for key, value in request.items() if key != "input_ids"} | {
        "input_tokens": len(tokens), "input_ids_sha256": digest,
    }


def load_workload(dataset_path, model_dir, settings):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    with Path(dataset_path).open(encoding="utf-8") as source:
        records = json.load(source)
    prefill, decode = select_requests(records, tokenizer, settings["sample_requests"],
                                      settings["max_input_tokens"], settings["decode_context_lengths"])
    metadata = {
        "path": str(dataset_path), "sha256": file_sha256(dataset_path), "dataset_records": len(records),
        "selection": "file order; first nonempty human/user message; local chat template with generation prompt; right token truncation",
        "prefill_requests": [request_metadata(request) for request in prefill],
        "decode_request_pools": {str(length): [request_metadata(request) for request in requests]
                                 for length, requests in decode.items()},
        "decode_batch_selection": "first batch_size requests from each context-length pool; one batch per size; no request replacement",
        "chat_template": tokenizer.chat_template,
    }
    return prefill, decode, metadata
