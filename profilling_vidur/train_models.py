"""CPU-only export from bundled Vidur profiles; fixed paths, no profiling."""
import ast
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
# Reuse already-present packages; never install or download dependencies.
sys.path.insert(0, str(ROOT / "dependencies"))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

DATA = ROOT / "data/profiling"
MODELS = ROOT / "models"
PARAMS = dict(n_estimators=250, max_depth=8, min_samples_split=2,
              random_state=42, n_jobs=4)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
METHODS = ["_read_input_file", "_load_compute_df", "_load_attention_df",
           "_get_compute_df_with_derived_features",
           "_get_attention_df_with_derived_features"]


def original_loaders():
    """Execute unchanged original loader/feature methods without GPU/ray imports."""
    path = ROOT / "source/vidur/execution_time_predictor/sklearn_execution_time_predictor.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {"pd": pd, "logger": logging.getLogger("original_vidur")}
    exec(compile(module, str(path), "exec"), scope)
    loader = scope[cls.name]()
    loader._model_config = SimpleNamespace(num_q_heads=32, num_kv_heads=32,
        embedding_dim=4096, mlp_hidden_dim=11008, use_gated_mlp=True, vocab_size=32768)
    loader._block_size = 16
    return loader


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_measurement_combinations(source, filters, shape_columns, output):
    """Count raw rows separately from full-row deduplication and unique shapes."""
    raw = pd.read_csv(source)
    for column, value in filters.items():
        if column in raw.columns:
            raw = raw[raw[column] == value]
    counts = raw.groupby(shape_columns, dropna=False).size().rename("raw_rows")
    unique = raw.drop_duplicates().groupby(shape_columns, dropna=False).size().rename("deduplicated_rows")
    counts.to_frame().join(unique).reset_index().to_csv(output, index=False)


def main():
    MODELS.mkdir(exist_ok=True)
    sources = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in sorted(DATA.rglob("*.csv"))]
    hashes = {s["path"]: s["sha256"] for s in sources}
    entries = []
    manifest_path = ROOT / "manifest.json"
    previous = json.loads(manifest_path.read_text())["models"] if manifest_path.exists() else []
    loader_tree = ast.parse((ROOT / "source/vidur/execution_time_predictor/sklearn_execution_time_predictor.py").read_text())
    loader_signature = hashlib.sha256("".join(
        ast.dump(n) for n in ast.walk(loader_tree)
        if isinstance(n, ast.FunctionDef) and n.name in METHODS).encode()).hexdigest()

    def train(prefix, component, frame, features, target, source, filters):
        # Original loaders deduplicate and fill only their documented optional zeros.
        # Reject remaining non-finite/negative targets rather than invent replacements.
        values = frame[features + [target]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & (values[:, -1] >= 0)
        clean = frame.loc[valid]
        if clean.empty:
            raise ValueError(f"No valid rows: {prefix}/{component}")
        X, y = clean[features], clean[target]
        provenance = dict(component=component, features=features, target=target,
                          source=str(source.relative_to(ROOT)), filters=filters,
                          params=PARAMS, source_sha256=hashes[str(source.relative_to(ROOT))],
                          sklearn_version=sklearn.__version__)
        identity = dict(**provenance, loader_sha256=loader_signature,
                        training_frame_sha256=hashlib.sha256(
                            pd.util.hash_pandas_object(clean[features + [target]], index=False).values.tobytes()).hexdigest())
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        output = MODELS / prefix / f"{component}_{key}.joblib"
        # Reuse the existing export; exporter-only/documentation changes are not training changes.
        for old in previous:
            if all(old.get(k) == v for k, v in provenance.items()) and all(
                k not in old or old[k] == identity[k]
                for k in ["loader_sha256", "training_frame_sha256"]
            ) and (ROOT / old["model_path"]).is_file():
                output = ROOT / old["model_path"]
                break
        output.parent.mkdir(parents=True, exist_ok=True)
        cached = output.exists()
        if not cached:
            model = RandomForestRegressor(**PARAMS).fit(X, y)
            joblib.dump(model, output, compress=3)
        model = joblib.load(output)
        predictions = model.predict(X)
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError(f"Invalid reloaded predictions: {output}")
        sample = X.iloc[[0]]
        entries.append(dict(**identity, model_path=str(output.relative_to(ROOT)),
            target_unit="ms", feature_units={f: ("requests" if f == "batch_size" else "token^2" if f.endswith("squared") else "tokens") for f in features},
            rows_after_original_filters=len(frame), training_rows=len(clean),
            rejected_invalid_rows=int((~valid).sum()), cache_hit=cached,
            feature_ranges={f: [float(X[f].min()), float(X[f].max())] for f in features},
            example_input=sample.iloc[0].tolist(), example_prediction_ms=float(model.predict(sample)[0]),
            reload_validation_rows=len(X), reload_predictions_finite=True,
            training_zero_targets=int((y == 0).sum())))
        print(f"{prefix}/{component}: {len(clean)} rows, reload OK", flush=True)

    loader = original_loaders()
    compute_source = DATA / "compute/a100" / MODEL_NAME / "mlp.csv"
    attention_source = compute_source.with_name("attention.csv")
    export_measurement_combinations(compute_source,
        dict(n_head=32, n_kv_head=32, n_embd=4096, n_expanded_embd=11008,
             use_gated_mlp=True, vocab_size=32768),
        ["num_tensor_parallel_workers", "num_tokens"], ROOT / "mlp_measurement_combinations.csv")
    export_measurement_combinations(attention_source,
        dict(n_embd=4096, n_q_head=32, n_kv_head=32, block_size=16),
        ["num_tensor_parallel_workers", "batch_size", "prefill_chunk_size", "kv_cache_size", "block_size"],
        ROOT / "attention_measurement_combinations.csv")
    for tp in [1, 2, 4, 8]:
        loader._replica_config = SimpleNamespace(tensor_parallel_size=tp)
        filters = dict(device="a100", model=MODEL_NAME, num_tensor_parallel_workers=tp,
            n_head=32, n_kv_head=32, n_embd=4096, n_expanded_embd=11008,
            use_gated_mlp=True, vocab_size=32768)
        prefix = f"compute/a100/Llama-2-7b-hf/tp{tp}"
        compute = loader._get_compute_df_with_derived_features(loader._load_compute_df(str(compute_source)))
        for component in ["attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj",
                          "mlp_act", "input_layernorm", "post_attention_layernorm", "attn_rope", "add"]:
            train(prefix, component, compute, ["num_tokens"], f"time_stats.{component}.median", compute_source, filters)
        attention = loader._get_attention_df_with_derived_features(loader._load_attention_df(str(attention_source)))
        attn_filters = dict(device="a100", model=MODEL_NAME, n_embd=4096,
                            n_q_head=32, n_kv_head=32, block_size=16,
                            num_tensor_parallel_workers=tp)
        train(prefix, "attn_kv_cache_save", attention, ["num_tokens"],
              "time_stats.attn_kv_cache_save.median", attention_source, attn_filters)
        for component, decode, features in [("attn_prefill", False, ["kv_cache_size", "prefill_chunk_size_squared"]),
                                             ("attn_decode", True, ["batch_size", "kv_cache_size"])]:
            train(prefix, component, attention[attention.is_decode == decode], features,
                  f"time_stats.{component}.median", attention_source,
                  dict(attn_filters, is_decode=decode))

    manifest = dict(origin="https://github.com/microsoft/vidur.git",
        commit="abae7f63aa857300f5cdc6f5e0d27860cd24721b",
        original_repository="/home/zdg/OrbitServe/vidur", original_trained_models_found=False,
        method="Unchanged original loader/feature methods via AST; fixed RF first grid candidate; no GridSearchCV",
        original_grid=dict(n_estimators=[250,500,750], max_depth=[8,16,32], min_samples_split=[2,5,10], cv=10),
        differences=["No hyperparameter search/CV/holdout; fixed random_state=42 and four CPU workers",
                     "Hash-addressed joblib cache instead of original pickle/lock cache; no dense prediction cache",
                     "Reject remaining non-finite features/targets and negative targets after original filters"],
        original_import_blockers=["ray", "fasteners"],
        versions=dict(python=sys.version, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
        cpu_only=True, gpu_profiling_run=False, data_files=sources,
        compute_scope=dict(device="a100", model=MODEL_NAME, tp=[1,2,4,8], block_size=16),
        missing_components=["CPU overhead: repository contains no cpu_overhead profiles",
                            "KV cache save: timing columns absent; original loader substitutes zeros, not measured"],
        additional_bundled_data_not_trained="Other compute GPUs/models and all network data (communication prediction excluded)",
        feature_derivations=dict(num_tokens_attention="max(prefill_chunk_size, batch_size)",
            prefill_chunk_size_squared="prefill_chunk_size ** 2", is_decode="prefill_chunk_size == 0"),
        model_count=len(entries), models=entries,
        validation=dict(all_reloaded=True, all_training_inputs_finite_predictions=True,
                        checked_predictions=sum(e["reload_validation_rows"] for e in entries),
                        accuracy_claim=False))
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} models; manifest.json written", flush=True)


if __name__ == "__main__":
    main()
