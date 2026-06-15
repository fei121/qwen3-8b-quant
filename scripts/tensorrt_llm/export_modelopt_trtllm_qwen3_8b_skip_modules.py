#!/usr/bin/env python
"""Export Qwen3-8B TensorRT-LLM INT8 SmoothQuant with selected modules left unquantized."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path


def dump_metadata(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv(value):
    items = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def parse_layers(value):
    if not value:
        return []
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def build_skip_patterns(modules, layers):
    patterns = []
    if layers:
        for layer in layers:
            for module in modules:
                patterns.extend(
                    [
                        f"*model.layers.{layer}.{module}*",
                        f"*layers.{layer}.{module}*",
                    ]
                )
    else:
        for module in modules:
            patterns.extend(
                [
                    f"*model.layers.*.{module}*",
                    f"*layers.*.{module}*",
                    f"*{module}*",
                ]
            )
    return patterns


def export_checkpoint(args):
    import modelopt.torch.quantization as mtq
    import tensorrt_llm.quantization.quantize_by_modelopt as qbm

    base_choices = qbm.quant_cfg_choices
    skip_patterns = build_skip_patterns(args.skip_modules, args.skip_layers)

    def patched_quant_cfg_choices():
        choices = base_choices()
        cfg = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        for pattern in skip_patterns:
            cfg["quant_cfg"][pattern] = {"enable": False}
        choices["int8_sq"] = cfg
        return choices

    qbm.quant_cfg_choices = patched_quant_cfg_choices
    out_dir = args.output_root / "int8_sq" / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    qbm.quantize_and_export(
        model_dir=str(args.model_dir),
        device="cuda",
        calib_dataset=args.calib_dataset,
        dtype=args.dtype,
        qformat="int8_sq",
        kv_cache_dtype=None,
        calib_size=args.calib_size,
        batch_size=args.batch_size,
        calib_max_seq_length=args.calib_max_seq_length,
        awq_block_size=128,
        output_dir=str(out_dir),
        tp_size=1,
        pp_size=1,
        cp_size=1,
        seed=args.seed,
        tokenizer_max_seq_length=args.tokenizer_max_seq_length,
        device_map=args.device_map,
        quantize_lm_head=False,
    )
    dump_metadata(
        args.output_root / "int8_sq" / "metadata.json",
        {
            "precision": "int8_sq",
            "algorithm": "ModelOpt INT8_SMOOTHQUANT_CFG with selected module quantizers disabled",
            "model_dir": str(args.model_dir),
            "calib_dataset": args.calib_dataset,
            "calib_size": args.calib_size,
            "calib_max_seq_length": args.calib_max_seq_length,
            "skip_modules": args.skip_modules,
            "skip_layers": args.skip_layers,
            "skip_patterns": skip_patterns,
            "checkpoint_dir": str(out_dir),
        },
    )
    return out_dir


def build_engine(args, checkpoint_dir):
    if shutil.which("trtllm-build") is None:
        raise RuntimeError("trtllm-build is not on PATH")
    engine_dir = args.output_root / "int8_sq" / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "trtllm-build",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(engine_dir),
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_input_len",
        str(args.max_input_len),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_num_tokens",
        str(args.max_num_tokens),
        "--gemm_plugin",
        "auto",
        "--gpt_attention_plugin",
        "auto",
    ]
    if args.gather_context_logits:
        command.append("--gather_context_logits")
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        engine_files = [path for path in engine_dir.glob("*.engine") if path.stat().st_size > 1024**3]
        if exc.returncode in {139, -11} and engine_files:
            print(
                f"WARNING: trtllm-build exited with {exc.returncode} after writing engine; "
                "treating TensorRT-LLM teardown segfault as success.",
                flush=True,
            )
        else:
            raise
    return engine_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calib-dataset", required=True)
    parser.add_argument("--calib-size", type=int, required=True)
    parser.add_argument("--skip-modules", type=parse_csv, required=True)
    parser.add_argument("--skip-layers", type=parse_layers, default=[])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-max-seq-length", type=int, default=1024)
    parser.add_argument("--tokenizer-max-seq-length", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=1280)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1280)
    parser.add_argument("--gather-context-logits", action="store_true")
    parser.add_argument("--delete-checkpoint-after-build", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = export_checkpoint(args)
    engine_dir = build_engine(args, checkpoint_dir)
    if args.delete_checkpoint_after_build:
        shutil.rmtree(checkpoint_dir)
        print(f"deleted checkpoint directory {checkpoint_dir}")
    print(f"built TensorRT-LLM engine at {engine_dir}")


if __name__ == "__main__":
    main()
