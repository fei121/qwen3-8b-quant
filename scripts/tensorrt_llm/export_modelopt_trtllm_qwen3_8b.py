#!/usr/bin/env python
"""Export Qwen3-8B checkpoints with NVIDIA ModelOpt for TensorRT-LLM.

This script intentionally treats MXFP4 as a fake-quant PyTorch path. The
TensorRT-LLM 1.0.0 ModelOpt exporter installed on the remote host exposes
INT8 SmoothQuant through qformat=int8_sq, but does not expose MXFP4 as an
exportable TensorRT-LLM checkpoint format.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PRECISIONS = ("bf16", "int8_sq", "mxfp4_fake")


def dump_metadata(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def export_checkpoint(args):
    if args.precision == "mxfp4_fake":
        out_dir = args.output_root / "mxfp4_fake"
        dump_metadata(
            out_dir / "metadata.json",
            {
                "precision": "mxfp4_fake",
                "algorithm": "modelopt.torch.quantization.MXFP4_DEFAULT_CFG",
                "deployment": "fake_quant_pytorch",
                "note": "MXFP4 is evaluated in PyTorch fake-quant mode; no TensorRT-LLM engine is exported here.",
            },
        )
        print(f"wrote MXFP4 fake-quant metadata to {out_dir}")
        return out_dir

    from tensorrt_llm.quantization.quantize_by_modelopt import quantize_and_export

    qformat = "full_prec" if args.precision == "bf16" else "int8_sq"
    out_dir = args.output_root / args.precision / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    quantize_and_export(
        model_dir=str(args.model_dir),
        device="cuda",
        calib_dataset=args.calib_dataset,
        dtype=args.dtype,
        qformat=qformat,
        kv_cache_dtype=None,
        calib_size=args.calib_size,
        batch_size=args.batch_size,
        calib_max_seq_length=args.calib_max_seq_length,
        awq_block_size=128,
        output_dir=str(out_dir),
        tp_size=args.tp_size,
        pp_size=1,
        cp_size=1,
        seed=args.seed,
        tokenizer_max_seq_length=args.tokenizer_max_seq_length,
        device_map=args.device_map,
        quantize_lm_head=False,
    )
    dump_metadata(
        args.output_root / args.precision / "metadata.json",
        {
            "precision": args.precision,
            "model_dir": str(args.model_dir),
            "checkpoint_dir": str(out_dir),
            "qformat": qformat,
            "algorithm": "full_prec BF16 export"
            if args.precision == "bf16"
            else "ModelOpt INT8_SMOOTHQUANT_CFG via TensorRT-LLM qformat=int8_sq",
            "dtype": args.dtype,
            "calib_dataset": args.calib_dataset,
            "calib_size": args.calib_size,
            "calib_max_seq_length": args.calib_max_seq_length,
        },
    )
    print(f"exported {args.precision} checkpoint to {out_dir}")
    return out_dir


def build_engine(args, checkpoint_dir):
    if args.precision == "mxfp4_fake" or not args.build_engine:
        return None
    if shutil.which("trtllm-build") is None:
        raise RuntimeError("trtllm-build is not on PATH")

    engine_dir = args.output_root / args.precision / "engine"
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
    if args.extra_build_arg:
        command.extend(args.extra_build_arg)
    if args.gather_context_logits:
        command.append("--gather_context_logits")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"built {args.precision} TensorRT-LLM engine at {engine_dir}")
    return engine_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--calib-dataset", default="cnn_dailymail")
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-max-seq-length", type=int, default=1024)
    parser.add_argument("--tokenizer-max-seq-length", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument(
        "--delete-checkpoint-after-build",
        action="store_true",
        help="Remove the TensorRT-LLM checkpoint after a successful engine build to save disk.",
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument(
        "--gather-context-logits",
        action="store_true",
        help="Build the engine with context logits enabled for lm_eval loglikelihood tasks such as C-Eval.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-build-arg",
        action="append",
        default=[],
        help="Additional trtllm-build arguments, appended verbatim one token at a time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = export_checkpoint(args)
    engine_dir = build_engine(args, checkpoint_dir)
    if engine_dir is not None and args.delete_checkpoint_after_build:
        shutil.rmtree(checkpoint_dir)
        print(f"deleted checkpoint directory {checkpoint_dir}")


if __name__ == "__main__":
    main()
