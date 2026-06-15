from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path


from scripts.vllm_llmcompressor.activation_diagnosis import (
    LINEAR_MODULE_SUFFIXES,
    capture_module_outputs,
    choose_suspicious_layers,
    compare_layer_hidden_states,
    compare_module_outputs,
    dump_json,
    generate_report,
    load_model,
    load_probe_prompts,
    load_tokenizer,
    plot_distribution_overlays,
    plot_layer_metrics,
    plot_module_metrics,
    save_layer_hidden_states,
    tokenize_prompt,
    unload_model,
    write_csv,
    write_jsonl,
)


def load_calibration_texts(path: str, limit: int | None) -> list[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(item["text"])
            if limit and len(rows) >= limit:
                break
    return rows


def load_probe_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item["text"]
            rows.append(
                {
                    "id": item.get("id", f"probe_{index}"),
                    "dataset": item.get("dataset", "probe_jsonl"),
                    "text": text,
                    "probe_index": index,
                }
            )
    return rows


def quantize_modelopt_int8_smoothquant(model, tokenizer, calib_texts, max_length: int, device: str):
    import torch
    import modelopt.torch.quantization as mtq

    model.config.use_cache = False
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)

    def forward_loop(calib_model):
        calib_model.eval()
        with torch.inference_mode():
            for idx, text in enumerate(calib_texts, start=1):
                inputs = tokenize_prompt(tokenizer, text, max_length, device)
                calib_model(**inputs, use_cache=False)
                del inputs
                if torch.cuda.is_available() and idx % 32 == 0:
                    torch.cuda.empty_cache()

    return mtq.quantize(model, config, forward_loop)


def run(args):
    import torch

    output_dir = Path(args.output_dir)
    work_dir = output_dir / "work"
    stats_dir = output_dir / "stats"
    charts_dir = output_dir / "charts"
    logs_dir = output_dir / "logs"
    for folder in (work_dir, stats_dir, charts_dir, logs_dir, output_dir / "inputs"):
        folder.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.bf16_model, local_files_only=not args.allow_download)
    dataset_errors = []
    if args.probe_jsonl:
        prompts = load_probe_jsonl(args.probe_jsonl)
    else:
        prompts, dataset_errors = load_probe_prompts(tokenizer, args.num_gsm8k, args.num_ceval, args.seed)
    write_jsonl(output_dir / "inputs" / "probe_prompts.jsonl", prompts)
    if dataset_errors:
        (logs_dir / "dataset_warnings.log").write_text("\n".join(dataset_errors) + "\n", encoding="utf-8")

    calib_texts = load_calibration_texts(args.calib_jsonl, args.calib_limit)
    print(f"Loaded {len(calib_texts)} calibration samples from {args.calib_jsonl}", flush=True)

    print("Loading BF16 model for layer capture", flush=True)
    bf16_model = load_model(args.bf16_model, args.bf16_dtype, device, local_files_only=not args.allow_download)
    bf16_layer_dir = work_dir / "bf16_layers"
    token_counts = save_layer_hidden_states(bf16_model, tokenizer, prompts, bf16_layer_dir, args.max_length, device)
    unload_model(bf16_model)
    bf16_model = None

    print("Loading BF16 model and applying ModelOpt INT8 SmoothQuant", flush=True)
    quant_model = load_model(args.bf16_model, args.bf16_dtype, device, local_files_only=not args.allow_download)
    quant_model = quantize_modelopt_int8_smoothquant(
        quant_model,
        tokenizer,
        calib_texts,
        args.calib_max_length,
        device,
    )
    layer_rows = compare_layer_hidden_states(
        quant_model,
        tokenizer,
        prompts,
        bf16_layer_dir,
        args.max_length,
        device,
        args.sample_limit,
    )

    suspicious_layers = choose_suspicious_layers(layer_rows, limit=args.top_layers)
    write_csv(stats_dir / "layer_stats.csv", [{key: value for key, value in row.items() if key != "samples"} for row in layer_rows])
    dump_json(stats_dir / "layer_stats.json", layer_rows)
    dump_json(stats_dir / "suspicious_layers.json", suspicious_layers)
    plot_layer_metrics(charts_dir, layer_rows)
    plot_distribution_overlays(charts_dir, layer_rows, suspicious_layers)

    module_rows = []
    module_prompts = prompts[: min(args.module_prompts, len(prompts))]
    if args.module_prompts > 0 and suspicious_layers:
        print("Capturing BF16 module outputs", flush=True)
        unload_model(quant_model)
        quant_model = None
        gc.collect()

        bf16_model = load_model(args.bf16_model, args.bf16_dtype, device, local_files_only=not args.allow_download)
        bf16_module_dir = work_dir / "bf16_modules"
        capture_module_outputs(
            bf16_model,
            tokenizer,
            module_prompts,
            suspicious_layers,
            LINEAR_MODULE_SUFFIXES,
            bf16_module_dir,
            args.max_length,
            device,
        )
        unload_model(bf16_model)
        bf16_model = None

        print("Rebuilding ModelOpt INT8 model for module comparison", flush=True)
        quant_model = load_model(args.bf16_model, args.bf16_dtype, device, local_files_only=not args.allow_download)
        quant_model = quantize_modelopt_int8_smoothquant(
            quant_model,
            tokenizer,
            calib_texts,
            args.calib_max_length,
            device,
        )
        module_rows = compare_module_outputs(
            quant_model,
            tokenizer,
            module_prompts,
            bf16_module_dir,
            suspicious_layers,
            LINEAR_MODULE_SUFFIXES,
            args.max_length,
            device,
            args.sample_limit,
        )
        write_csv(stats_dir / "module_error_rank.csv", module_rows)
        dump_json(stats_dir / "module_error_rank.json", module_rows)
        plot_module_metrics(charts_dir, module_rows)
        unload_model(quant_model)
        quant_model = None
    else:
        unload_model(quant_model)
        quant_model = None

    metadata = {
        "diagnosis": "BF16 HF vs ModelOpt INT8 SmoothQuant PyTorch fake-quant path",
        "bf16_model": args.bf16_model,
        "quant_model": "in-memory ModelOpt INT8_SMOOTHQUANT_CFG",
        "calib_jsonl": args.calib_jsonl,
        "calib_samples": len(calib_texts),
        "num_prompts": len(prompts),
        "probe_jsonl": args.probe_jsonl,
        "num_gsm8k": args.num_gsm8k,
        "num_ceval": args.num_ceval,
        "max_length": args.max_length,
        "calib_max_length": args.calib_max_length,
        "sample_limit": args.sample_limit,
        "top_layers": args.top_layers,
        "module_prompt_count": len(module_prompts),
        "min_tokens": min(token_counts) if token_counts else None,
        "max_tokens": max(token_counts) if token_counts else None,
        "mean_tokens": sum(token_counts) / len(token_counts) if token_counts else None,
        "dataset_errors": dataset_errors,
        "device": device,
        "torch_version": torch.__version__,
    }
    dump_json(output_dir / "metadata.json", metadata)
    generate_report(output_dir, metadata, layer_rows, suspicious_layers, module_rows)
    report_path = output_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("BF16 vs W8A8", "BF16 vs ModelOpt INT8 SmoothQuant")
    report = report.replace("W8A8", "ModelOpt INT8")
    report_path.write_text(report, encoding="utf-8")

    if not args.keep_work:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Qwen3-8B BF16 vs ModelOpt INT8 SmoothQuant activations.")
    parser.add_argument("--bf16-model", default="/path/to/Qwen3-8B")
    parser.add_argument(
        "--calib-jsonl",
        default="/path/to/outputs/qwen3_8b_trtllm_int8_local_aligned_fullcalib/calib/ceval_dev_chat_seed42_full/train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/path/to/outputs/qwen3_8b_modelopt_int8_activation_diagnosis",
    )
    parser.add_argument("--bf16-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num-gsm8k", type=int, default=16)
    parser.add_argument("--num-ceval", type=int, default=16)
    parser.add_argument("--probe-jsonl", default="")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--calib-max-length", type=int, default=1024)
    parser.add_argument("--calib-limit", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--top-layers", type=int, default=5)
    parser.add_argument("--module-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
