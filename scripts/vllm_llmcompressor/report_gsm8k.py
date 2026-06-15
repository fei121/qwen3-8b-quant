import argparse
import json
from pathlib import Path

try:
    from scripts.common import extract_gsm8k_score, load_json, normalize_serve_metrics
except ModuleNotFoundError:
    from common import extract_gsm8k_score, load_json, normalize_serve_metrics


RUN_LABELS = {
    "baseline_bf16": "BF16 基线",
    "baseline_fp16": "FP16 基线",
    "int8_w8a8": "INT8 W8A8",
    "mxfp4a16": "MXFP4A16",
}

QUANT_ALGORITHMS = {
    "baseline_bf16": "未量化；vLLM dtype=auto/BF16",
    "baseline_fp16": "未量化；vLLM dtype=float16",
    "int8_w8a8": "LLM Compressor SmoothQuant + GPTQ，W8A8 INT8",
    "mxfp4a16": "LLM Compressor QuantizationModifier，MXFP4A16 weight-only",
}


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt0(value):
    if value is None:
        return "-"
    return f"{value:.0f}" if isinstance(value, float) else str(value)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_json(folder, pattern):
    folder = Path(folder)
    paths = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def summarize_run(output_root, name):
    folder = Path(output_root) / name
    gsm8k_path = folder / "gsm8k.json"
    if not gsm8k_path.exists():
        gsm8k_path = latest_json(folder, "gsm8k*.json")
    gsm8k = load_json(gsm8k_path) if gsm8k_path else None
    offline = load_json(folder / "offline_bench.json")
    serve = load_json(folder / "serve_bench.json")
    return {
        "run_name": name,
        "gsm8k_path": str(gsm8k_path) if gsm8k_path else None,
        "gsm8k_score": extract_gsm8k_score(gsm8k),
        "offline_bench": offline,
        "serve_bench": normalize_serve_metrics(serve) if serve else None,
    }


def build_summary(output_root):
    summary = {name: summarize_run(output_root, name) for name in RUN_LABELS}
    baseline = summary["baseline_bf16"]["gsm8k_score"]
    if baseline is not None:
        for name in ("int8_w8a8", "mxfp4a16"):
            score = summary[name]["gsm8k_score"]
            summary[name]["accuracy_drop_vs_bf16"] = (
                round(baseline - score, 8) if score is not None else None
            )
    return summary


def accuracy_table(summary):
    lines = [
        "| 实验组 | 精度/量化算法 | GSM8K 分数<br>`exact_match,flexible-extract` | 相对 BF16 下降 | 原始结果文件 |",
        "|---|---|---:|---:|---|",
    ]
    for name in RUN_LABELS:
        run = summary.get(name, {})
        lines.append(
            "| {label} | {algorithm} | {score} | {drop} | `{path}` |".format(
                label=RUN_LABELS[name],
                algorithm=QUANT_ALGORITHMS[name],
                score=fmt(run.get("gsm8k_score"), 4),
                drop=fmt(run.get("accuracy_drop_vs_bf16"), 4),
                path=run.get("gsm8k_path") or "-",
            )
        )
    return "\n".join(lines)


def offline_table(summary):
    lines = [
        "| 实验组 | 峰值显存 GB | Prefill tokens/s | Decode tokens/s |",
        "|---|---:|---:|---:|",
    ]
    for name in ("baseline_bf16", "int8_w8a8", "mxfp4a16"):
        bench = (summary.get(name, {}) or {}).get("offline_bench") or {}
        lines.append(
            "| {label} | {vram} | {prefill} | {decode} |".format(
                label=RUN_LABELS[name],
                vram=fmt(bench.get("peak_vram_gb"), 3),
                prefill=fmt0(bench.get("prefill_tokens_per_s")),
                decode=fmt0(bench.get("decode_tokens_per_s")),
            )
        )
    return "\n".join(lines)


def serve_table(summary):
    lines = [
        "| 实验组 | 请求吞吐 req/s | 总 tokens/s | 输出 tokens/s | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | E2E P50/P95/P99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline_bf16", "int8_w8a8", "mxfp4a16"):
        bench = (summary.get(name, {}) or {}).get("serve_bench") or {}
        lines.append(
            "| {label} | {rps} | {total} | {output} | {ttft} | {tpot} | {e2e} |".format(
                label=RUN_LABELS[name],
                rps=fmt(bench.get("request_throughput"), 3),
                total=fmt0(bench.get("total_tokens_per_s")),
                output=fmt0(bench.get("output_tokens_per_s")),
                ttft="/".join(fmt0(bench.get(k)) for k in ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms")),
                tpot="/".join(fmt0(bench.get(k)) for k in ("p50_tpot_ms", "p95_tpot_ms", "p99_tpot_ms")),
                e2e="/".join(fmt0(bench.get(k)) for k in ("p50_e2e_latency_ms", "p95_e2e_latency_ms", "p99_e2e_latency_ms")),
            )
        )
    return "\n".join(lines)


def svg_bar_chart(path, title, values, unit="", width=760, height=360):
    labels = list(values)
    nums = [values[label] or 0 for label in labels]
    max_value = max(nums) if nums else 1
    colors = ["#3268a8", "#6a7f2a", "#b65a3c", "#6b5fb5"]
    margin_left, margin_bottom, top = 120, 60, 54
    chart_w = width - margin_left - 40
    chart_h = height - top - margin_bottom
    bar_gap = 24
    bar_h = (chart_h - bar_gap * (len(labels) - 1)) / max(len(labels), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">{title}</text>',
    ]
    for index, label in enumerate(labels):
        y = top + index * (bar_h + bar_gap)
        value = nums[index]
        bar_w = 0 if max_value == 0 else chart_w * value / max_value
        parts.extend(
            [
                f'<text x="{margin_left - 12}" y="{y + bar_h * 0.62}" text-anchor="end" font-size="15" font-family="Arial">{label}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[index % len(colors)]}" rx="3"/>',
                f'<text x="{margin_left + bar_w + 8}" y="{y + bar_h * 0.62}" font-size="14" font-family="Arial">{fmt(value, 4)}{unit}</text>',
            ]
        )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def generate_visuals(output_root, summary):
    charts_dir = Path(output_root) / "summary" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    accuracy = {RUN_LABELS[k]: (summary.get(k, {}) or {}).get("gsm8k_score") for k in RUN_LABELS}
    decode = {
        RUN_LABELS[k]: ((summary.get(k, {}) or {}).get("offline_bench") or {}).get("decode_tokens_per_s")
        for k in ("baseline_bf16", "int8_w8a8", "mxfp4a16")
    }
    output = {
        RUN_LABELS[k]: ((summary.get(k, {}) or {}).get("serve_bench") or {}).get("output_tokens_per_s")
        for k in ("baseline_bf16", "int8_w8a8", "mxfp4a16")
    }
    svg_bar_chart(charts_dir / "accuracy.svg", "GSM8K 分数", accuracy)
    svg_bar_chart(charts_dir / "offline_decode.svg", "离线 Decode tokens/s", decode)
    svg_bar_chart(charts_dir / "serve_output.svg", "在线服务输出 tokens/s", output)


def generate_report(output_root, model_label, summary):
    baseline = summary["baseline_bf16"]["gsm8k_score"]
    int8 = summary["int8_w8a8"]["gsm8k_score"]
    mxfp4 = summary["mxfp4a16"]["gsm8k_score"]
    report = f"""# {model_label} 量化实验报告

## 实验范围

- 模型：`{model_label}`
- 推理框架：vLLM
- 量化方案：LLM Compressor INT8 W8A8、MXFP4A16
- 精度评测：EleutherAI `lm-evaluation-harness` 官方 `gsm8k` 任务，5-shot
- 主 GSM8K 指标：`exact_match,flexible-extract`
- 性能评测：vLLM offline generation、`vllm bench serve`

## 可视化

![GSM8K 分数](charts/accuracy.svg)

![离线 Decode tokens/s](charts/offline_decode.svg)

![在线服务输出 tokens/s](charts/serve_output.svg)

## 精度结果

{accuracy_table(summary)}

## 离线推理性能

{offline_table(summary)}

## 在线服务性能

{serve_table(summary)}

## 主要结论

- INT8 W8A8 的 GSM8K 分数为 {fmt(int8, 4)}，相对 BF16 下降 {fmt(baseline - int8, 4) if baseline is not None and int8 is not None else "-"} 个绝对点。
- MXFP4A16 的 GSM8K 分数为 {fmt(mxfp4, 4)}，相对 BF16 下降 {fmt(baseline - mxfp4, 4) if baseline is not None and mxfp4 is not None else "-"} 个绝对点。
- 性能结果需要结合精度下降一起看；低精度格式速度更快不一定意味着部署收益更好。

## 产物路径

- 汇总 JSON：`{output_root}/summary/summary.json`
- 中文报告：`{output_root}/summary/report.md`
- 可视化图表：`{output_root}/summary/charts/`
- 原始评测与 benchmark JSON：`{output_root}/baseline_*`、`{output_root}/int8_w8a8`、`{output_root}/mxfp4a16`
"""
    report_path = Path(output_root) / "summary" / "report.md"
    report_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model-label", default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    summary = build_summary(args.output_root)
    summary_path = Path(args.output_root) / "summary" / "summary.json"
    dump_json(summary_path, summary)
    generate_visuals(args.output_root, summary)
    generate_report(args.output_root, args.model_label, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
