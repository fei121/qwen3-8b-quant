import argparse
import json
from pathlib import Path

try:
    from scripts.common import extract_lm_eval_score, load_json
except ModuleNotFoundError:
    from common import extract_lm_eval_score, load_json


RUN_LABELS = {
    "baseline_bf16": "BF16 基线",
    "int8_w8a8": "INT8 W8A8",
    "mxfp4a16": "MXFP4A16",
}

QUANT_ALGORITHMS = {
    "baseline_bf16": "未量化；vLLM dtype=auto/BF16",
    "int8_w8a8": "LLM Compressor SmoothQuant + GPTQ，W8A8 INT8",
    "mxfp4a16": "LLM Compressor QuantizationModifier，MXFP4A16 weight-only",
}


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_json(folder, pattern):
    folder = Path(folder)
    paths = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def summarize_run(output_root, run_name, task_name):
    folder = Path(output_root) / run_name
    ceval_path = folder / "ceval.json"
    if not ceval_path.exists():
        ceval_path = latest_json(folder, "ceval*.json")
    result = load_json(ceval_path) if ceval_path else None
    return {
        "run_name": run_name,
        "ceval_path": str(ceval_path) if ceval_path else None,
        "ceval_score": extract_lm_eval_score(result, task_name),
    }


def build_summary(output_root, task_name):
    summary = {
        run_name: summarize_run(output_root, run_name, task_name)
        for run_name in RUN_LABELS
    }
    baseline = summary["baseline_bf16"]["ceval_score"]
    if baseline is not None:
        for run_name in ("int8_w8a8", "mxfp4a16"):
            score = summary[run_name]["ceval_score"]
            summary[run_name]["accuracy_drop_vs_bf16"] = (
                round(baseline - score, 8) if score is not None else None
            )
    return summary


def accuracy_table(summary):
    lines = [
        "| 实验组 | 精度/量化算法 | C-Eval 分数 | 相对 BF16 下降 | 原始结果文件 |",
        "|---|---|---:|---:|---|",
    ]
    for run_name, label in RUN_LABELS.items():
        run = summary.get(run_name, {})
        lines.append(
            "| {label} | {algorithm} | {score} | {drop} | `{path}` |".format(
                label=label,
                algorithm=QUANT_ALGORITHMS[run_name],
                score=fmt(run.get("ceval_score")),
                drop=fmt(run.get("accuracy_drop_vs_bf16")),
                path=run.get("ceval_path") or "-",
            )
        )
    return "\n".join(lines)


def generate_report(output_root, summary_dir, model_label, task_name, summary):
    baseline = summary["baseline_bf16"]["ceval_score"]
    int8 = summary["int8_w8a8"]["ceval_score"]
    mxfp4 = summary["mxfp4a16"]["ceval_score"]
    report = f"""# {model_label} C-Eval 量化评测报告

## 实验范围

- 模型：`{model_label}`
- 推理框架：vLLM
- 量化方案：LLM Compressor INT8 W8A8、MXFP4A16
- 精度评测：EleutherAI `lm-evaluation-harness` `{task_name}` 任务
- 主指标：`acc,none`，若无聚合项则对子任务可用 `acc` 指标取平均

## 精度结果

{accuracy_table(summary)}

## 主要结论

- INT8 W8A8 的 C-Eval 分数为 {fmt(int8)}，相对 BF16 下降 {fmt(baseline - int8) if baseline is not None and int8 is not None else "-"} 个绝对点。
- MXFP4A16 的 C-Eval 分数为 {fmt(mxfp4)}，相对 BF16 下降 {fmt(baseline - mxfp4) if baseline is not None and mxfp4 is not None else "-"} 个绝对点。

## 产物路径

- 汇总 JSON：`{output_root}/{summary_dir}/summary.json`
- 中文报告：`{output_root}/{summary_dir}/report.md`
- 原始评测 JSON：`{output_root}/baseline_bf16/ceval.json`、`{output_root}/int8_w8a8/ceval.json`、`{output_root}/mxfp4a16/ceval.json`
"""
    report_path = Path(output_root) / summary_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs_qwen3_8b")
    parser.add_argument("--model-label", default="Qwen/Qwen3-8B")
    parser.add_argument("--task-name", default="ceval-valid")
    parser.add_argument("--summary-dir", default="ceval_summary")
    args = parser.parse_args()

    summary = build_summary(args.output_root, args.task_name)
    summary_path = Path(args.output_root) / args.summary_dir / "summary.json"
    dump_json(summary_path, summary)
    generate_report(args.output_root, args.summary_dir, args.model_label, args.task_name, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
