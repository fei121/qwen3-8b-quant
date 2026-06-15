from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path


LINEAR_MODULE_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


FALLBACK_GSM8K = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest for $2 each. How much does she make each day?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and puts in $50,000 in repairs. This increased the value by 150%. How much profit did he make?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. If there are 20 chickens, how many cups of feed are needed in 7 days?",
]


FALLBACK_CEVAL = [
    "以下哪一项最能体现依法治国的基本要求？\nA. 以道德代替法律\nB. 法律面前人人平等\nC. 只强调行政命令\nD. 完全依赖个人经验",
    "若函数 f(x)=x^2，则 f(3) 等于多少？\nA. 3\nB. 6\nC. 9\nD. 12",
    "计算机网络中，TCP 协议主要提供什么服务？\nA. 不可靠数据报传输\nB. 可靠的面向连接传输\nC. 物理信号编码\nD. 图像渲染",
    "在化学反应中，催化剂的作用通常是？\nA. 改变化学平衡常数\nB. 降低反应活化能\nC. 消耗全部反应物\nD. 增加生成物质量",
]


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


@dataclass
class PairAccumulator:
    name: str
    sample_limit: int = 8192
    count: int = 0
    ref_sum: float = 0.0
    quant_sum: float = 0.0
    ref_sq_sum: float = 0.0
    quant_sq_sum: float = 0.0
    dot_sum: float = 0.0
    diff_sum: float = 0.0
    diff_sq_sum: float = 0.0
    abs_diff_sum: float = 0.0
    ref_samples: list[float] = field(default_factory=list)
    quant_samples: list[float] = field(default_factory=list)
    diff_samples: list[float] = field(default_factory=list)
    abs_diff_samples: list[float] = field(default_factory=list)

    def update_lists(self, ref_values, quant_values):
        pairs = [(float(ref), float(quant)) for ref, quant in zip(ref_values, quant_values)]
        for ref, quant in pairs:
            diff = quant - ref
            self.count += 1
            self.ref_sum += ref
            self.quant_sum += quant
            self.ref_sq_sum += ref * ref
            self.quant_sq_sum += quant * quant
            self.dot_sum += ref * quant
            self.diff_sum += diff
            self.diff_sq_sum += diff * diff
            self.abs_diff_sum += abs(diff)
        self._append_samples_from_lists(pairs)

    def update_tensors(self, ref_tensor, quant_tensor):
        import torch

        ref = ref_tensor.detach().float().cpu().reshape(-1)
        quant = quant_tensor.detach().float().cpu().reshape(-1)
        if ref.numel() != quant.numel():
            raise ValueError(f"{self.name} shape mismatch: {tuple(ref_tensor.shape)} vs {tuple(quant_tensor.shape)}")
        diff = quant - ref
        abs_diff = diff.abs()
        self.count += int(ref.numel())
        self.ref_sum += float(ref.sum().item())
        self.quant_sum += float(quant.sum().item())
        self.ref_sq_sum += float((ref * ref).sum().item())
        self.quant_sq_sum += float((quant * quant).sum().item())
        self.dot_sum += float((ref * quant).sum().item())
        self.diff_sum += float(diff.sum().item())
        self.diff_sq_sum += float((diff * diff).sum().item())
        self.abs_diff_sum += float(abs_diff.sum().item())

        remaining = self.sample_limit - len(self.ref_samples)
        if remaining <= 0 or ref.numel() == 0:
            return
        take = min(remaining, int(ref.numel()))
        if take == int(ref.numel()):
            indices = torch.arange(ref.numel())
        else:
            indices = torch.linspace(0, ref.numel() - 1, steps=take).long()
        self.ref_samples.extend(float(value) for value in ref[indices].tolist())
        self.quant_samples.extend(float(value) for value in quant[indices].tolist())
        self.diff_samples.extend(float(value) for value in diff[indices].tolist())
        self.abs_diff_samples.extend(float(value) for value in abs_diff[indices].tolist())

    def _append_samples_from_lists(self, pairs):
        remaining = self.sample_limit - len(self.ref_samples)
        if remaining <= 0:
            return
        for ref, quant in pairs[:remaining]:
            diff = quant - ref
            self.ref_samples.append(ref)
            self.quant_samples.append(quant)
            self.diff_samples.append(diff)
            self.abs_diff_samples.append(abs(diff))

    def finalize(self, include_samples=False):
        if self.count == 0:
            return {"name": self.name, "count": 0}
        mse = self.diff_sq_sum / self.count
        ref_mean = self.ref_sum / self.count
        quant_mean = self.quant_sum / self.count
        ref_var = max(self.ref_sq_sum / self.count - ref_mean * ref_mean, 0.0)
        quant_var = max(self.quant_sq_sum / self.count - quant_mean * quant_mean, 0.0)
        denom = math.sqrt(self.ref_sq_sum) * math.sqrt(self.quant_sq_sum)
        cosine = self.dot_sum / denom if denom else None
        sqnr_db = math.inf if self.diff_sq_sum == 0 else 10.0 * math.log10(max(self.ref_sq_sum, 1e-30) / self.diff_sq_sum)
        result = {
            "name": self.name,
            "count": self.count,
            "ref_mean": ref_mean,
            "quant_mean": quant_mean,
            "ref_std": math.sqrt(ref_var),
            "quant_std": math.sqrt(quant_var),
            "diff_mean": self.diff_sum / self.count,
            "mae": self.abs_diff_sum / self.count,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "cosine": cosine,
            "sqnr_db": sqnr_db,
            "abs_error_p50": percentile(self.abs_diff_samples, 50),
            "abs_error_p90": percentile(self.abs_diff_samples, 90),
            "abs_error_p99": percentile(self.abs_diff_samples, 99),
            "abs_error_p999": percentile(self.abs_diff_samples, 99.9),
            "abs_error_max_sampled": max(self.abs_diff_samples) if self.abs_diff_samples else None,
            "sample_count": len(self.ref_samples),
        }
        if include_samples:
            result["samples"] = {
                "ref": self.ref_samples,
                "quant": self.quant_samples,
                "diff": self.diff_samples,
                "abs_diff": self.abs_diff_samples,
            }
        return result


def choose_suspicious_layers(rows, limit=5):
    def sort_key(row):
        cosine = row.get("cosine")
        sqnr = row.get("sqnr_db")
        p99 = row.get("abs_error_p99")
        return (
            cosine if cosine is not None else 2.0,
            sqnr if sqnr is not None and not math.isinf(sqnr) else 1e9,
            -(p99 or 0.0),
        )

    ranked = sorted(rows, key=sort_key)
    return [int(row["layer"]) for row in ranked[:limit]]


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    sample_keys = {"samples"}
    fieldnames = [key for key in rows[0] if key not in sample_keys]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_tokenizer(model_path, local_files_only=True):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path, dtype_name, device, local_files_only=True):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype_name],
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)
    return model


def unload_model(model):
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def format_gsm8k_prompt(row):
    return f"Question: {row['question']}\nAnswer:"


def format_ceval_prompt(row):
    question = row.get("question", "")
    choices = []
    for key in ("A", "B", "C", "D"):
        if key in row and row[key] is not None:
            choices.append(f"{key}. {row[key]}")
    return "以下是一道单项选择题，请选择正确答案。\n" + question + "\n" + "\n".join(choices) + "\n答案："


def apply_chat_template(tokenizer, text):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def fallback_prompts(tokenizer, num_gsm8k, num_ceval):
    rows = []
    for idx, text in enumerate((FALLBACK_GSM8K * ((num_gsm8k // len(FALLBACK_GSM8K)) + 1))[:num_gsm8k]):
        rows.append({"id": f"gsm8k_fallback_{idx}", "dataset": "gsm8k_fallback", "text": apply_chat_template(tokenizer, text)})
    for idx, text in enumerate((FALLBACK_CEVAL * ((num_ceval // len(FALLBACK_CEVAL)) + 1))[:num_ceval]):
        rows.append({"id": f"ceval_fallback_{idx}", "dataset": "ceval_fallback", "text": apply_chat_template(tokenizer, text)})
    return rows


def load_probe_prompts(tokenizer, num_gsm8k, num_ceval, seed):
    rows = []
    errors = []
    random.seed(seed)
    try:
        from datasets import load_dataset

        gsm8k = load_dataset("gsm8k", "main", split=f"test[:{num_gsm8k}]")
        for idx, row in enumerate(gsm8k):
            rows.append(
                {
                    "id": f"gsm8k_{idx}",
                    "dataset": "gsm8k",
                    "text": apply_chat_template(tokenizer, format_gsm8k_prompt(row)),
                }
            )
    except Exception as exc:
        errors.append(f"gsm8k load failed: {type(exc).__name__}: {exc}")

    ceval_subjects = [
        "high_school_mathematics",
        "computer_network",
        "college_physics",
        "high_school_chinese",
        "law",
        "middle_school_chemistry",
    ]
    try:
        from datasets import load_dataset

        ceval_count = 0
        for subject in ceval_subjects:
            if ceval_count >= num_ceval:
                break
            split = load_dataset("ceval/ceval-exam", subject, split="val")
            for row in split:
                if ceval_count >= num_ceval:
                    break
                rows.append(
                    {
                        "id": f"ceval_{subject}_{ceval_count}",
                        "dataset": f"ceval/{subject}",
                        "text": apply_chat_template(tokenizer, format_ceval_prompt(row)),
                    }
                )
                ceval_count += 1
    except Exception as exc:
        errors.append(f"ceval load failed: {type(exc).__name__}: {exc}")

    if len(rows) < num_gsm8k + num_ceval:
        fallback = fallback_prompts(tokenizer, num_gsm8k, num_ceval)
        existing = len(rows)
        rows.extend(fallback[: max(0, num_gsm8k + num_ceval - existing)])
    for index, row in enumerate(rows):
        row["probe_index"] = index
    return rows[: num_gsm8k + num_ceval], errors


def tokenize_prompt(tokenizer, text, max_length, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def save_layer_hidden_states(model, tokenizer, prompts, output_dir, max_length, device):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_counts = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenize_prompt(tokenizer, prompt["text"], max_length, device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            layers = {str(idx): tensor.detach().to("cpu", dtype=torch.float16) for idx, tensor in enumerate(hidden_states)}
            torch.save({"id": prompt["id"], "layers": layers}, output_dir / f"{prompt['probe_index']:04d}.pt")
            token_counts.append(int(inputs["input_ids"].shape[-1]))
            del outputs, hidden_states, layers, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return token_counts


def compare_layer_hidden_states(model, tokenizer, prompts, bf16_dir, max_length, device, sample_limit):
    import torch

    accumulators = None
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenize_prompt(tokenizer, prompt["text"], max_length, device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            quant_hidden_states = outputs.hidden_states[1:]
            bf16_payload = torch.load(Path(bf16_dir) / f"{prompt['probe_index']:04d}.pt", map_location="cpu")
            if accumulators is None:
                accumulators = [PairAccumulator(f"layer_{idx}", sample_limit=sample_limit) for idx in range(len(quant_hidden_states))]
            for idx, quant_tensor in enumerate(quant_hidden_states):
                accumulators[idx].update_tensors(bf16_payload["layers"][str(idx)], quant_tensor.detach().cpu())
            del outputs, quant_hidden_states, bf16_payload, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows = []
    for idx, accumulator in enumerate(accumulators or []):
        row = accumulator.finalize(include_samples=True)
        row["layer"] = idx
        rows.append(row)
    return rows


def capture_module_outputs(model, tokenizer, prompts, layers, module_suffixes, output_dir, max_length, device):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_map = {}
    for layer in layers:
        for suffix in module_suffixes:
            name = f"model.layers.{layer}.{suffix}"
            try:
                module_map[f"layer_{layer}.{suffix}"] = model.get_submodule(name)
            except AttributeError:
                continue

    with torch.inference_mode():
        for prompt in prompts:
            captures = {}
            handles = []
            for key, module in module_map.items():
                def make_hook(capture_key):
                    def hook(_module, _inputs, output):
                        captures[capture_key] = output.detach().to("cpu", dtype=torch.float16)
                    return hook

                handles.append(module.register_forward_hook(make_hook(key)))
            inputs = tokenize_prompt(tokenizer, prompt["text"], max_length, device)
            model(**inputs, use_cache=False)
            for handle in handles:
                handle.remove()
            torch.save({"id": prompt["id"], "modules": captures}, output_dir / f"{prompt['probe_index']:04d}.pt")
            del captures, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def compare_module_outputs(model, tokenizer, prompts, bf16_dir, layers, module_suffixes, max_length, device, sample_limit):
    import torch

    keys = [f"layer_{layer}.{suffix}" for layer in layers for suffix in module_suffixes]
    accumulators = {key: PairAccumulator(key, sample_limit=sample_limit) for key in keys}
    with torch.inference_mode():
        for prompt in prompts:
            captures = {}
            handles = []
            for layer in layers:
                for suffix in module_suffixes:
                    key = f"layer_{layer}.{suffix}"
                    name = f"model.layers.{layer}.{suffix}"
                    try:
                        module = model.get_submodule(name)
                    except AttributeError:
                        continue

                    def make_hook(capture_key):
                        def hook(_module, _inputs, output):
                            captures[capture_key] = output.detach().to("cpu", dtype=torch.float16)
                        return hook

                    handles.append(module.register_forward_hook(make_hook(key)))
            inputs = tokenize_prompt(tokenizer, prompt["text"], max_length, device)
            model(**inputs, use_cache=False)
            for handle in handles:
                handle.remove()
            bf16_payload = torch.load(Path(bf16_dir) / f"{prompt['probe_index']:04d}.pt", map_location="cpu")
            for key, quant_tensor in captures.items():
                bf16_tensor = bf16_payload["modules"].get(key)
                if bf16_tensor is not None:
                    accumulators[key].update_tensors(bf16_tensor, quant_tensor)
            del captures, inputs, bf16_payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows = []
    for key, accumulator in accumulators.items():
        row = accumulator.finalize(include_samples=False)
        row["module"] = key
        layer_part, suffix = key.split(".", 1)
        row["layer"] = int(layer_part.replace("layer_", ""))
        row["module_suffix"] = suffix
        rows.append(row)
    rows.sort(key=lambda row: (
        row.get("cosine") if row.get("cosine") is not None else 2.0,
        row.get("sqnr_db") if row.get("sqnr_db") is not None and not math.isinf(row.get("sqnr_db")) else 1e9,
        -(row.get("abs_error_p99") or 0.0),
    ))
    return rows


def plot_layer_metrics(charts_dir, layer_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    layers = [row["layer"] for row in layer_rows]

    def line_chart(filename, values, ylabel, title):
        plt.figure(figsize=(11, 4.8))
        plt.plot(layers, values, marker="o", linewidth=1.6)
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(charts_dir / filename, dpi=160)
        plt.close()

    line_chart("layer_cosine.png", [row.get("cosine") for row in layer_rows], "Cosine similarity", "BF16 vs W8A8 block output cosine")
    line_chart("layer_sqnr.png", [row.get("sqnr_db") for row in layer_rows], "SQNR (dB)", "BF16 vs W8A8 block output SQNR")
    line_chart("layer_p99_abs_error.png", [row.get("abs_error_p99") for row in layer_rows], "p99 abs error", "BF16 vs W8A8 block output p99 absolute error")

    heatmap = np.array([
        [1.0 - (row.get("cosine") or 0.0) for row in layer_rows],
        [row.get("rmse") or 0.0 for row in layer_rows],
        [row.get("abs_error_p99") or 0.0 for row in layer_rows],
    ])
    for row_idx in range(heatmap.shape[0]):
        max_value = heatmap[row_idx].max()
        if max_value > 0:
            heatmap[row_idx] = heatmap[row_idx] / max_value
    plt.figure(figsize=(12, 3.6))
    plt.imshow(heatmap, aspect="auto", cmap="magma")
    plt.yticks([0, 1, 2], ["1 - cosine", "RMSE", "p99 abs err"])
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.colorbar(label="Normalized per metric")
    plt.title("Layer-level activation divergence heatmap")
    plt.tight_layout()
    plt.savefig(charts_dir / "layer_error_heatmap.png", dpi=160)
    plt.close()


def plot_distribution_overlays(charts_dir, layer_rows, suspicious_layers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    rows_by_layer = {row["layer"]: row for row in layer_rows}
    for layer in suspicious_layers:
        row = rows_by_layer[layer]
        samples = row.get("samples", {})
        ref = samples.get("ref", [])
        quant = samples.get("quant", [])
        diff = samples.get("diff", [])
        if not ref or not quant:
            continue
        plt.figure(figsize=(11, 4.8))
        plt.hist(ref, bins=80, alpha=0.46, density=True, label="BF16", color="#3268a8")
        plt.hist(quant, bins=80, alpha=0.46, density=True, label="W8A8", color="#b65a3c")
        plt.title(f"Layer {layer} activation distribution")
        plt.xlabel("Activation value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / f"distribution_layer_{layer:02d}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(11, 4.8))
        plt.hist(diff, bins=80, alpha=0.78, density=True, color="#6a7f2a")
        plt.title(f"Layer {layer} quant - BF16 error distribution")
        plt.xlabel("Activation error")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(charts_dir / f"error_distribution_layer_{layer:02d}.png", dpi=160)
        plt.close()


def plot_module_metrics(charts_dir, module_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    top = module_rows[:20]
    labels = [row["module"].replace("self_attn.", "attn.").replace("mlp.", "mlp.") for row in top]
    cosines = [row.get("cosine") for row in top]
    sqnr = [row.get("sqnr_db") for row in top]

    plt.figure(figsize=(11, max(5, len(top) * 0.34)))
    plt.barh(range(len(top)), cosines, color="#3268a8")
    plt.yticks(range(len(top)), labels)
    plt.xlabel("Cosine similarity")
    plt.title("Most divergent module outputs by cosine")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(charts_dir / "module_cosine_rank.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, max(5, len(top) * 0.34)))
    plt.barh(range(len(top)), sqnr, color="#b65a3c")
    plt.yticks(range(len(top)), labels)
    plt.xlabel("SQNR (dB)")
    plt.title("Most divergent module outputs by SQNR")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(charts_dir / "module_sqnr_rank.png", dpi=160)
    plt.close()


def fmt(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows, columns, limit=None):
    selected = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(label for label, _key in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(fmt(row.get(key), 6) for _label, key in columns) + " |")
    return "\n".join(lines)


def generate_report(output_dir, metadata, layer_rows, suspicious_layers, module_rows):
    output_dir = Path(output_dir)
    worst_by_cosine = sorted(layer_rows, key=lambda row: row.get("cosine") if row.get("cosine") is not None else 2.0)[0]
    worst_by_sqnr = sorted(layer_rows, key=lambda row: row.get("sqnr_db") if row.get("sqnr_db") is not None and not math.isinf(row.get("sqnr_db")) else 1e9)[0]
    worst_module = module_rows[0] if module_rows else None
    mean_cosine = sum(row.get("cosine") or 0.0 for row in layer_rows) / max(len(layer_rows), 1)
    min_cosine = worst_by_cosine.get("cosine")

    if min_cosine is not None and min_cosine > 0.995:
        conclusion = "W8A8 的逐层 hidden states 与 BF16 整体高度一致，当前小幅掉点更像是多层轻微量化噪声累积，而不是单层灾难性漂移。"
    elif min_cosine is not None and min_cosine > 0.98:
        conclusion = "W8A8 的整体激活方向仍然稳定，但少数层出现可见偏差，应优先围绕这些层做 SmoothQuant 强度、校准集和 selective skip 对比。"
    else:
        conclusion = "W8A8 在少数层出现明显激活偏移，建议优先跳过这些层或其关键 Linear 模块后重新量化验证。"

    report = f"""# Qwen3-8B BF16 vs W8A8 激活分布诊断报告

## 结论摘要

{conclusion}

- 逐层平均 cosine similarity：`{fmt(mean_cosine, 6)}`。
- 最低逐层 cosine similarity：layer `{worst_by_cosine['layer']}`，cosine `{fmt(worst_by_cosine.get('cosine'), 6)}`，SQNR `{fmt(worst_by_cosine.get('sqnr_db'), 3)}` dB。
- 最低逐层 SQNR：layer `{worst_by_sqnr['layer']}`，SQNR `{fmt(worst_by_sqnr.get('sqnr_db'), 3)}` dB，p99 abs error `{fmt(worst_by_sqnr.get('abs_error_p99'), 6)}`。
- 本轮建议重点排查层：`{', '.join(str(layer) for layer in suspicious_layers)}`。
"""
    if worst_module:
        report += f"- 模块级 native forward 对比中最敏感模块：`{worst_module['module']}`，cosine `{fmt(worst_module.get('cosine'), 6)}`，SQNR `{fmt(worst_module.get('sqnr_db'), 3)}` dB。\n"

    report += f"""
## 实验配置

| 项目 | 值 |
|---|---|
| BF16 model | `{metadata['bf16_model']}` |
| W8A8 model | `{metadata['quant_model']}` |
| prompt 数 | `{metadata['num_prompts']}` |
| token 长度 | min `{metadata['min_tokens']}`, mean `{fmt(metadata['mean_tokens'], 2)}`, max `{metadata['max_tokens']}` |
| max length | `{metadata['max_length']}` |
| block 输出采样上限/层 | `{metadata['sample_limit']}` |
| module probe prompt 数 | `{metadata['module_prompt_count']}` |
| 说明 | module 级对比是在各模型 native forward 下比较模块输出，可定位误差显现位置；因 SmoothQuant 会改变前序缩放关系，不把它直接解释成严格因果归因。 |

## 可视化

![Layer error heatmap](charts/layer_error_heatmap.png)

![Layer cosine](charts/layer_cosine.png)

![Layer SQNR](charts/layer_sqnr.png)

![Layer p99 abs error](charts/layer_p99_abs_error.png)
"""

    for layer in suspicious_layers:
        report += f"\n![Layer {layer} activation distribution](charts/distribution_layer_{layer:02d}.png)\n"
        report += f"\n![Layer {layer} error distribution](charts/error_distribution_layer_{layer:02d}.png)\n"

    if module_rows:
        report += """
![Module cosine rank](charts/module_cosine_rank.png)

![Module SQNR rank](charts/module_sqnr_rank.png)
"""

    report += """
## 逐层误差排名

"""
    report += markdown_table(
        sorted(layer_rows, key=lambda row: row.get("cosine") if row.get("cosine") is not None else 2.0),
        [
            ("layer", "layer"),
            ("cosine", "cosine"),
            ("SQNR dB", "sqnr_db"),
            ("RMSE", "rmse"),
            ("p99 abs err", "abs_error_p99"),
            ("diff mean", "diff_mean"),
        ],
        limit=12,
    )

    if module_rows:
        report += """

## 模块级误差排名

"""
        report += markdown_table(
            module_rows,
            [
                ("module", "module"),
                ("cosine", "cosine"),
                ("SQNR dB", "sqnr_db"),
                ("RMSE", "rmse"),
                ("p99 abs err", "abs_error_p99"),
                ("diff mean", "diff_mean"),
            ],
            limit=20,
        )

    report += """

## 解释与下一步

1. 如果低 cosine / 低 SQNR 集中在少数层，下一轮优先尝试 selective skip：跳过这些层或其中排名靠前的 Linear 模块。
2. 如果 `mlp.down_proj`、`self_attn.o_proj` 等输出投影反复出现在模块误差前列，优先测试这些模块保留高精度或使用更保守 scheme。
3. 如果所有层曲线都比较平滑且没有突变，优先调 `SmoothQuantModifier(smoothing_strength=...)` 和校准集，而不是跳单层。
4. 每次只改一个变量，重新生成 W8A8 模型后必须用 vLLM 复跑 GSM8K/C-Eval，不能只凭激活诊断判断精度改善。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def probe_load(args):
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device", device)
    tokenizer = load_tokenizer(args.bf16_model, local_files_only=not args.allow_download)
    print("tokenizer", tokenizer.__class__.__name__)
    for label, path, dtype in [
        ("bf16", args.bf16_model, args.bf16_dtype),
        ("quant", args.quant_model, args.quant_dtype),
    ]:
        print(f"loading {label}: {path}")
        model = load_model(path, dtype, device, local_files_only=not args.allow_download)
        print(label, model.__class__.__name__, "layers", len(model.model.layers))
        inputs = tokenize_prompt(tokenizer, apply_chat_template(tokenizer, "Probe prompt."), min(args.max_length, 64), device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        print(label, "hidden states", len(outputs.hidden_states), "last", tuple(outputs.hidden_states[-1].shape))
        del outputs, inputs
        unload_model(model)
        model = None


def run(args):
    import torch

    output_dir = Path(args.output_dir)
    work_dir = output_dir / "work"
    stats_dir = output_dir / "stats"
    charts_dir = output_dir / "charts"
    logs_dir = output_dir / "logs"
    for folder in (work_dir, stats_dir, charts_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.bf16_model, local_files_only=not args.allow_download)
    prompts, dataset_errors = load_probe_prompts(tokenizer, args.num_gsm8k, args.num_ceval, args.seed)
    write_jsonl(output_dir / "inputs" / "probe_prompts.jsonl", prompts)
    if dataset_errors:
        (logs_dir / "dataset_warnings.log").write_text("\n".join(dataset_errors) + "\n", encoding="utf-8")

    print("Loading BF16 model")
    bf16_model = load_model(args.bf16_model, args.bf16_dtype, device, local_files_only=not args.allow_download)
    bf16_layer_dir = work_dir / "bf16_layers"
    token_counts = save_layer_hidden_states(bf16_model, tokenizer, prompts, bf16_layer_dir, args.max_length, device)
    unload_model(bf16_model)
    bf16_model = None

    print("Loading W8A8 model")
    quant_model = load_model(args.quant_model, args.quant_dtype, device, local_files_only=not args.allow_download)
    layer_rows = compare_layer_hidden_states(quant_model, tokenizer, prompts, bf16_layer_dir, args.max_length, device, args.sample_limit)
    unload_model(quant_model)
    quant_model = None

    suspicious_layers = choose_suspicious_layers(layer_rows, limit=args.top_layers)
    write_csv(stats_dir / "layer_stats.csv", [{key: value for key, value in row.items() if key != "samples"} for row in layer_rows])
    dump_json(stats_dir / "layer_stats.json", layer_rows)
    dump_json(stats_dir / "suspicious_layers.json", suspicious_layers)
    plot_layer_metrics(charts_dir, layer_rows)
    plot_distribution_overlays(charts_dir, layer_rows, suspicious_layers)

    module_rows = []
    module_prompts = prompts[: min(args.module_prompts, len(prompts))]
    if args.module_prompts > 0 and suspicious_layers:
        print("Loading BF16 model for module pass")
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

        print("Loading W8A8 model for module pass")
        quant_model = load_model(args.quant_model, args.quant_dtype, device, local_files_only=not args.allow_download)
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
        unload_model(quant_model)
        quant_model = None
        write_csv(stats_dir / "module_error_rank.csv", module_rows)
        dump_json(stats_dir / "module_error_rank.json", module_rows)
        plot_module_metrics(charts_dir, module_rows)

    metadata = {
        "bf16_model": args.bf16_model,
        "quant_model": args.quant_model,
        "num_prompts": len(prompts),
        "num_gsm8k": args.num_gsm8k,
        "num_ceval": args.num_ceval,
        "max_length": args.max_length,
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
    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Qwen3-8B BF16 and W8A8 activation distributions.")
    parser.add_argument("--bf16-model", default="/path/to/Qwen3-8B")
    parser.add_argument("--quant-model", default="/path/to/qwen3_8b_int8_w8a8_smoothquant_gptq")
    parser.add_argument("--output-dir", default="/path/to/outputs/qwen3_8b_activation_diagnosis")
    parser.add_argument("--bf16-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--quant-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--num-gsm8k", type=int, default=16)
    parser.add_argument("--num-ceval", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--top-layers", type=int, default=5)
    parser.add_argument("--module-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--probe-load-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.probe_load_only:
        probe_load(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
