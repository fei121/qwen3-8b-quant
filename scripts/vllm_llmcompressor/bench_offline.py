import argparse
import json
import subprocess
import time
from pathlib import Path

from vllm import LLM, SamplingParams


PROMPTS = [
    "Solve carefully: If a shop sold 18 notebooks on Monday and 27 on Tuesday, how many notebooks were sold?",
    "Explain why quantization can reduce model memory while preserving most model quality.",
    "Write three concise deployment checks for a language model server running on one GPU.",
    "A train travels 48 miles in one hour and 36 miles in the next hour. What is the total distance?",
]


def gpu_memory_gb():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return round(max(values) / 1024, 3) if values else None


def repeated_prompts(count):
    return [PROMPTS[index % len(PROMPTS)] for index in range(count)]


def run_generation(llm, prompts, max_tokens):
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    return elapsed, prompt_tokens, output_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--decode-max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    prompts = repeated_prompts(args.num_prompts)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    run_generation(llm, prompts[:4], max_tokens=8)
    memory_samples = [gpu_memory_gb()]

    prefill_elapsed, prefill_prompt_tokens, _ = run_generation(
        llm, prompts, max_tokens=1
    )
    memory_samples.append(gpu_memory_gb())
    decode_elapsed, _, decode_output_tokens = run_generation(
        llm, prompts, max_tokens=args.decode_max_tokens
    )
    memory_samples.append(gpu_memory_gb())

    peak_vram = max(value for value in memory_samples if value is not None)
    payload = {
        "model": args.model,
        "run_name": args.run_name,
        "num_prompts": args.num_prompts,
        "decode_max_tokens": args.decode_max_tokens,
        "peak_vram_gb": peak_vram,
        "prefill_tokens_per_s": round(prefill_prompt_tokens / prefill_elapsed, 3),
        "decode_tokens_per_s": round(decode_output_tokens / decode_elapsed, 3),
        "prefill_elapsed_s": round(prefill_elapsed, 4),
        "decode_elapsed_s": round(decode_elapsed, 4),
    }

    output = Path(args.output_dir) / args.run_name / "offline_bench.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

