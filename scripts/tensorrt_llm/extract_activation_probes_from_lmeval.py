#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def first_prompt_from_arguments(arguments):
    if not isinstance(arguments, dict):
        return None
    for value in arguments.values():
        if isinstance(value, dict) and isinstance(value.get("arg_0"), str):
            return value["arg_0"]
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return None


def collect_from_patterns(patterns, limit, dataset_name):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if len(rows) >= limit:
                        return rows
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    prompt = first_prompt_from_arguments(item.get("arguments"))
                    if not prompt:
                        continue
                    rows.append(
                        {
                            "id": f"{dataset_name}_{len(rows)}",
                            "dataset": dataset_name,
                            "source_file": path,
                            "doc_id": item.get("doc_id"),
                            "text": prompt,
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gsm8k", type=int, default=16)
    parser.add_argument("--ceval", type=int, default=16)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    gsm8k = collect_from_patterns(
        [str(eval_root / "gsm8k_train_full" / "**" / "samples_gsm8k*.jsonl")],
        args.gsm8k,
        "gsm8k",
    )
    ceval = collect_from_patterns(
        [str(eval_root / "ceval_dev_full" / "**" / "samples_ceval-valid*.jsonl")],
        args.ceval,
        "ceval",
    )
    rows = gsm8k + ceval
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row["probe_index"] = index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "gsm8k": len(gsm8k), "ceval": len(ceval), "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
