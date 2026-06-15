#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from transformers import AutoTokenizer


def format_question(item):
    return (
        f"{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        "答案："
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
    )
    subjects = get_dataset_config_names("ceval/ceval-exam")
    rows = []
    for subject in subjects:
        dataset = load_dataset("ceval/ceval-exam", subject, split=args.split)
        for item in dataset:
            content = format_question(item) + str(item["answer"])
            messages = [
                {
                    "role": "system",
                    "content": "你是一个中文考试助手。请根据题目选择唯一正确答案。",
                },
                {"role": "user", "content": content},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            rows.append({"text": text, "subject": subject})

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    metadata = {
        "source": "ceval/ceval-exam",
        "split": args.split,
        "seed": args.seed,
        "limit": args.limit,
        "size": len(rows),
        "subjects": len(subjects),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
