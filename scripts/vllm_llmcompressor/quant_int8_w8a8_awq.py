import os
import random
from pathlib import Path

from datasets import Dataset, get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "."))
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_ROOT / "models/qwen3_8b_int8_awq_w8a16"))
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "2048"))
SEED = int(os.environ.get("SEED", "42"))
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
CALIBRATION_DATASET = os.environ.get("CALIBRATION_DATASET", "gsm8k").lower()


def tokenize_text_dataset(dataset, tokenizer):
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def build_gsm8k_calibration_dataset(tokenizer):
    dataset = load_dataset("gsm8k", "main", split="train")
    sample_count = min(NUM_CALIBRATION_SAMPLES, len(dataset))
    dataset = dataset.shuffle(seed=SEED).select(range(sample_count))

    def format_sample(example):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Solve the math problem step by step.",
            },
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    return tokenize_text_dataset(dataset, tokenizer)


def format_ceval_question(item):
    return (
        f"{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        "答案："
    )


def build_ceval_calibration_dataset(tokenizer):
    rows = []
    for subject in get_dataset_config_names("ceval/ceval-exam"):
        dataset = load_dataset("ceval/ceval-exam", subject, split="dev")
        for item in dataset:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个中文考试助手。请根据题目选择唯一正确答案。",
                },
                {"role": "user", "content": format_ceval_question(item) + str(item["answer"])},
            ]
            rows.append(
                {
                    "text": tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                }
            )

    random.Random(SEED).shuffle(rows)
    rows = rows[:NUM_CALIBRATION_SAMPLES]
    return tokenize_text_dataset(Dataset.from_list(rows), tokenizer)


def build_calibration_dataset(tokenizer):
    if CALIBRATION_DATASET == "gsm8k":
        return build_gsm8k_calibration_dataset(tokenizer)
    if CALIBRATION_DATASET == "ceval":
        return build_ceval_calibration_dataset(tokenizer)
    raise ValueError("CALIBRATION_DATASET must be one of: gsm8k, ceval")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"using calibration dataset: {CALIBRATION_DATASET}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )

    recipe = [
        AWQModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": False,
                        "strategy": "group",
                        "group_size": 128,
                    },
                }
            },
            ignore=["lm_head"],
        )
    ]
    oneshot(
        model=model,
        dataset=build_calibration_dataset(tokenizer),
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved AWQ-calibrated INT8 W8A16 model to {OUT_DIR}")


if __name__ == "__main__":
    main()
