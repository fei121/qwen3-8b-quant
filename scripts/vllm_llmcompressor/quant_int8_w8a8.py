import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "."))
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_ROOT / "models/qwen3_8b_int8_w8a8"))
NUM_CALIBRATION_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "512"))
MAX_SEQUENCE_LENGTH = int(os.environ.get("MAX_SEQUENCE_LENGTH", "2048"))
SEED = int(os.environ.get("SEED", "42"))
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"


def build_calibration_dataset(tokenizer):
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_CALIBRATION_SAMPLES))

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

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
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
    print(f"saved INT8 W8A8 model to {OUT_DIR}")


if __name__ == "__main__":
    main()
