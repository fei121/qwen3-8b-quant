import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "."))
OUT_DIR = Path(os.environ.get("OUT_DIR", DATA_ROOT / "models/qwen3_8b_mxfp4a16"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        trust_remote_code=True,
    )
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="MXFP4A16",
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved MXFP4A16 model to {OUT_DIR}")


if __name__ == "__main__":
    main()

