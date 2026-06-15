import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch


PACKAGES = ("torch", "transformers", "vllm", "llmcompressor", "lm_eval")


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def main():
    outputs = Path("outputs/env")
    outputs.mkdir(parents=True, exist_ok=True)

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "packages": {name: version(name) for name in PACKAGES},
        "torch_cuda": torch.version.cuda,
        "nvidia_smi": nvidia_smi(),
    }

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        payload["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": round(prop.total_memory / 1024**3, 2),
        }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    with (outputs / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())

