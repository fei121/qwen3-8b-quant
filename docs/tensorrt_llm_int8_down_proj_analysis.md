# TensorRT-LLM INT8 `mlp.down_proj` Root-Cause Analysis

This report summarizes the follow-up study for the TensorRT-LLM / ModelOpt INT8 SmoothQuant accuracy drop on Qwen3-8B.

## Experiment Setup

- Model: Qwen3-8B.
- Backend: TensorRT-LLM 1.0.0.
- Quantization: ModelOpt INT8 SmoothQuant.
- Main ablation variable: keep the first N layers of `mlp.down_proj` in BF16 while other linear modules remain INT8.
- C-Eval calibration: full C-Eval dev set, 260 examples.
- GSM8K calibration: full GSM8K train set, 7473 examples.
- C-Eval metric: `acc`.
- GSM8K metric: `exact_match,flexible-extract`.

## Why C-Eval and GSM8K Differ

C-Eval is a multiple-choice log-likelihood ranking task. It is highly sensitive to small perturbations that flip the ordering of A/B/C/D options, especially when the margin between choices is small.

GSM8K is a generation task. The model must produce a reasoning trace and final numeric answer. Quantization errors can accumulate across generation steps, but alternative reasoning text can still be accepted if the final extracted answer is correct. This makes the GSM8K curve noisier than the C-Eval curve.

## Why `mlp.down_proj` Matters

![Transformer MLP residual mechanism](../assets/figures/mlp_residual_mechanism.png)

Qwen3-8B uses a gated MLP/SwiGLU structure. `gate_proj` and `up_proj` expand hidden states from 4096 to 12288 dimensions, apply gating, and then `down_proj` projects back to 4096 dimensions. That projection is the MLP branch's write-back point into the residual stream.

The `down_proj` input has already passed through nonlinear gating and elementwise multiplication, so its dynamic range can contain large activation outliers. If W8A8 quantization error is high at `down_proj`, the error is injected directly into the residual stream and propagates into later layers.

## C-Eval Cumulative Ablation

![C-Eval cumulative down_proj](../assets/figures/ceval_cumulative_down_proj.svg)

| Control / best point | C-Eval acc |
|---|---:|
| INT8 SmoothQuant baseline | 0.687221 |
| TensorRT-LLM BF16 baseline | 0.785300 |
| Best: first 34 layers `mlp.down_proj` kept BF16 | 0.787519 |
| All 36 layers `mlp.down_proj` kept BF16 | 0.783804 |

C-Eval recovers steadily as more early `mlp.down_proj` modules remain BF16. After around 30 layers, the score reaches BF16-level accuracy.

## GSM8K Cumulative Ablation

![GSM8K cumulative down_proj](../assets/figures/gsm8k_cumulative_down_proj.svg)

| Control / best point | GSM8K flexible |
|---|---:|
| INT8 SmoothQuant baseline | 0.827142 |
| TensorRT-LLM BF16 baseline | 0.884800 |
| Best: first 29 layers `mlp.down_proj` kept BF16 | 0.881729 |
| All 36 layers `mlp.down_proj` kept BF16 | 0.877938 |

GSM8K also recovers, but the curve is less monotonic. The best point is first 29 layers, very close to the BF16 baseline.

## Supporting Diagnostics

Activation diagnostics initially highlighted layers 16-21 as drift-heavy. That was a useful clue, but whole-layer skips over those layers only recovered a small fraction of C-Eval accuracy. Module-level ablation showed that:

- Keeping all `mlp.down_proj` modules BF16 recovered C-Eval to `0.783804`.
- Keeping all `self_attn.o_proj` modules BF16 only reached `0.690936`.
- Keeping early `mlp.down_proj + self_attn.o_proj` modules BF16 recovered much more accuracy than skipping layers 16-21 alone.

![Layer error heatmap](../assets/figures/layer_error_heatmap.png)

![Layer cosine](../assets/figures/layer_cosine.png)

## Conclusion

1. Cumulative `mlp.down_proj` skips explain most TensorRT-LLM INT8 SmoothQuant accuracy loss in this setup.
2. The issue is not primarily `self_attn.o_proj`.
3. Layers 16-21 are where drift is very visible, but early `mlp.down_proj` quantization appears to be the more important source of accumulated error.
4. A practical next step is a mixed-precision TensorRT-LLM recipe that keeps selected `mlp.down_proj` modules high precision while quantizing the rest of the model.

Structured ablation data lives in [`../results/tensorrt_llm_down_proj_ablation.csv`](../results/tensorrt_llm_down_proj_ablation.csv).
