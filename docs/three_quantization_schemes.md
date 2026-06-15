# Qwen3-8B 三类量化方案对比

本文整理 Qwen3-8B 在三条工程链路上的量化结果：

1. **Intel AutoRound INT8 W8A8**：由 AutoRound 导出，使用 vLLM 评测。
2. **Intel AutoRound MXFP4 fakequant**：作为低比特精度参考，不作为真实部署速度结论。
3. **NVIDIA TensorRT-LLM INT8 SmoothQuant**：通过 ModelOpt 导出，并以 TensorRT-LLM engine 评测。

同时加入两个 BF16 参照：

- vLLM BF16 baseline。
- TensorRT-LLM BF16 engine baseline。

所有 GSM8K 和 C-Eval 结果均为 5-shot。GSM8K 报告 `exact_match,flexible-extract`，C-Eval valid 报告 `acc`。

## 精度总览

| 生态 | 方法 | 精度格式 | GSM8K flexible | C-Eval acc | 说明 |
|---|---|---:|---:|---:|---|
| vLLM | BF16 baseline | BF16 | 0.8802 | 0.7905 | vLLM 全精度参照 |
| TensorRT-LLM | BF16 baseline | BF16 | 0.8848 | 0.7853 | TensorRT-LLM engine 参照 |
| AutoRound | AutoRound | INT8 W8A8 | 0.8749 | 0.7764 | 精度接近 vLLM BF16 |
| AutoRound | AutoRound | MXFP4 fakequant | 0.8613 | 0.7667 | fakequant，只看精度，不看真实速度 |
| TensorRT-LLM | ModelOpt SmoothQuant | INT8 W8A8 | 0.7983 | 0.6872 | 当前配置下掉点明显 |

补充的 vLLM 生态结果：

| 生态 | 方法 | 精度格式 | GSM8K flexible | C-Eval acc |
|---|---|---:|---:|---:|
| vLLM | LLM Compressor SmoothQuant + GPTQ | INT8 W8A8 | 0.8719 | 0.7853 |
| vLLM | LLM Compressor MXFP4A16 | MXFP4A16 | 0.8643 | 0.7608 |

## 相对 BF16 的变化

| 方法 | 对比基线 | GSM8K 变化 | C-Eval 变化 |
|---|---|---:|---:|
| AutoRound INT8 W8A8 | vLLM BF16 | -0.0053 | -0.0141 |
| AutoRound MXFP4 fakequant | vLLM BF16 | -0.0190 | -0.0238 |
| LLM Compressor INT8 W8A8 | vLLM BF16 | -0.0083 | -0.0052 |
| LLM Compressor MXFP4A16 | vLLM BF16 | -0.0159 | -0.0297 |
| TensorRT-LLM INT8 SmoothQuant | TensorRT-LLM BF16 | -0.0864 | -0.0981 |

AutoRound INT8 和 LLM Compressor INT8 都能较好保持精度。TensorRT-LLM INT8 SmoothQuant 在本项目测试配置下有同 workflow 内的速度收益，但精度损失明显更大，因此需要后续根因分析。

## 性能数据如何解读

本项目收集了两类速度数据：

- vLLM offline / serve benchmark，适合看 vLLM 生态内的吞吐能力。
- lm-eval / TensorRT-LLM wrapper 的端到端耗时，包含评测框架、调度和数据处理开销。

这两类数据不能混成同一个 benchmark 直接比较。对 TensorRT-LLM 更可靠的说法是：在同一 TensorRT-LLM workflow 内，INT8 SmoothQuant 比 BF16 engine 更快；但这不等价于它必然快于 vLLM。

## 结论

- **INT8 精度保持最好的一组结果**：AutoRound INT8 与 vLLM LLM Compressor INT8。
- **低比特参考价值最高的结果**：AutoRound MXFP4 fakequant 与 vLLM MXFP4A16，二者说明低比特方向值得继续探索；其中只有 vLLM MXFP4A16 是真实部署形态的 benchmark。
- **最重要的异常现象**：TensorRT-LLM INT8 SmoothQuant 掉点远超其他 INT8 链路，因此引出了 [`tensorrt_llm_int8_down_proj_analysis.md`](tensorrt_llm_int8_down_proj_analysis.md) 中的 `mlp.down_proj` 根因分析。

结构化数据位于 [`../results/qwen3_8b_quantization_summary.csv`](../results/qwen3_8b_quantization_summary.csv)。
