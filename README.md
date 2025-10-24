# Low Bit Inference:

[![Docker](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml) [![Run on VastAI](https://img.shields.io/badge/Run_on-VastAI-blue)](https://cloud.vast.ai?ref_id=288801&template_id=9b22ab4bd296c4a6f1ce3f6eece5e6b4) [![Run on Runpod](https://img.shields.io/badge/Run_on-Runpod-green)](https://console.runpod.io/deploy?template=q0ucwygekf&ref=9969n21w)

A survey of various quantization formats, such as MXFP8 and NVFP4, and contemporary tools used for quantization, including TorchAO and GemLite, with inference of Llama-3.1 as an example.

## Project Summary & Presentation:
* Survey of quantization presentation slides: https://docs.google.com/presentation/d/1fEeao2TyFgooLXeNd0r6hLvC93czzdQLRbBAVWHddCQ/edit?usp=sharing
* Recording from the Eleuther AI ML Performance Reading Group: https://www.youtube.com/watch?v=NpQv0R0w_qY 

## Benchmarking Results (on 1 RTX 4090):
| Library | Weight Bits | Activation Bits | Throughput (tokens/sec) |
|-------------|-------------|-----------------|-------------------------|
| Torch-Eager | bf16 | bf16 | 37.52 |
| Torch-Compile | bf16 | bf16 | 48.71 |
| TorchAO | Autoquant | bf16 | 58.16 |
| TorchAO | int4 | bf16 | 105.58 |
| TorchAO | int8 | bf16 | TBD |
| TorchAO | mxfp8 | bf16 | TBD |
| TorchAO | fp8 | bf16 | TBD |
| TorchAO-QuantLLM | fp6 | bf16 | TBD |
| GemLite | int4 | int4 | TBD |
| GemLite | int8 | int8 | TBD |
| GemLite | mxfp8 | mxfp8 | TBD |
| GemLite | mxfp6 | mxfp6 | TBD |
| GemLite | nvfp4 | bf16 | TBD |
| GemLite | nvfp4 | nvfp4 | TBD |
| GemLite | uint1 (bitnet) | uint1 | TBD |

## Benchmarking Roadmap:
- [x] Calculate theoretical performance limit: get token/s and multiply it by model size for bandwidth and by model flops for compute util.
- [x] Use existing TorchAO configs. TorchAO does linear layer quant only, further, we use weights only quant from torchao.
- [ ] torchao weights only config cause cuda oom when doing tps only profiling for more than 1 iteration. For some dtypes, if we enable quantization of lm_head, then it ooms even on the first profiling iteration.
- [ ] Gemlite weights and activations quant.

## Setup:
Manual Docker pull and run if not using VastAI or RunPod:
```
docker pull ghcr.io/vipulsharma18/low-bit-inference:main
docker run --gpus all -d ghcr.io/vipulsharma18/low-bit-inference:main
```

GitHub setup within the container:
```
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
```

Toy config for quick testing:
```
import torch
from low_bit_inference.utils.config_utils import get_config
from low_bit_inference.hf_loader import load_model_tokenizer_prompt_cache
config = get_config()
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
tokenized_prompt = tokenizer([config.prompt], return_tensors="pt").to(config.device)
```

> Note: .vscode folder has a launch.json file with different debugging and testing launch configurations for easy use.

## TorchAO Quantization Notes:
> Note: All of these are **affine quantization** schemes.               
> quantized_val = input_high_precision_float_val / scale + zero_point

**Summary of TorchAO Functions**:      
* quantize_affine: original fp32, fp16, bf16 tensor.      
* dequantize_affine: quantized tensor o/p of above.       
* choose_qparams_affine: fp32, bf16, fp16 example i/p tensor for calculating scaling factor.                
* ZeroPointDomain: none, int or float, set dtype of zero point.             

> Note: Use bf16 model for all affine quants other than fp6, which works better with fp16.

**Different Quantization Configs**:
* **INT4**: A16W4 WeightOnly Quantization Int4WeightOnlyConfig, A8W4 Int8DynamicActivationInt4WeightConfig                  
* **INT8**: A16W8 Int8 WeightOnly Quantization Int8WeightOnlyConfig, A8W8 Int8 Dynamic Quantization, Int8DynamicActivationInt8WeightConfig                  
* **Miscellaneous int4 and int8**: GemliteUIntXWeightOnlyConfig
* **Float8**: A16W8 Float8 WeightOnly Quantization Float8WeightOnlyConfig, A8W8 Float8 Dynamic Quantization with: Tensorwise or Rowwise Scaling Float8DynamicActivationFloat8WeightConfig, A8W4 Float8 DQ, Float8DynamicActivationInt4WeightConfig, A8W8 Float8StaticActivationFloat8WeightConfig           
* **FP6**: A16W6 Floating Point WeightOnly Quantization FPXWeightOnlyConfig        

**Limitations of TorchAO Quantization Configs**:
* Only quantizes linear layers, we can quantize non-linear layers as well.
* Static activation quantization seems to be lacking in a lot of dtypes.

## Gemlite Quantization Notes:
The main API for GemLite is the gemlite.GemLiteLinear object. \
It supports weight quantization bitwidth from 8 to 1 (in powers of 2), and \
groupsize that are divisible by 32. \
The input (activations) can be 32, 16, or 8 bits, and the accumulation can be done in 32, 16, or 8 bits. \
The scale is allowed to be 16 or 32 bit floats.

At its core, it's an affine quantization scheme. We use the above created object to pack our weights \
along with pre-calculated scales and zero points. \
The gemlite_linear object can then be used just like a torch linear layer.

Similar to TorchAO, GemLite provides different quantization configs in the gemlite.helper module:               
* Weight only quantization supports MXFP8, MXFP4, NVFP4, INT8, FP8. Examples: A16W8, A16W4, A16W2, A16W1, A16W158.
* Activation and weight quantization: A8W8, A8W4, A8W2, A8W1, A8W158.
* 4 bit activation is also supported with configs like A4W8, and A4W4.
* Actual quantization options:               
    * **A16**: A16W8, A16Wn, A16W8_INT, A16Wn_HQQ_INT, A16W8_HQQ_INT, A16W4_HQQ_INT, A16W2_HQQ_INT, A16W1_HQQ_INT, A16Wn_MXFP, A16W8_MXFP, A16W4_MXFP
    * **A8**: A8W8_dynamic, A8W8_int8_dynamic, A8W8_INT8_dynamic, A8W8_fp8_dynamic, A8W8_FP8_dynamic, A8Wn_HQQ_INT_dynamic, A8W4_HQQ_INT_dynamic, A8W2_HQQ_INT_dynamic, A8W8_MXFP_dynamic, A8Wn_MXFP_dynamic, A8W8_MXFP_dynamic, A8W4_MXFP_dynamic
    * **A4**: A4W4_MXFP_dynamic, A4W4_NVFP_dynamic
    * **BitNet**: A16W158_INT, A8W158_INT_dynamic

**Limitations**:                
* Just like TorchAO, GemLite focuses on linear layers. The current widespread belief in the quantization community is that non-linear layers don't behave well after quantization and adversely effect model performance. This leads to most work focusing on linear layers.

## Possible extensions:
- [ ] Megakernel for Llama-3.1-8B: https://github.com/HazyResearch/Megakernels/blob/main/demos/low-latency-llama/llama.cuh , https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- [ ] llama.cpp deployment with our model: https://github.com/ggml-org/llama.cpp.
- [ ] FP1.58 kernel from microsoft bitblas: https://github.com/microsoft/BitBLAS
- [ ] FP1.58 with kernel based on “An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks,”.
- [ ] A low-bit Megakernel for FP1.58

## General Benchmarking Notes:
* **Metrics**:
    * TPS: Tokens/sec or Throughput = #decode tokens/(prefill+decode time)
    * TTFT: Time to first token = prefill time
    * TPOT: Time per output token = decode time/#decode tokens
    * Prefill Throughput: #prefill tokens/prefill time
    * Decode Throughput: #decode tokens/decode time
    * Latency: prefill + decode time

* **Compiler Mode and Options Map**: 'default': {}, 'reduce-overhead': {'triton,cudagraphs': True}, 'max-autotune-no-cudagraphs': {'max_autotune': True, 'coordinate_descent_tuning': True}, 'max-autotune': {'max_autotune': True, 'triton.cudagraphs': True, 'coordinate_descent_tuning': True}

* **Prefill Compilation**: Since we have a known prompt length, we're doing compilation for prefill stage as well. In practice, we'd do compile with different prompt lengths before serving to ensure compile cache is hit. Such issues don't occur in the decode stage as the input is always 1 token long (with a static KV cache, i.e., the KV don't change length and the query is 1 length).

* **Decoding**: HF by default uses greedy decoding but we can do speculative decoding, and structured/guided generation to speed-up generation at the cost of VRAM size and memory access.

* **Tokenizer**: The tokenizer should be a Rust-based implementation, not python. HF-Transformers' AutoTokenizer automatically prefers a Rust based implementation and falls back to Python if Rust implementation not availble. But for a new model, we'll need to create our own Rust-based implementation.

* **Profiling interpretability**: Run with CUDA_LAUNCH_BLOCKING=1 to make GPU-CPU sync after each kernel, to get more interpretable profiling results for each kernel.

## References:
[1] A. Hoque, L. Wright, C.-C. Yang, M. Srivatsa, and R. Ganti, “Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition,” Feb. 22, 2024, arXiv: arXiv:2402.00025. doi: 10.48550/arXiv.2402.00025.    
[2] “Accelerating LLM Inference with GemLite, TorchAO and SGLang – PyTorch.” Accessed: Aug. 19, 2025. [Online]. Available: https://pytorch.org/blog/accelerating-llm-inference/    
[3] M. Dehghankar, M. Erfanian, and A. Asudeh, “An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks,” May 02, 2025, arXiv: arXiv:2411.06360. doi: 10.48550/arXiv.2411.06360.    
[4] S. Bekman, stas00/ml-engineering. (Aug. 20, 2025). Python. Accessed: Aug. 20, 2025. [Online]. Available: https://github.com/stas00/ml-engineering          
[5] https://github.com/meta-pytorch/gpt-fast                    
[6] https://pytorch.org/blog/accelerating-generative-ai-2/                     
[7] https://huggingface.co/blog/kv-cache      
[8] https://github.com/pytorch/pytorch/issues/157950        
[9] Benchmarks on Llama-3.1-8B done by torchao team: https://github.com/pytorch/ao/tree/main/torchao/_models/llama       
[10] S. Salaria, Z. Liu, and N. M. Gonzalez, “Meta-Metrics and Best Practices for System-Level Inference Performance Benchmarking,” Aug. 14, 2025, arXiv: arXiv:2508.10251. doi: 10.48550/arXiv.2508.10251.          
[11] Z. Zhou et al., “A Survey on Efficient Inference for Large Language Models,” July 19, 2024, arXiv: arXiv:2404.14294. doi: 10.48550/arXiv.2404.14294.            
[12] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A Survey of Quantization Methods for Efficient Neural Network Inference,” June 21, 2021, arXiv: arXiv:2103.13630. doi: 10.48550/arXiv.2103.13630.            
[13] https://github.com/microsoft/BitBLAS            
