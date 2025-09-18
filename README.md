# Low Bit Inference:

[![Docker](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml) [![Run on VastAI](https://img.shields.io/badge/Run_on-VastAI-blue)](https://cloud.vast.ai?ref_id=288801&template_id=9b22ab4bd296c4a6f1ce3f6eece5e6b4) [![Run on Runpod](https://img.shields.io/badge/Run_on-Runpod-green)](https://console.runpod.io/deploy?template=q0ucwygekf&ref=9969n21w)

Experimenting with different methods for low bit inference of Llama-3.1.

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

## Benchmark (on 1 RTX 4090):
| Weight Bits | Activation Bits | Throughput (tokens/sec) |
|-------------|-----------------|-------------------------|
| 16 (bf16) | 16 (bf16) | TBD |
| 16 (bf16) | 16 (bf16) | TBD |
| Auto | Auto | TBD |
| 4 | 16 (bf16) | TBD |
| 4 | 4 | TBD |
| 8 | 16 (bf16) | TBD |
| 8 | 8 | TBD |
| 8 | 16 (bf16) | TBD |
| 8 | 8 | TBD |
| 6 | 16 (bf16) | TBD |
| 6 | 6 | TBD |
| 4 | 16 (bf16) | TBD |
| 4 | 4 | TBD |
| 1 | 1 | TBD |
| 1.58 | 1.58 | TBD |

## TorchAO Quantization Configs:
Note: All these are affine transforms available in TorchAO. They are not custom transforms.

quantized_val = input_high_precision_float_val / scale + zero_point

**Functions/Params**:      
quantize_affine: original fp32, fp16, bf16 tensor.      
dequantize_affine: quantized tensor o/p of above.       

choose_qparams_affine: fp32, bf16, fp16 example i/p tensor for calculating scaling factor.

ZeroPointDomain: none, int or float, set dtype of zero point.

Note: Use bf16 model for all affine quants other than fp6.

**INT4**:       
A16W4 WeightOnly Quantization Int4WeightOnlyConfig    
A8W4 Int8DynamicActivationInt4WeightConfig    

**INT8**:       
A16W8 Int8 WeightOnly Quantization Int8WeightOnlyConfig     
A8W8 Int8 Dynamic Quantization Int8DynamicActivationInt8WeightConfig               

**Miscellaneous int4 and int8**:                
GemliteUIntXWeightOnlyConfig

**Float8**:     
A16W8 Float8 WeightOnly Quantization Float8WeightOnlyConfig     
A8W8 Float8 Dynamic Quantization with Tensorwise or Rowwise Scaling Float8DynamicActivationFloat8WeightConfig       
A8W4 Float8 DQ Float8DynamicActivationInt4WeightConfig      
A8W8 Float8StaticActivationFloat8WeightConfig       

**FP6**:        
A16W6 Floating Point WeightOnly Quantization FPXWeightOnlyConfig        

**Limitations of TorchAO Quantization Configs**:
* Only quantizes linear layers, we can quantize non-linear layers as well.
* Static activation quantization seems to be lacking in a lot of dtypes.

## Benchmarking Roadmap:
- [x] Calculate theoretical performance limit: get token/s and multiply it by model size for bandwidth and by model flops for compute util.
- [ ] Use existing TorchAO configs.
- [ ] Quantize non-linear layers which isn't done in TorchAO yet.
- [ ] Enforce static quantization.

## Possible extensions:
- [ ] Megakernel for Llama-3.1-8B: https://github.com/HazyResearch/Megakernels/blob/main/demos/low-latency-llama/llama.cuh , https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- [ ] llama.cpp deployment with our model: https://github.com/ggml-org/llama.cpp.

## Benchmarking Notes:
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
