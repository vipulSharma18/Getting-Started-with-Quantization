# low-bit-inference

[![Docker](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/vipulSharma18/low-bit-inference/actions/workflows/docker-publish.yml) [![Run on VastAI](https://img.shields.io/badge/Run_on-VastAI-blue)](https://cloud.vast.ai?ref_id=288801&template_id=bc0609fee288cad6d15b1262dbc83214) [![Run on Runpod](https://img.shields.io/badge/Run_on-Runpod-green)](https://console.runpod.io/deploy?template=q0ucwygekf&ref=9969n21w)

Experimenting with different methods for low bit inference of Llama-3.1.

GitHub setup within the container:
```
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
```

## Run benchmark:
> Note: Run without tps_only=True to get trace of CPU and GPU kernels.

**Baseline result (I think HF does compile by default)**: 

47.90370070952305 tokens per second
```
python -m low_bit_inference.baseline configs/profile_baseline.yaml tps_only=True
```

**Full graph torch compile with inductor**: 

Only compiling model.forward function: 48.54674292202595 tokens per second
```
python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True
```

**Torch compile with custom inductor configs**: 

Only compiling model.forward function: 48.59706179755857 tokens per second
```
python -m low_bit_inference.torchinductor_tuned configs/profile_inductor.yaml tps_only=True
```

**Torch compile with TorchAO AutoQuant**: 
```
python -m low_bit_inference.torchinductor_autoquant configs/profile_inductor.yaml tps_only=True
```

**Torch compile with TorchAO Int4**: 
```
python -m low_bit_inference.torchinductor_int4 configs/profile_inductor.yaml tps_only=True
```

**Torch compile with TorchAO Int8**: 
```
python -m low_bit_inference.torchinductor_int8 configs/profile_inductor.yaml tps_only=True
```

**Torch compile with TorchAO FP8**: 
```
python -m low_bit_inference.torchinductor_fp8 configs/profile_inductor.yaml tps_only=True
```

## Benchmarking Roadmap:
- [ ] Calculate theoretical performance limit and roofline model for Llama-3.1 8B to have a target - HTA will give us the MFU that we can try to maximize.
- [ ] Megakernel for Llama-3.1-8B: https://github.com/HazyResearch/Megakernels/blob/main/demos/low-latency-llama/llama.cuh , https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- [ ] HQQ is the fastest dynamic/on-the-fly quantization out there. https://mobiusml.github.io/hqq_blog/

## Benchmarking Notes:
* **Prefill Compilation**: Since we have a known prompt length, we're doing compilation for prefill stage as well. In practice, we'd do compile with different prompt lengths before serving to ensure compile cache is hit. Such issues don't occur in the decode stage as the input is always 1 token long (with a static KV cache, i.e., the KV don't change length and the query is 1 length).

* **Decoding**: HF by default uses greedy decoding but we can do speculative decoding, and structured/guided generation to speed-up generation.

* **KV Cache**: Need to make the KV Cache static/constant shape to allow for torch.compile to work. For our custom model, we'll need to do create custom static cache, for now using HF.

* **Tokenizer**: The tokenizer should be a Rust-based implementation, not python. HF-Transformers' AutoTokenizer automatically prefers a Rust based implementation and falls back to Python if Rust implementation not availble. But for a new model, we'll need to create our own Rust-based implementation.

* **Profiling interpretability**: Run with CUDA_LAUNCH_BLOCKING=1 to make GPU-CPU sync after each kernel, to get more interpretable profiling results for each kernel.

* Carson Poole's tweet about what one needs to optimize: https://x.com/CarsonPoole/status/1843751758331613573

## References:
[1] A. Hoque, L. Wright, C.-C. Yang, M. Srivatsa, and R. Ganti, “Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition,” Feb. 22, 2024, arXiv: arXiv:2402.00025. doi: 10.48550/arXiv.2402.00025.    
[2] “Accelerating LLM Inference with GemLite, TorchAO and SGLang – PyTorch.” Accessed: Aug. 19, 2025. [Online]. Available: https://pytorch.org/blog/accelerating-llm-inference/    
[3] M. Dehghankar, M. Erfanian, and A. Asudeh, “An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks,” May 02, 2025, arXiv: arXiv:2411.06360. doi: 10.48550/arXiv.2411.06360.    
[4] S. Bekman, stas00/ml-engineering. (Aug. 20, 2025). Python. Accessed: Aug. 20, 2025. [Online]. Available: https://github.com/stas00/ml-engineering    
