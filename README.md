# low-bit-vllm

[![Docker](https://github.com/vipulSharma18/low-bit-vllm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/vipulSharma18/low-bit-vllm/actions/workflows/docker-publish.yml) [![Run on VastAI](https://img.shields.io/badge/Run_on-VastAI-blue)](https://cloud.vast.ai?ref_id=288801&template_id=bc0609fee288cad6d15b1262dbc83214)

Experimenting with different methods for low bit inference of Llama-3.1 with vLLM support.

GitHub setup within the container:
```
gh auth login
git config --global user.email "vipuls181999@gmail.com"
git config --global user.name "Vipul Sharma"
```

## Benchmarking:
Run with CUDA_LAUNCH_BLOCKING=1 to make GPU-CPU sync after each kernel, to get more interpretable profiling results for each kernel.

(Ignore) Other results from sekstini on EleutherAI discord (Llama-7b):

fp16 + compile: 69.04 tokens/sec, 930.65 GB/s       
fp16 + compile + TP=2 (NCCL_P2P_DISABLE=1): 113.26 tokens/sec, 792.91 GB/s      
fp16 + compile + compile prefill + int4 (G=32) draft model: 112.53 tokens/s, 1592.94 GB/s (lol)     

int8 + compile: 126.07 tokens/s, 866.56 GB/s        
int8 + compile + TP=2 (NCCL_P2P_DISABLE=1): 183.31 tokens/sec, 666.53 GB/s      

int4 (G=32) + compile: 189.76 tokens/s, 833.44 GB/s     
int4 (G=32) + compile + TP=2 (NCCL_P2P_DISABLE=1): 218.99 tokens/s, 518.64 GB/s     
int4 (G=32) + compile + compile prefill + int4 (G=32) draft model: 215.25 tokens/s, 929.95 GB/s (bandwidth prob incorrect?)     

pytorch-triton==2.1.0+bcad9dabe1        
torch==2.2.0.dev20231205+cu121      

fp16 + compile: 69.04 tokens/sec, 930.65 GB/s       
fp16 + compile + TP=2 (NCCL_P2P_DISABLE=1): 113.26 tokens/sec, 792.91 GB/s      
fp16 + compile + compile prefill + int4 (G=32) draft model: 112.53 tokens/s, 1592.94 GB/s (lol)     

int8 + compile: 126.07 tokens/s, 866.56 GB/s        
int8 + compile + TP=2 (NCCL_P2P_DISABLE=1): 183.31 tokens/sec, 666.53 GB/s      

int4 (G=32) + compile: 189.76 tokens/s, 833.44 GB/s     
int4 (G=32) + compile + TP=2 (NCCL_P2P_DISABLE=1): 218.99 tokens/s, 518.64 GB/s     
int4 (G=32) + compile + compile prefill + int4 (G=32) draft model: 215.25 tokens/s, 929.95 GB/s (bandwidth prob incorrect?)     

rtx 4090 (1000 GB/s mem/bw)     
pytorch-triton==2.1.0+bcad9dabe1        
torch==2.2.0.dev20231205+cu121      

## References:
[1] A. Hoque, L. Wright, C.-C. Yang, M. Srivatsa, and R. Ganti, “Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition,” Feb. 22, 2024, arXiv: arXiv:2402.00025. doi: 10.48550/arXiv.2402.00025.    
[2] “Accelerating LLM Inference with GemLite, TorchAO and SGLang – PyTorch.” Accessed: Aug. 19, 2025. [Online]. Available: https://pytorch.org/blog/accelerating-llm-inference/    
[3] M. Dehghankar, M. Erfanian, and A. Asudeh, “An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks,” May 02, 2025, arXiv: arXiv:2411.06360. doi: 10.48550/arXiv.2411.06360.    
[4] S. Bekman, stas00/ml-engineering. (Aug. 20, 2025). Python. Accessed: Aug. 20, 2025. [Online]. Available: https://github.com/stas00/ml-engineering    
