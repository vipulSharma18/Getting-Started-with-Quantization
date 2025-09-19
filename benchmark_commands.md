## Run Benchmark:
> Note: Run without tps_only=True to get trace of CPU and GPU kernels.      
> Note: Run with skip_first=1 wait=2 warmup=1 active=1 for small runs (1 flop and quant, 2 compile, 1 warmup, 1 measure).      
> Note: Run with profile_compile=True to get profile trace of compilation process.      
> Note: Run with TORCH_TRACE="log/compile" to generate dynamo logs that can be parsed with tlparse log/compile/* --overwrite. Alternatively, can use TORCH_LOGS="dynamic,guards,recompiles,perf_hints,fusion".      

### Basic torch, inductor, and torchao runs:
```
python -m low_bit_inference.torch_baseline configs/profile_baseline.yaml tps_only=True
python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True
```

```
python -m low_bit_inference.torchao.autoquant configs/profile_inductor_torchao.yaml tps_only=True
Profiling complete. Metrics: tokens per second (tps): 58.16137474427701, time to first token (ttft): 0.34503413085937495, time per output token (tpot): 0.016850888671875, prefill throughput: 26.084390469485204, decode throughput: 59.36347375212815, latency: 17.198946484374996, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.9106460117336168 TFLOPs, FLOP per token used (FLOPs/tps): 0.01565722983229902 TFLOP/token, Bandwidth used (tps*model_size): 437.38728091187875 GB/s, Power used in last sample period: 103.419 W.
```

### TorchAO weight only configs:
```
python -m low_bit_inference.torchao.int4wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.torchao.int8wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.torchao.fp8wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.torchao.fp6wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.torchao.fp4wo configs/profile_inductor_torchao.yaml tps_only=True
```

### Gemlite weight and activation configs:
```
python -m low_bit_inference.gemlite.int4_full configs/profile_inductor_gemlite.yaml tps_only=True
python -m low_bit_inference.gemlite.int8_full configs/profile_inductor_gemlite.yaml tps_only=True
python -m low_bit_inference.gemlite.fp8_full configs/profile_inductor_gemlite.yaml tps_only=True
python -m low_bit_inference.gemlite.fp6_full configs/profile_inductor_gemlite.yaml tps_only=True
python -m low_bit_inference.gemlite.fp4_full configs/profile_inductor_gemlite.yaml tps_only=True
```

### Gemlite extreme-low-bit weights and activation configs:
```
python -m low_bit_inference.gemlite.fp1_full configs/profile_inductor_gemlite.yaml tps_only=True
python -m low_bit_inference.gemlite.fp1_58_full configs/profile_inductor_gemlite.yaml tps_only=True
```

### Custom triton extreme-low-bit weights and activation configs:
```
python -m low_bit_inference.extreme_bit_custom.fp1_full configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.extreme_bit_custom.fp1_58_full configs/profile_inductor_torchao.yaml tps_only=True
```