## Run Benchmark:
> Note: Run without tps_only=True to get trace of CPU and GPU kernels.      
> Note: Run with skip_first=1 wait=2 warmup=1 active=1 for small runs (1 flop and quant, 2 compile, 1 warmup, 1 measure).      
> Note: Run with profile_compile=True to get profile trace of compilation process.      
> Note: Run with TORCH_TRACE="log/compile" to generate dynamo logs that can be parsed with tlparse log/compile/* --overwrite. Alternatively, can use TORCH_LOGS="dynamic,guards,recompiles,perf_hints,fusion".      

### Basic torch, inductor, and torchao runs:
```
python -m low_bit_inference.torch_baseline configs/profile_baseline.yaml tps_only=True

Profiling complete. Metrics: tokens per second (tps): 37.5209574520438, time to first token (ttft): 0.028416204452514648, time per output token (tpot): 0.026628995703124998, prefill throughput: 318.0295140910124, decode throughput: 37.56522666445902, latency: 26.660446484375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.5874677316899094 TFLOPs, FLOP per token used (FLOPs/tps): 0.015657055991728418 TFLOP/token, Bandwidth used (tps*model_size): 563.1839282031767 GB/s, Power used in last sample period: 188.481 W.
```

```
python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True

Profiling complete. Metrics: tokens per second (tps): 48.71664759078724, time to first token (ttft): 0.028065196990966795, time per output token (tpot): 0.02049889375, prefill throughput: 320.7174129312574, decode throughput: 48.79033158924184, latency: 20.529896484375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.7628948365100735 TFLOPs, FLOP per token used (FLOPs/tps): 0.015659838561107062 TFLOP/token, Bandwidth used (tps*model_size): 731.2295533539188 GB/s, Power used in last sample period: 98.861 W.
```

```
python -m low_bit_inference.torchao_exp.autoquant configs/profile_inductor_torchao.yaml tps_only=True

Profiling complete. Metrics: tokens per second (tps): 58.16137474427701, time to first token (ttft): 0.34503413085937495, time per output token (tpot): 0.016850888671875, prefill throughput: 26.084390469485204, decode throughput: 59.36347375212815, latency: 17.198946484374996, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.9106460117336168 TFLOPs, FLOP per token used (FLOPs/tps): 0.01565722983229902 TFLOP/token, Bandwidth used (tps*model_size): 437.38728091187875 GB/s, Power used in last sample period: 103.419 W.
```

### TorchAO weight only quantization configs:
```
python -m low_bit_inference.torchao_exp.A_bf16_W_int4 configs/profile_inductor_torchao.yaml tps_only=True

Profiling complete. Metrics: tokens per second (tps): 105.58340153538859, time to first token (ttft): 0.03942031326293945, time per output token (tpot): 0.009428716406249999, prefill throughput: 228.47623131299096, decode throughput: 106.06017183709037, latency: 9.47129453125, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.05713749124097591 TFLOPs, FLOP per token used (FLOPs/tps): 0.0005411597884713445 TFLOP/token, Bandwidth used (tps*model_size): 421.0014105290168 GB/s, Power used in last sample period: 111.599 W.
```

```
python -m low_bit_inference.torchao_exp.A_bf16_W_int8 configs/profile_inductor_torchao.yaml tps_only=True
```

```
python -m low_bit_inference.torchao_exp.A_bf16_W_fp8 configs/profile_inductor_torchao.yaml tps_only=True
```

### Gemlite demo (A16W8_HQQ_INT):
```
python -m low_bit_inference.gemlite_demo configs/profile_inductor_gemlite.yaml tps_only=True
```

### Gemlite weight only quantization configs:
```
python -m low_bit_inference.gemlite_exp.A_bf16_W_int1 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_bf16_W_fp1_58 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_bf16_W_mxfp4 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_bf16_W_mxfp8 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_bf16_W_int4 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_bf16_W_int8 configs/profile_inductor_gemlite.yaml tps_only=True
```

### Gemlite weight and activation configs:
```
python -m low_bit_inference.gemlite_exp.A_nvfp4_W_nvfp4 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_mxfp4_W_mxfp4 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_int8_W_int8 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_mxfp8_W_mxfp8 configs/profile_inductor_gemlite.yaml tps_only=True
```

```
python -m low_bit_inference.gemlite_exp.A_int8_W_fp1_58 configs/profile_inductor_gemlite.yaml tps_only=True
```