## Run Benchmark:
> Note: Run without tps_only=True to avoid tracing overhead.      
> Note: Run with skip_first=1 wait=2 warmup=1 active=1 for small runs (1 flop and quant, 2 compile, 1 warmup, 1 measure).      
> Note: Run with profile_compile=True to get profile trace of compilation process.      
> Note: Run with TORCH_TRACE="log/compile" to generate dynamo logs that can be parsed with tlparse log/compile/* --overwrite. Alternatively, can use TORCH_LOGS="dynamic,guards,recompiles,perf_hints,fusion".      
> Note: Run with oom_profile=True tps_only=False skip_first=0 warmup=0 wait=0 active=1 repeat=5 to get each step's memory and runtime trace.       
> Note: Run with kernel_profile=True tps_only=False to profile forward pass latency at the kernel level.

Example:
```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=False quantization_method=mxfp8_mxfp8 kernel_profile=True oom_profile=True tps_only=False skip_first=0 warmup=0 wait=0 active=1 repeat=5
```

### Basic torch, inductor, and torchao runs:
```
python -m low_bit_inference.torch_baseline configs/profile_baseline.yaml tps_only=True

# sample results on rtx5090
Profiling complete. Metrics: tokens per second (tps): 55.447283977856, time to first token (ttft): 0.017867238616943358, time per output token (tpot): 0.0179350673828125, prefill throughput: 335.8111675423359, decode throughput: 55.75672104025158, latency: 3.6070316406250003, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.8592026798253974 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495847915085209 TFLOP/token, Bandwidth used (tps*model_size): 832.2553932361084 GB/s, Power used in last sample period: 227.366 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 37.5209574520438, time to first token (ttft): 0.028416204452514648, time per output token (tpot): 0.026628995703124998, prefill throughput: 318.0295140910124, decode throughput: 37.56522666445902, latency: 26.660446484375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.5874677316899094 TFLOPs, FLOP per token used (FLOPs/tps): 0.015657055991728418 TFLOP/token, Bandwidth used (tps*model_size): 563.1839282031767 GB/s, Power used in last sample period: 188.481 W.
```

```
python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 83.04209256564624, time to first token (ttft): 0.018208115386962893, time per output token (tpot): 0.011939783447265626, prefill throughput: 329.52784331002465, decode throughput: 83.7536164861968, latency: 2.4084172851562498, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.2868082582453881 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495855396805463 TFLOP/token, Bandwidth used (tps*model_size): 1246.4493198796283 GB/s, Power used in last sample period: 99.97 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 48.71664759078724, time to first token (ttft): 0.028065196990966795, time per output token (tpot): 0.02049889375, prefill throughput: 320.7174129312574, decode throughput: 48.79033158924184, latency: 20.529896484375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.7628948365100735 TFLOPs, FLOP per token used (FLOPs/tps): 0.015659838561107062 TFLOP/token, Bandwidth used (tps*model_size): 731.2295533539188 GB/s, Power used in last sample period: 98.861 W.
```

```
python -m low_bit_inference.torchao_autoquant configs/profile_inductor_torchao.yaml tps_only=True

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 89.01048835430088, time to first token (ttft): 0.11665095062255859, time per output token (tpot): 0.01064035791015625, prefill throughput: 51.43550463946163, decode throughput: 93.9818155752118, latency: 2.2469265136718755, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.3792935518729563 TFLOPs, FLOP per token used (FLOPs/tps): 0.01549585422318729 TFLOP/token, Bandwidth used (tps*model_size): 716.0262084228075 GB/s, Power used in last sample period: 218.853 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 58.16137474427701, time to first token (ttft): 0.34503413085937495, time per output token (tpot): 0.016850888671875, prefill throughput: 26.084390469485204, decode throughput: 59.36347375212815, latency: 17.198946484374996, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.9106460117336168 TFLOPs, FLOP per token used (FLOPs/tps): 0.01565722983229902 TFLOP/token, Bandwidth used (tps*model_size): 437.38728091187875 GB/s, Power used in last sample period: 103.419 W.
```

### TorchAO weight only quantization configs:
```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_int4

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 71.63311748869404, time to first token (ttft): 0.04302642593383789, time per output token (tpot): 0.013956582617187501, prefill throughput: 210.56828982102866, decode throughput: 71.87038832087492, latency: 14.0028458984375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.11367968853371617 TFLOPs, FLOP per token used (FLOPs/tps): 0.001586971117816538 TFLOP/token, Bandwidth used (tps*model_size): 340.89990183423555 GB/s, Power used in last sample period: 145.707 W.
```

```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_int8
```

```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_fp8
# current running job
```

### Gemlite weight only quantization configs:
```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int1
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_fp1.58
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_mxfp4
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_mxfp8

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 66.95707900014932, time to first token (ttft): 0.06158909378051758, time per output token (tpot): 0.014870641796875, prefill throughput: 146.13413610726815, decode throughput: 67.24704871472184, latency: 14.935041015625, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.10658418404439747 TFLOPs, FLOP per token used (FLOPs/tps): 0.001591828461396289 TFLOP/token, Bandwidth used (tps*model_size): 552.3051143799872 GB/s, Power used in last sample period: 122.034 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int4
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int8
```

### Gemlite weight and activation configs:
```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=int8_fp1.58
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=int8_int8
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=mxfp4_mxfp4
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=mxfp8_mxfp8
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=nvfp4_nvfp4
```