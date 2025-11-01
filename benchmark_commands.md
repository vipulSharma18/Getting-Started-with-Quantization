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
Profiling complete. Metrics: tokens per second (tps): 35.828349371750974, time to first token (ttft): 0.029532710647583005, time per output token (tpot): 0.02782305859375, prefill throughput: 205.64202481693366, decode throughput: 36.040108274803146, latency: 5.597216308593749, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.553698674657553 TFLOPs, FLOP per token used (FLOPs/tps): 0.015454205520674062 TFLOP/token, Bandwidth used (tps*model_size): 537.7781354862367 GB/s, Power used in last sample period: 244.966 W.
```

```
python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 83.04209256564624, time to first token (ttft): 0.018208115386962893, time per output token (tpot): 0.011939783447265626, prefill throughput: 329.52784331002465, decode throughput: 83.7536164861968, latency: 2.4084172851562498, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.2868082582453881 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495855396805463 TFLOP/token, Bandwidth used (tps*model_size): 1246.4493198796283 GB/s, Power used in last sample period: 99.97 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 51.90198884162085, time to first token (ttft): 0.028177746963500977, time per output token (tpot): 0.019119974609375, prefill throughput: 213.66992526131426, decode throughput: 52.34401351166946, latency: 3.8567181640625003, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.8035773214435423 TFLOPs, FLOP per token used (FLOPs/tps): 0.015482592081310451 TFLOP/token, Bandwidth used (tps*model_size): 779.0410464536072 GB/s, Power used in last sample period: 189.164 W.
```

```
python -m low_bit_inference.torchao_autoquant configs/profile_inductor_torchao.yaml tps_only=True

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 89.01048835430088, time to first token (ttft): 0.11665095062255859, time per output token (tpot): 0.01064035791015625, prefill throughput: 51.43550463946163, decode throughput: 93.9818155752118, latency: 2.2469265136718755, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.3792935518729563 TFLOPs, FLOP per token used (FLOPs/tps): 0.01549585422318729 TFLOP/token, Bandwidth used (tps*model_size): 716.0262084228075 GB/s, Power used in last sample period: 218.853 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 66.07413011360592, time to first token (ttft): 0.0828336135864258, time per output token (tpot): 0.014706406982421876, prefill throughput: 72.43439668407419, decode throughput: 67.99811712459484, latency: 3.0269265625000004, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.0238673412943098 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495736978056349 TFLOP/token, Bandwidth used (tps*model_size): 531.5194842181145 GB/s, Power used in last sample period: 163.239 W.
```

### TorchAO weight only quantization configs:
```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_int4

# sample results on rtx5090
Profiling complete. Metrics: tokens per second (tps): 110.46259500545133, time to first token (ttft): 0.02517151985168457, time per output token (tpot): 0.008915862670898436, prefill throughput: 238.36594605444378, decode throughput: 112.16038181658757, latency: 1.8105791992187499, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.131255948915432 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882388686319972 TFLOP/token, Bandwidth used (tps*model_size): 525.688244681752 GB/s, Power used in last sample period: 222.833 W.

# sample results on rtx4090
Profiling complete. Metrics: tokens per second (tps): 81.38248950404981, time to first token (ttft): 0.04253532180786133, time per output token (tpot): 0.0120787470703125, prefill throughput: 141.82320108465413, decode throughput: 82.92354854174573, latency: 2.46152880859375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.0965454030236465 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011863166586817443 TFLOP/token, Bandwidth used (tps*model_size): 387.2968768577614 GB/s, Power used in last sample period: 175.086 W.
```

```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_int8

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 86.23675288271309, time to first token (ttft): 0.03870194549560546, time per output token (tpot): 0.0113895751953125, prefill throughput: 155.03200544933094, decode throughput: 87.80114584243424, latency: 2.31923701171875, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.3362891486210169 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495587483893862 TFLOP/token, Bandwidth used (tps*model_size): 693.713475062853 GB/s, Power used in last sample period: 150.572 W.

# rtx4090
Profiling complete. Metrics: tokens per second (tps): 67.2899972102495, time to first token (ttft): 0.0438882308959961, time per output token (tpot): 0.014625551025390626, prefill throughput: 136.71622550027232, decode throughput: 68.37357590441968, latency: 2.97221396484375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 1.0427147199017095 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495835386108249 TFLOP/token, Bandwidth used (tps*model_size): 541.3002721145971 GB/s, Power used in last sample period: 160.448 W.
```

```
python -m low_bit_inference.torchao_exp configs/profile_inductor_torchao.yaml tps_only=True quantization_method=bf16_fp8

# rtx 5090
Profiling complete. Metrics: tokens per second (tps): 7.817179876834605, time to first token (ttft): 0.13013018493652342, time per output token (tpot): 0.1272614765625, prefill throughput: 46.10767693983601, decode throughput: 7.857837496515249, latency: 25.584674218750003, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.12113389544623317 TFLOPs, FLOP per token used (FLOPs/tps): 0.015495856223700416 TFLOP/token, Bandwidth used (tps*model_size): 62.81911364359309 GB/s, Power used in last sample period: 373.53 W.

# rtx 4090
Profiling complete. Metrics: tokens per second (tps): 4.682847726809481, time to first token (ttft): 0.2161080322265625, time per output token (tpot): 0.21244604687500002, prefill throughput: 27.763907725868034, decode throughput: 4.707077991198945, latency: 42.709059375, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.07256472741832656 TFLOPs, FLOP per token used (FLOPs/tps): 0.01549585458499766 TFLOP/token, Bandwidth used (tps*model_size): 37.6315177801953 GB/s, Power used in last sample period: 183.867 W.
```

### Gemlite weight only quantization configs:
```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int1

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 166.61028598524325, time to first token (ttft): 0.043272089385986326, time per output token (tpot): 0.00577452880859375, prefill throughput: 138.66475240778237, decode throughput: 173.17430644332086, latency: 1.20040625, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.19797405326738346 TFLOPs, FLOP per token used (FLOPs/tps): 0.001188246284415586 TFLOP/token, Bandwidth used (tps*model_size): 393.17383054058246 GB/s, Power used in last sample period: 239.039 W.

# rtx4090
Profiling complete. Metrics: tokens per second (tps): 141.33079103428514, time to first token (ttft): 0.0639854591369629, time per output token (tpot): 0.006741152587890625, prefill throughput: 93.97336017203008, decode throughput: 148.34833545246514, latency: 1.4151974121093749, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.1679266008024844 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011881812843016453 TFLOP/token, Bandwidth used (tps*model_size): 333.51823481775995 GB/s, Power used in last sample period: 183.656 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_mxfp4

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 121.00605146173318, time to first token (ttft): 0.06190641174316406, time per output token (tpot): 0.007942053833007811, prefill throughput: 97.06742572488966, decode throughput: 125.91354764424747, latency: 1.6528378173828127, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.14378258313105757 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882263853269124 TFLOP/token, Bandwidth used (tps*model_size): 575.86576213253 GB/s, Power used in last sample period: 215.382 W.

# rtx4090
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_mxfp8

# sample results on rtx5090
Profiling complete. Metrics: tokens per second (tps): 98.25408111554313, time to first token (ttft): 0.06182771148681641, time per output token (tpot): 0.009855205200195312, prefill throughput: 97.16440711028424, decode throughput: 101.4699114366454, latency: 2.0355458007812497, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.11674966526854338 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882424011604173 TFLOP/token, Bandwidth used (tps*model_size): 810.4629461016299 GB/s, Power used in last sample period: 196.493 W.

# sample results on rtx4090

old, delete:
Profiling complete. Metrics: tokens per second (tps): 66.95707900014932, time to first token (ttft): 0.06158909378051758, time per output token (tpot): 0.014870641796875, prefill throughput: 146.13413610726815, decode throughput: 67.24704871472184, latency: 14.935041015625, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.10658418404439747 TFLOPs, FLOP per token used (FLOPs/tps): 0.001591828461396289 TFLOP/token, Bandwidth used (tps*model_size): 552.3051143799872 GB/s, Power used in last sample period: 122.034 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int4

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 136.6813193758208, time to first token (ttft): 0.043536249542236324, time per output token (tpot): 0.007087373901367188, prefill throughput: 137.8166563205426, decode throughput: 141.0959865120411, latency: 1.4632577880859376, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.16241108901997708 TFLOPs, FLOP per token used (FLOPs/tps): 0.001188246424322327 TFLOP/token, Bandwidth used (tps*model_size): 680.2748145278941 GB/s, Power used in last sample period: 98.478 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=bf16_int8

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 108.1547502692575, time to first token (ttft): 0.062385735321044924, time per output token (tpot): 0.0089208193359375, prefill throughput: 96.32169801802641, decode throughput: 112.09782073412448, latency: 1.849205126953125, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.12851429374499249 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882445609189493 TFLOP/token, Bandwidth used (tps*model_size): 869.1364738717975 GB/s, Power used in last sample period: 242.857 W.
```

### Gemlite weight and activation configs:
```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=int8_int8

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 113.00976782484236, time to first token (ttft): 0.04994605407714843, time per output token (tpot): 0.008587850219726563, prefill throughput: 120.13003313171816, decode throughput: 116.44358137964214, latency: 1.7697586181640623, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.13428344885051954 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882463917512864 TFLOP/token, Bandwidth used (tps*model_size): 908.1516149390329 GB/s, Power used in last sample period: 284.398 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=mxfp4_mxfp4

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 101.54175102609244, time to first token (ttft): 0.07823471221923828, time per output token (tpot): 0.009443981445312501, prefill throughput: 76.88701858713739, decode throughput: 105.88838698506757, latency: 1.9696458007812503, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.12065585131384414 TFLOPs, FLOP per token used (FLOPs/tps): 0.0011882388288029433 TFLOP/token, Bandwidth used (tps*model_size): 483.2354839824207 GB/s, Power used in last sample period: 188.71 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=mxfp8_mxfp8

# rtx5090
Profiling complete. Metrics: tokens per second (tps): 97.02547164248702, time to first token (ttft): 0.05242381439208985, time per output token (tpot): 0.010033220947265626, prefill throughput: 114.45249741816842, decode throughput: 99.66889460261453, latency: 2.061314501953125, iterations: 5
GPU util: FLOPs (total generate flops/latency): 0.11529016588920511 TFLOPs, FLOP per token used (FLOPs/tps): 0.001188246384557847 TFLOP/token, Bandwidth used (tps*model_size): 800.328583825416 GB/s, Power used in last sample period: 180.996 W.
```

```
python -m low_bit_inference.gemlite_exp configs/profile_inductor_gemlite.yaml tps_only=True quantization_method=nvfp4_nvfp4

# rtx5090 and rtx4090 both have MLIR error for TTIR to TTGIR MLIR pass.
```