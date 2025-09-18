## Run Benchmark:
> Note: Run without tps_only=True to get trace of CPU and GPU kernels.      
> Note: Run with skip_first=1 wait=2 warmup=1 active=1 for small runs (1 flop and quant, 2 compile, 1 warmup, 1 measure).      
> Note: Run with profile_compile=True to get profile trace of compilation process.      
> Note: Run with TORCH_TRACE="log/compile" to generate dynamo logs that can be parsed with tlparse log/compile/* --overwrite. Alternatively, can use TORCH_LOGS="dynamic,guards,recompiles,perf_hints,fusion".      

```
python -m low_bit_inference.torch_baseline configs/profile_baseline.yaml tps_only=True

python -m low_bit_inference.torchinductor configs/profile_inductor.yaml tps_only=True
python -m low_bit_inference.torchao.autoquant configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.torchao.int4wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.int4_full configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.torchao.int8wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.int8_full configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.torchao.fp8wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.fp8_full configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.torchao.fp6wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.fp6_full configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.torchao.fp4wo configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.fp4_full configs/profile_inductor_torchao.yaml tps_only=True

python -m low_bit_inference.gemlite.fp1_full configs/profile_inductor_torchao.yaml tps_only=True
python -m low_bit_inference.gemlite.fp1_58_full configs/profile_inductor_torchao.yaml tps_only=True
```