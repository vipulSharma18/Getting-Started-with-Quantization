import math
import os
import gc
import torch
import torchao
from datetime import datetime
from omegaconf import OmegaConf
# utils
from .utils.hf_utils import load_model_tokenizer
from .utils.config_utils import get_config
# optims
from .optims.kv_cache_optim import setup_cache


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer = load_model_tokenizer(config)
print(f"Model loaded {config.model_id}.")

prompt = config.prompt if isinstance(config.prompt, list) else [config.prompt]
print(f"Using prompt: {prompt}")

if config.use_cache:
    print("Setting up cache")
    model.config.use_cache = config.use_cache
    model.generation_config.cache_implementation = None  # remove the gen config var otherwise value error for using both ways
    cache_size = 2**math.ceil(math.log(len(prompt) + config.max_new_tokens, 2))
    past_key_values = setup_cache(cache_size, model.config, config)

# compile the model here if you want
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
os.environ["TORCHINDUCTOR_BENCHMARK_FUSION"] = "1"
os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
os.environ["TORCHINDUCTOR_FREEZING"] = "1" 
model.forward = torchao.autoquant(torch.compile(model.forward, fullgraph=True, dynamic=False, mode="max-autotune"))


model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

profiling_schedule = torch.profiler.schedule(
    skip_first = config.skip_first,
    wait = config.wait,
    warmup = config.warmup,
    active = config.active,
    repeat = config.repeat
)

cumulative_time = 0.0
generated_token_count = 0
mul_factor = max(1, config.repeat)

for i in range(config.skip_first + mul_factor*(config.wait + config.warmup + config.active)):
    print(f"Profiling Iteration {i}.")
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(config.device)  # will return a dict of token ids and attention mask
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = profiling_schedule,
        on_trace_ready = torch.profiler.tensorboard_trace_handler(config.profiling_dir, worker_name=f"{i}"),
        record_shapes = True,
        profile_memory = True,
        with_stack = True,  # this will add overhead, set it to False for benchmarking.
        with_flops = True,
    ) as prof:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            generated_token_ids = model.generate(
                **tokenized_prompt,
                do_sample=config.do_sample,
                top_p=config.top_p,
                top_k=config.top_k,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                past_key_values=past_key_values,
            )
            end.record()
        prof.step()
    
    torch.cuda.synchronize()
    step_time = start.elapsed_time(end)
    generated_tokens = tokenizer.batch_decode(generated_token_ids[0], skip_special_tokens=True)
    if i>=config.skip_first:
        cumulative_time += step_time
        generated_token_count += len(generated_tokens)
    print(f"Generated tokens (last 5): {generated_tokens[-5:]}, len: {len(generated_tokens)}, time: {step_time/1000}s")
    past_key_values.reset()
    del tokenized_prompt, generated_tokens, generated_token_ids
    gc.collect()
    torch.cuda.empty_cache()

print(f"Profiling complete, tokens per second: {generated_token_count/(cumulative_time/1000)}")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
prof.export_chrome_trace(os.path.join(config.profiling_dir, "trace.json"))