import math
import os
import torch
from omegaconf import OmegaConf
# utils
from .utils.hf_utils import load_model_tokenizer
from .utils.config_utils import get_config
# optims
from .optims.kv_cache_optim import setup_cache


# note: pass configs from cli for omegaconf to read, not as part of code.
config = get_config("config/profile_template.py")
print("config used:", OmegaConf.to_yaml(config), sep="\n")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer = load_model_tokenizer(config)
print(f"Model loaded {config.model_id}.")

prompt = config.prompt if isinstance(config.prompt, list) else list(config.prompt)
print(f"Using prompt: {prompt}")

if config.use_cache:
    print("Setting up cache")
    model.config.use_cache = config.use_cache
    cache_size = 2**math.ceil(math.log(len(prompt) + config.max_new_tokens, 2))
    past_key_values = setup_cache(cache_size, model.config, config)

## compile the model here if you want
# model.forward = torch.compile(model.forward)

model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

profiling_schedule = torch.profiler.schedule(
    skip_first = config.skip_first,
    wait = config.wait,
    warmup = config.warmup,
    active = config.active,
    repeat = config.repeat
)

for i in range(config.skip_first + config.repeat*(config.wait + config.warmup + config.active)):
    print(f"Profiling Iteration {i}.")
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(config.device)  # will return a dict of token ids and attention mask
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = profiling_schedule,
        on_trace_ready = torch.profiler.tensorboard_trace_handler(config.profiling_dir),
        record_shapes = True,
        profile_memory = True,
        with_stack = True,  # this will add overhead, set it to False for benchmarking.
        with_flops = True,
    ) as prof:
        with torch.inference_mode():
            generated_token_ids = model.generate(
                **tokenized_prompt,
                do_sample=config.do_sample,
                max_new_tokens=config.max_new_tokens,
                past_key_values=past_key_values,
            )
        prof.step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # tokens per second benchmarks don't seem to include tokenization and detokenization overhead
    # dummy batch dimension removed
    generated_tokens = tokenizer.batch_decode(generated_token_ids[0], skip_special_tokens=True)
    print(f"Generated tokens: {generated_tokens}")
    past_key_values.reset()
print("profiling complete")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
print("-"*20)
print(prof.key_averages(group_by_stack_n=8).table(sort_by="cpu_time_total", row_limit=100))

prof.export_chrome_trace(os.path.join(config.profiling_dir, "trace.json"))