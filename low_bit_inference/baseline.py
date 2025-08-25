import math
import torch
from omegaconf import OmegaConf
# utils
from .utils.hf_utils import load_model_tokenizer
from .utils.tokenize_utils import tokenize_prompt
from .utils.config_utils import get_config
# optims
from .optims.kv_cache_optim import setup_cache


config = get_config("config/profile_baseline.py")
print("config used:", OmegaConf.to_yaml(config), sep="\n")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer = load_model_tokenizer(config)
print(f"Model loaded {config.model_id}.")

if config.use_cache:
    print("Setting cache")
    cache_size = 2**math.ceil(math.log(config.max_new_tokens, 2))
    setup_cache(cache_size)

model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

profiling_schedule = torch.profiler.schedule(
    skip_first = config.skip_first,
    wait = config.wait,
    warmup = config.warmup,
    active = config.active,
    repeat = config.repeat
)

torch.cuda.synchronize()
for i in range(config.skip_first + config.repeat*(config.wait + config.warmup + config.active)):
    print(f"Profiling Iteration {i}.")
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
            tokenized_prompt = tokenize_prompt(config.prompt, config.chat_template)
            out = model.generate(
                tokenized_prompt,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                top_p=config.top_k,
            )
        prof.step()
print("profiling complete")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
print("-"*20)
print(prof.key_averages(group_by_stack_n=8).table(sort_by="cpu_time_total", row_limit=100))

prof.export_chrome_trace("trace.json")