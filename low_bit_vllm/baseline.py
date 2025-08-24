import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenize_utils import tokenize_prompt

# hf model args
device = 'cuda:0'
compute_dtype = torch.float16
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
cache_dir = "/root/.cache/huggingface/hub"
# profiling args
profiling_dir = os.getcwd()
profiling_dir = os.path.join(profiling_dir[:os.find("low_bit_vllm")], "log/baseline")
skip_first = 0
wait = 0
warmup = 0
active = 1
repeat = 1
# prompt args
prompt = "Write an essay about large language models."
max_new_tokens = 1024
chat_template = ""
top_k = 3

print(f"Loading pretrained model and tokenizer: {model_id}.")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model     = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",
    cache_dir=cache_dir,
    device_map="cpu"
)
params = sum(p.numel() for p in model.parameters())
print(f"Model loaded {model_id}. Number of parameters in model: {params}")

def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name  

autoname_modules(model)
model = model.to(device)
print("Model moved to GPU, starting profiling.")
torch.cuda.synchronize()

profiling_schedule = torch.profiler.schedule(
    skip_first = skip_first,
    wait = wait,
    warmup = warmup,
    active = active,
    repeat = repeat
)

for i in range(skip_first + repeat*(wait + warmup + active)):
    print(f"Profiling Iteration {i}.")
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = profiling_schedule,
        on_trace_ready = torch.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes = True,
        profile_memory = True,
        with_stack = True,  # this will add overhead, set it to False for benchmarking.
        with_flops = True,
    ) as prof:
        with torch.inference_mode():
            tokenized_prompt = tokenize_prompt(prompt, chat_template)
            out = model.generate(
                tokenized_prompt,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                top_p=top_k,
            )
        prof.step()
print("profiling complete")

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
print("-"*20)
print(prof.key_averages(group_by_stack_n=8).table(sort_by="cpu_time_total", row_limit=100))

prof.export_chrome_trace("trace.json")