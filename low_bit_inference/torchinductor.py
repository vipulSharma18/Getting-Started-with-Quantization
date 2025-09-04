import torch
from omegaconf import OmegaConf
# utils
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config
from .utils.profile_utils import profile_model


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

# compile the model here if you want
# basic cuda and cudnn configs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch._inductor.config.benchmark_kernel = True
torch._inductor.config.max_autotune = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.benchmark_fusion = True

def get_compiled_call(model, dynamic = None):
    compiled_call = torch.compile(model.__call__, fullgraph=True, dynamic=dynamic)
    return compiled_call

model.get_compiled_call = get_compiled_call
model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)