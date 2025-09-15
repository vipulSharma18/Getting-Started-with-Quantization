import torch
import torchao
from omegaconf import OmegaConf
# utils
from .hf_loader import load_model_tokenizer_prompt_cache
from .utils.config_utils import get_config, to_torch_dtype
from .utils.profile_utils import profile_model


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

# compile the model here if you want
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch._inductor.config.benchmark_kernel = True
torch._inductor.config.benchmark_fusion = True
torch._inductor.config.freezing = True

model = model.to(config.device)
print("Model moved to GPU, starting profiling.")

assert config.compile_decode and config.quantize
print(f"Compile config: decode {config.compile_decode}, \
    prefill {config.compile_prefill}. Quantize status: {config.quantize}")

def cache_init(past_key_values, model, config, kv_compiled=False):
    if not kv_compiled:
        # just doing this so that the key and vals are output of cudagraph and hence mutating them in update doesn't cause cudagraph skipping
        past_key_values.early_initialization = torch.compile(past_key_values.early_initialization, mode="reduce-overhead")

    past_key_values.early_initialization(
        batch_size=1,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=to_torch_dtype(config.compute_dtype),
        device=config.device,
    )

    return past_key_values

def model_quantize(causal_model):
    causal_model.model = torchao.autoquant(causal_model.model)
    causal_model.lm_head = torchao.autoquant(causal_model.lm_head)

model.quantization_function = model_quantize

profile_model(model, tokenizer, prompt, config, past_key_values, cache_init)