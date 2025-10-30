import gc
import torch
import torchao
from omegaconf import OmegaConf
# utils
from low_bit_inference.hf_loader import load_model_tokenizer_prompt_cache
from low_bit_inference.utils.config_utils import get_config, to_torch_dtype
from low_bit_inference.utils.profile_utils import profile_model


config = get_config()
print("config used -- ", OmegaConf.to_yaml(config), sep="\n")
torch.cuda.set_device(config.device)
print(f"PyTorch sees {torch.cuda.device_count()} devices, current device: {torch.cuda.current_device()}")

print(f"Loading pretrained model and tokenizer: {config.model_id}.")
model, tokenizer, prompt, past_key_values = load_model_tokenizer_prompt_cache(config)
print(f"Model loaded {config.model_id}.")

# compile the model here if you want
torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch._inductor.config.benchmark_kernel = True
torch._inductor.config.benchmark_fusion = True

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

def model_quantize(causal_model, config, quantized=False):
    # use torchao autoquant
    def filter_fn(module, fqn, *args):
        # Exempt embedding layer from quantization
        flag = fqn not in ["embed_tokens", "lm_head"]
        default_flag = torchao.quantization.quant_api._is_linear(module, fqn, *args)
        return flag and default_flag

    if not quantized:
        causal_model.model = torchao.autoquant(
            causal_model.model,
            manual=True,
            filter_fn=filter_fn
        )
    else:
        causal_model.model.finalize_autoquant()
    torch.cuda.empty_cache()
    gc.collect()

model.quantization_function = model_quantize

profile_model(model, tokenizer, prompt, config, past_key_values, cache_init)