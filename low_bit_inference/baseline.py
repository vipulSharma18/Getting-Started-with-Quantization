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

## compile the model here if you want
# model.forward = torch.compile(model.forward)
torch.set_float32_matmul_precision('high')

def get_compiled_call(model):
    """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
    non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
    want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
    (where we want the speed-ups of compiled version with static shapes)."""
    # Only reset it if not present or different from previous config
    if "llama4" in self.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
        return self.__call__
    compile_config = compile_config or CompileConfig()
    default_config = getattr(self.generation_config, "compile_config", None) or CompileConfig()
    if (
        not hasattr(self, "_compiled_call")
        or getattr(self, "_last_compile_config", default_config) != compile_config
    ):
        self._last_compile_config = compile_config
        self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
    return self._compiled_call

model = model.to(config.device)
print(f"Model moved to {config.device}, starting profiling.")

profile_model(model, tokenizer, past_key_values, prompt, config)