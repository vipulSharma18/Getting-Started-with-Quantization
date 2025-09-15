import math
from .optims.cache_optim import StaticCache
from transformers import AutoTokenizer
from .model import LlamaForCausalLM
from .utils.config_utils import to_torch_dtype


def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name

def load_model_tokenizer_prompt_cache(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        cache_dir=config.cache_dir,
    )
    # if a rust based tokenizer is not avialable, this falls back to Python implementation which is slower.
    model = LlamaForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=to_torch_dtype(config.compute_dtype),
        attn_implementation=config.attn_implementation,
        cache_dir=config.cache_dir,
        device_map=config.device,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {params}")
    model.eval()
    autoname_modules(model)

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
    tokenizer.padding_side = config.padding_side

    prompt = config.prompt if isinstance(config.prompt, list) else [config.prompt]
    print(f"Using prompt: {prompt}")

    model.generation_config.max_new_tokens = config.max_new_tokens
    model.generation_config.chat_template = config.chat_template
    model.generation_config.do_sample = config.do_sample
    model.generation_config.top_k = config.top_k
    model.generation_config.top_p = config.top_p
    model.generation_config.temperature = config.temperature

    model.compile_decode = config.compile_decode
    model.compile_prefill = config.compile_prefill
    model.quantize = config.quantize

    past_key_values = None
    if config.use_cache:
        model.config.use_cache = config.use_cache
        # set to None as explicitly passing kv cache
        model.generation_config.cache_implementation = None
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
        cache_size = 2**math.ceil(math.log(prompt_len + config.max_new_tokens, 2))
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=cache_size,
            device=config.device,
            dtype=to_torch_dtype(config.compute_dtype)
        )

    return model, tokenizer, prompt, past_key_values