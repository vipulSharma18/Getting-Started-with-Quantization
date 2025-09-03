import math
from transformers import AutoTokenizer
from .model import LlamaForCausalLM
from .utils.config_utils import to_torch_dtype
from .optims.attention_optim import setup_cache


def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name

def load_model_tokenizer_prompt_cache(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=config.cache_dir)  # if a rust based tokenizer is not avialable, this falls back to Python implementation which is slower.
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

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
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

    past_key_values = None
    if config.use_cache:
        print("Setting up cache")
        model.config.use_cache = config.use_cache
        model.generation_config.cache_implementation = None  # remove the gen config var otherwise value error for setting it here and passing an explicit key value store past_key_values
        cache_size = 2**math.ceil(math.log(len(prompt) + config.max_new_tokens, 2))
        past_key_values = setup_cache(cache_size, model.config, config)

    return model, tokenizer, prompt, past_key_values