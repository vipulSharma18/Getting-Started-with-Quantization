def autoname_modules(m):
    for name, module in m.named_modules():
        module.name = name  

def prep_for_inference(model, tokenizer):
    # ref: https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py#L238
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
    tokenizer.padding_side = "right"
    model.eval()
    model.generation_config.cache_implementation = "static"
    model.config.use_cache = True
    return model, tokenizer 