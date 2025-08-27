import torch

def compile_model(model, tokenizer):
    model = torch.compile(model, fullgraph=True, dynamic=False, mode="max-autotune")
    model.forward = torch.compile(model.forward, fullgraph=True, dynamic=False, mode="max-autotune")

    model._update_model_kwargs_for_generation = torch.compiler.disable(model._update_model_kwargs_for_generation)
    model.prepare_inputs_for_generation = torch.compiler.disable(model.prepare_inputs_for_generation)

    model.generate = torch.compile(model.generate, mode="reduce-overhead")
    tokenizer.batch_decode = torch.compile(tokenizer.batch_decode, mode="reduce-overhead")

    return model, tokenzier