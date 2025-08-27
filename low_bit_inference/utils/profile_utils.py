import torch
import gc


def profile_model(model, tokenizer, past_key_values, prompt, config):
    """
    Reused model profiling code.
    """
    profiling_schedule = torch.profiler.schedule(
        skip_first = config.skip_first,
        wait = config.wait,
        warmup = config.warmup,
        active = config.active,
        repeat = config.repeat
    )

    cumulative_time = 0.0
    generated_token_count = 0
    mul_factor = max(1, config.repeat)

    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(config.device)  # will return a dict of token ids and attention mask

    if config.tps_only:
        activities = []
        profiling_flag = False
        trace_handler = lambda x: pass
    else:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        profiling_flag = True
        trace_handler = torch.profiler.tensorboard_trace_handler(config.profiling_dir)

    with torch.profiler.profile(
        activities = activities,
        schedule = profiling_schedule,
        on_trace_ready = trace_handler,
        record_shapes = profiling_flag,
        profile_memory = profiling_flag,
        with_stack = profiling_flag,  # this will add considerable overhead, set it to False for benchmarking.
        with_flops = profiling_flag,
    ) as prof:
        for i in range(config.skip_first + mul_factor*(config.wait + config.warmup + config.active)):    
            print(f"Profiling iteration {i}")
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                start.record()
                generated_token_ids = model.generate(
                    **tokenized_prompt,
                    do_sample=config.do_sample,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    temperature=config.temperature,
                    max_new_tokens=config.max_new_tokens,
                    past_key_values=past_key_values,
                )
                end.record()
            torch.cuda.synchronize()
            step_time = start.elapsed_time(end)
            prof.step()
            generated_tokens = tokenizer.batch_decode(generated_token_ids[0], skip_special_tokens=True)
            if i>=config.skip_first:
                cumulative_time += step_time
                generated_token_count += len(generated_tokens)
            print(f"Generated tokens (last 5): {generated_tokens[-5:]}, len: {len(generated_tokens)}, time: {step_time/1000}s")
            print(f"Profiling step_num: {prof.step_num}, action taken: {profiling_schedule(prof.step_num)}")
            past_key_values.reset()
            del generated_tokens, generated_token_ids
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Profiling complete, tokens per second: {generated_token_count/(cumulative_time/1000)}")
    try:
        if prof.profiler is not None:
            prof.export_chrome_trace(config.profiling_dir + "/trace.json")
    except Exception as e:
        print("Trace was already saved. Exiting.")

def dump_device_tensors(device_id=0):
    device = torch.device(f'cuda:{device_id}')
    
    # Get all tensors in memory
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device == device:
                tensors.append({
                    'shape': tuple(obj.shape),
                    'dtype': obj.dtype,
                    'size_mb': obj.numel() * obj.element_size() / (1024**2),
                    'requires_grad': obj.requires_grad,
                    'id': id(obj)
                })
        except:
            pass
    
    # Sort by size
    tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    
    total_mb = sum(t['size_mb'] for t in tensors)
    print(f"Device {device_id} - {len(tensors)} tensors, {total_mb:.2f}MB total")
    
    for i, t in enumerate(tensors[:20]):  # top 20
        print(f"{i+1:2d}. {t['shape']} {t['dtype']} {t['size_mb']:.2f}MB grad:{t['requires_grad']}")
    
    return tensors