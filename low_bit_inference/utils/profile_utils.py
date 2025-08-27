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

    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule = profiling_schedule,
        on_trace_ready = torch.profiler.tensorboard_trace_handler(config.profiling_dir),
        record_shapes = True,
        profile_memory = True,
        with_stack = True,  # this will add overhead, set it to False for benchmarking.
        with_flops = True,
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
            prof.step()
            torch.cuda.synchronize()
            step_time = start.elapsed_time(end)
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