import torch
import gc


def none_context():
    """
    A no-op context manager that doesn't do anything but has a step function.
    Useful as a drop-in replacement for torch.profiler.profile when profiling is disabled.
    """
    class NoneContext:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def step(self):
            """No-op step function that doesn't do anything."""
            pass
    
    return NoneContext()


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

    mul_factor = max(1, config.repeat)
    total_steps = config.skip_first + mul_factor*(config.wait + config.warmup + config.active)

    cumulative_time = 0.0
    generated_token_count = 0

    # returns a dict of token_ids and attention_mask keys
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(config.device)

    if config.tps_only:
        prof = none_context()
    else:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        profiling_flag = True
        trace_handler = torch.profiler.tensorboard_trace_handler(config.profiling_dir)
        prof = torch.profiler.profile(
            activities = activities,
            schedule = profiling_schedule,
            on_trace_ready = trace_handler,
            record_shapes = profiling_flag,
            profile_memory = profiling_flag,
            with_stack = profiling_flag,  # this will add considerable overhead, set it to False for benchmarking.
            with_flops = profiling_flag,
        )

    with prof:
        for i in range(total_steps):
            print(f"Profiling iteration {i} out of total {total_steps}")
            torch.compiler.cudagraph_mark_step_begin() 
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                start.record()
                generated_token_ids = model.generate(
                    **tokenized_prompt,
                    past_key_values=past_key_values,
                )
                end.record()
            torch.cuda.synchronize()
            step_time = start.elapsed_time(end)
            generated_tokens = tokenizer.batch_decode(generated_token_ids[0], skip_special_tokens=True)
            prof.step()
            curr_action = profiling_schedule(i)
            print(f"Generated tokens (last 5): {generated_tokens[-5:]}, len: {len(generated_tokens)}, time: {step_time/1000}s")
            print(f"Profiling step_num: {i}, curr action: {curr_action}, or in reality NONE if tps_only is True.")
            if curr_action in [torch.profiler.ProfilerAction.RECORD, torch.profiler.ProfilerAction.RECORD_AND_SAVE]:
                cumulative_time += step_time
                generated_token_count += len(generated_tokens)  # generated_tokens contains prefill and new decode stage tokens
                print(f"Profile step {i} included for tps calculation.")
            # need to reset which involves inplace ops and hence need to go in inference mode otherwise error
            with torch.inference_mode():
                past_key_values.reset()
            del generated_tokens, generated_token_ids
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Profiling complete, tokens per second: {generated_token_count/(cumulative_time/1000)}")
    try:
        if not config.tps_only and prof.profiler is not None:
            prof.export_chrome_trace(config.profiling_dir + "/trace.json")
    except Exception:
        print("Trace was already saved. Exiting.")
