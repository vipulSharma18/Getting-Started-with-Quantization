import gc
from contextlib import nullcontext
import torch
from torchao.utils import get_model_size_in_bytes
from torch.utils.flop_counter import FlopCounterMode


def compile_util(model):
    if model.compile_decode:
        model.compiled_forward_decode = model.get_compiled_call(
            dynamic=False,
            mode="max-autotune",
            fullgraph=True,
        )

    if model.compile_prefill:
        model.compiled_forward_prefill = model.get_compiled_call(
            dynamic=True,
            fullgraph=True,
            mode="default",
        )


def enable_inductor_profiling():
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.graph_diagram = True
    torch._inductor.config.trace.compile_profile = True


class NoProfiler(nullcontext):
    def step(self):
        """No-op step function that doesn't do anything."""
        pass


def flops_bandwidth_power(tps, model_size, model_flops):
    """
    Returns the model flops utilization, and the bandwidth utilization given a GPU
    architecture and a token/s decoding rate.

    architecture_metadata = {
        "rtx5090": {"peak_flops_f4f32": 1676, "peak_flops_f8f16": 838, "peak_flops_f8f32": 419, \
            "peak_flops_f16f16": 419, "peak_flops_f16f32": 209, "peak_flops_i8": 838, \
            "peak_bandwidth": 1792, "vram": 32},
        "rtx4090": {"peak_flops_f4f32": 0, "peak_flops_f8f16": 660, "peak_flops_f8f32": 330, \
            "peak_flops_f16f16": 330, "peak_flops_f16f32": 165.2, "peak_flops_i8": 660, \
            "peak_bandwidth": 1008, "vram": 24},
        "rtx3090": {"peak_flops_f4f32": 0, "peak_flops_f8f16": 0, "peak_flops_f8f32": 0, \
            "peak_flops_f16f16": 142.3, "peak_flops_f16f32": 71.2, "peak_flops_i8": 284.7, \
            "peak_bandwidth": 936, "vram": 24},
    }
    """
    # metadata = architecture_metadata[architecture]
    flops = tps * model_flops
    bandwidth = tps * model_size
    power = torch.cuda.power_draw()/1e3
    return {"flops": flops, "bandwidth": bandwidth, "power": power}


def profile_model(model, tokenizer, prompt, config, past_key_values, cache_init):
    """
    Model profiling code.
    Metrics:
    --------
    TPS: Tokens/sec or Throughput = #decode tokens/(prefill+decode time)
    TTFT: Time to first token = prefill time
    TPOT: Time per output token = decode time/#decode tokens
    Prefill Throughput: #prefill tokens/prefill time
    Decode Throughput: #decode tokens/decode time
    Latency: prefill + decode time
    """
    metrics = {"tps": 0, \
        "ttft": 0, \
        "tpot": 0, \
        "prefill_throughput": 0, \
        "decode_throughput": 0, \
        "latency": 0, \
        "iterations": 0
    }

    # enable logging for inductor compilation times and graph
    if config.profile_compile:
        enable_inductor_profiling()

    profiling_schedule = torch.profiler.schedule(
        skip_first = config.skip_first,
        wait = config.wait,
        warmup = config.warmup,
        active = config.active,
        repeat = config.repeat
    )

    mul_factor = max(1, config.repeat)
    total_steps = config.skip_first + mul_factor*(config.wait + config.warmup + config.active)

    # returns a dict of token_ids and attention_mask keys
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(config.device)

    if config.tps_only:
        prof = NoProfiler()
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

    kv_compiled = False
    compile_iter = 1 # delay compile by 1 iter for quantization and/or flop counting
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)

    with prof:
        for i in range(total_steps):
            print(f"Profiling iteration {i+1} out of total {total_steps}.")
            torch.compiler.cudagraph_mark_step_begin()
            past_key_values = cache_init(past_key_values, model, config, kv_compiled)
            kv_compiled = True

            # mark step begin will release past cudagraph's memory, so clean it up
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # timers
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            
            if i==0:
                flop_context = flop_counter
            else:
                flop_context = nullcontext()

            with torch.inference_mode():
                if i==0 and model.quantize:
                    model.quantization_function(model, quantized=False)
                if i==compile_iter:
                    compile_util(model)

                with flop_context:
                    start.record()
                    generated_token_ids = model.generate(
                        **tokenized_prompt,
                        past_key_values=past_key_values,
                        prefill_start=prefill_start,
                        prefill_end=prefill_end,
                        decode_start=decode_start,
                        decode_end=decode_end,
                    )
                    end.record()

                if i==0 and model.quantize:
                    model.quantization_function(model, quantized=True)
            torch.cuda.synchronize()
            generated_tokens = tokenizer.batch_decode(generated_token_ids[0], skip_special_tokens=True)

            # gather metrics
            latency = start.elapsed_time(end)/1e3
            prefill_time = prefill_start.elapsed_time(prefill_end)/1e3
            decode_time = decode_start.elapsed_time(decode_end)/1e3
            prefill_tokens = len(tokenized_prompt["input_ids"][0])
            decode_tokens = len(generated_tokens) - prefill_tokens
            prefill_throughput = prefill_tokens/prefill_time
            decode_throughput = decode_tokens/decode_time
            tpot = decode_time/decode_tokens
            throughput = decode_tokens/latency
            prof.step()

            # log metrics
            curr_action = profiling_schedule(i)
            print(f"Generated tokens (last 5): {generated_tokens[-5:]}, \
                total len: {len(generated_tokens)}, \
                prefill len: {prefill_tokens}, \
                decode len: {decode_tokens}, \
                latency: {latency}s, \
                prefill_time: {prefill_time}s, \
                decode_time: {decode_time}s, \
                prefill_throughput: {prefill_throughput}tps, \
                decode_throughput: {decode_throughput}tps, \
                throughput: {throughput}tps, \
                ttft: {prefill_time}s, \
                tpot: {tpot}s. \
            ")

            print(f"Profiling step_num: {i+1}, curr action: {curr_action}.")
            if curr_action in [torch.profiler.ProfilerAction.RECORD, torch.profiler.ProfilerAction.RECORD_AND_SAVE]:
                print(f"Profile step {i+1} included for tps calculation.")
                old_iter = metrics["iterations"]
                metrics["iterations"] += 1
                metrics["tps"] = (metrics["tps"]*old_iter + throughput)/metrics["iterations"]
                metrics["ttft"] = (metrics["ttft"]*old_iter + prefill_time)/metrics["iterations"]
                metrics["tpot"] = (metrics["tpot"]*old_iter + tpot)/metrics["iterations"]
                metrics["prefill_throughput"] = (metrics["prefill_throughput"]*old_iter + prefill_throughput)/metrics["iterations"]
                metrics["decode_throughput"] = (metrics["decode_throughput"]*old_iter + decode_throughput)/metrics["iterations"]
                metrics["latency"] = (metrics["latency"]*old_iter + latency)/metrics["iterations"]

            # cleanup step:
            with torch.inference_mode():
                # tensors created in inference mode might force to be modifed in inference mode for all ops
                past_key_values.reset()
            del generated_tokens, generated_token_ids
            gc.collect()
            torch.cuda.empty_cache()

    print(
        "Profiling complete. Metrics: "
        f"tokens per second (tps): {metrics['tps']}, "
        f"time to first token (ttft): {metrics['ttft']}, "
        f"time per output token (tpot): {metrics['tpot']}, "
        f"prefill throughput: {metrics['prefill_throughput']}, "
        f"decode throughput: {metrics['decode_throughput']}, "
        f"latency: {metrics['latency']}, "
        f"iterations: {metrics['iterations']}"
    )
    model_size = get_model_size_in_bytes(model, ignore_embeddings=True)/1e9
    model_flops =  flop_counter.get_total_flops()
    gpu_utils = flops_bandwidth_power(metrics['tps'], model_size, model_flops)
    print("GPU util: "
        f"FLOPS used (tps*model flops): {gpu_utils["flops"]}, "
        f"Bandwidth used (tps*model_size): {gpu_utils["bandwidth"]}, "
        f"Power used in last sample period: {gpu_utils["power"]}."
    )

    try:
        if not config.tps_only and prof.profiler is not None:
            prof.export_chrome_trace(config.profiling_dir + "/trace.json")
            print("Manually exported the trace.")
    except Exception:
        print("Trace was already saved. Exiting.")
