# worker.py
import os
import time
import json
import asyncio
import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

# --- Gather Environment Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME")
SERVED_MODEL_NAME = os.environ.get("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
ROLE = os.environ.get("OPENAI_RESPONSE_ROLE")

# Engine arguments dynamically built from explicit environment variables
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tokenizer_mode=os.environ.get("TOKENIZER_MODE"),
    trust_remote_code=os.environ.get("TRUST_REMOTE_CODE") == "true",
    load_format=os.environ.get("LOAD_FORMAT"),
    dtype=os.environ.get("DTYPE"),
    kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE"),
    max_model_len=int(os.environ.get("MAX_MODEL_LEN")),
    distributed_executor_backend=os.environ.get("DISTRIBUTED_EXECUTOR_BACKEND"),
    pipeline_parallel_size=int(os.environ.get("PIPELINE_PARALLEL_SIZE")),
    tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE")),
    enable_prefix_caching=os.environ.get("ENABLE_PREFIX_CACHING") == "true",
    disable_sliding_window=os.environ.get("DISABLE_SLIDING_WINDOW") == "true",
    max_num_seqs=int(os.environ.get("MAX_NUM_SEQS")),
    max_logprobs=int(os.environ.get("MAX_LOGPROBS")),
    disable_log_stats=os.environ.get("DISABLE_LOG_STATS") == "true",
    enable_lora=os.environ.get("ENABLE_LORA") == "true",
    device=os.environ.get("DEVICE"),
    scheduler_delay_factor=float(os.environ.get("SCHEDULER_DELAY_FACTOR")),
    enable_chunked_prefill=os.environ.get("ENABLE_CHUNKED_PREFILL") == "true",
    num_speculative_tokens=int(os.environ.get("NUM_SPECULATIVE_TOKENS")),
    gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION")),
    block_size=int(os.environ.get("BLOCK_SIZE")),
    swap_space=int(os.environ.get("SWAP_SPACE")),
    enforce_eager=os.environ.get("ENFORCE_EAGER") == "true",
    disable_custom_all_reduce=os.environ.get("DISABLE_CUSTOM_ALL_REDUCE") == "true",
)

# Handle variables that should only be injected if they aren't disabled/zero
max_loading_workers = int(os.environ.get("MAX_PARALLEL_LOADING_WORKERS"))
if max_loading_workers > 0:
    engine_args.max_parallel_loading_workers = max_loading_workers

max_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS"))
if max_batched_tokens > 0:
    engine_args.max_num_batched_tokens = max_batched_tokens

# --- Initialize vLLM Engine ---
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def handler(job):
    """
    Directly embeds the vLLM execution and processes incoming inference requests,
    streaming output or returning the full raw OpenAI format depending on inputs.
    """
    job_input = job["input"]
    
    messages = job_input.get("messages", [])
    prompt = job_input.get("prompt")
    stream = job_input.get("stream", False)

    # Convert request properties into vLLM SamplingParams
    sampling_params = SamplingParams(
        n=1,
        presence_penalty=job_input.get("presence_penalty", 0.0),
        frequency_penalty=job_input.get("frequency_penalty", 0.0),
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 1.0),
        top_k=job_input.get("top_k", -1),
        max_tokens=job_input.get("max_tokens", 8192),
        stop=job_input.get("stop", []),
        ignore_eos=False
    )

    req_id = random_uuid()
    created_time = int(time.time())

    # Format using chat template if `messages` array is provided instead of raw `prompt`
    if messages and not prompt:
        tokenizer = await engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    if not prompt:
        return {"error": "A 'prompt' or 'messages' array must be provided in the input payload."}

    results_generator = engine.generate(prompt, sampling_params, req_id)

    # --- Streaming Handling ---
    if stream:
        async def stream_output():
            previous_texts = [""] * sampling_params.n
            async for res in results_generator:
                for output in res.outputs:
                    i = output.index
                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    
                    if delta_text:
                        yield {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": SERVED_MODEL_NAME,
                            "choices": [{
                                "index": i,
                                "delta": {"role": ROLE, "content": delta_text},
                                "finish_reason": None
                            }]
                        }
                if res.finished:
                    for output in res.outputs:
                        yield {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": SERVED_MODEL_NAME,
                            "choices": [{
                                "index": output.index,
                                "delta": {},
                                "finish_reason": output.finish_reason
                            }]
                        }
        
        # Sequentially yield JSON blocks back out to RunPod connection 
        async for chunk in stream_output():
            yield chunk
        return

    # --- Synchronous Handling ---
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    choices = []
    for output in final_output.outputs:
        choices.append({
            "index": output.index,
            "message": {
                "role": ROLE,
                "content": output.text
            },
            "finish_reason": output.finish_reason
        })

    usage = {
        "prompt_tokens": len(final_output.prompt_token_ids),
        "completion_tokens": sum(len(o.token_ids) for o in final_output.outputs),
        "total_tokens": len(final_output.prompt_token_ids) + sum(len(o.token_ids) for o in final_output.outputs)
    }

    # Return structure respecting RAW_OPENAI_OUTPUT overrides
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_time,
        "model": SERVED_MODEL_NAME,
        "choices": choices,
        "usage": usage
    }

if __name__ == "__main__":
    concurrency_limit = int(os.environ.get("MAX_CONCURRENCY", 30))
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda x: concurrency_limit
    })