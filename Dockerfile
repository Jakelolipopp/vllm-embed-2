# Dockerfile
FROM vllm/vllm-openai:latest

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Enable hf_transfer for significantly faster model downloading during build
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Download the model directly into the container image using the new 'hf' CLI
RUN mkdir -p /model && \
    hf download sakamakismile/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4 --local-dir /model

# Set model environment variable to the local directory path
ENV MODEL_NAME="/model"

# Set engine and routing configuration variables per specification
ENV TOKENIZER_MODE="auto"
ENV SKIP_TOKENIZER_INIT="false"
ENV TRUST_REMOTE_CODE="false"
ENV LOAD_FORMAT="auto"
ENV DTYPE="auto"
ENV KV_CACHE_DTYPE="auto"
ENV MAX_MODEL_LEN="175000"
ENV DISTRIBUTED_EXECUTOR_BACKEND="mp"
ENV RAY_WORKERS_USE_NSIGHT="false"
ENV PIPELINE_PARALLEL_SIZE="1"
ENV TENSOR_PARALLEL_SIZE="1"
ENV MAX_PARALLEL_LOADING_WORKERS="0"
ENV ENABLE_PREFIX_CACHING="false"
ENV DISABLE_SLIDING_WINDOW="false"
ENV MAX_NUM_BATCHED_TOKENS="0"
ENV MAX_NUM_SEQS="256"
ENV MAX_LOGPROBS="20"
ENV DISABLE_LOG_STATS="false"
ENV ENABLE_LORA="false"
ENV LORA_DTYPE="auto"
ENV MAX_CPU_LORAS="0"
ENV FULLY_SHARDED_LORAS="false"
ENV DEVICE="auto"
ENV SCHEDULER_DELAY_FACTOR="0"
ENV ENABLE_CHUNKED_PREFILL="false"
ENV NUM_SPECULATIVE_TOKENS="0"
ENV NGRAM_PROMPT_LOOKUP_MAX="0"
ENV ENABLE_LOG_REQUESTS="false"
ENV GPU_MEMORY_UTILIZATION="0.95"
ENV BLOCK_SIZE="16"
ENV SWAP_SPACE="4"
ENV ENFORCE_EAGER="false"
ENV DISABLE_CUSTOM_ALL_REDUCE="false"

# RunPod & serving specific configurations
ENV DEFAULT_BATCH_SIZE="50"
ENV DEFAULT_MIN_BATCH_SIZE="1"
ENV DEFAULT_BATCH_SIZE_GROWTH_FACTOR="3"
ENV RAW_OPENAI_OUTPUT="true"
ENV OPENAI_RESPONSE_ROLE="assistant"
ENV OPENAI_SERVED_MODEL_NAME_OVERRIDE="q3.6-35a5-uncensored"
ENV MAX_CONCURRENCY="30"

# Extensibility & Tool Configurations
ENV ENABLE_EXPERT_PARALLEL="false"
ENV ENABLE_AUTO_TOOL_CHOICE="true"
ENV TOOL_CALL_PARSER="mistral"
ENV REASONING_PARSER="qwen3"

COPY worker.py /worker.py
WORKDIR /

CMD ["python3", "-u", "/worker.py"]