vllm serve ~/models/embedder/bge-m3 \
    --served-model-name bge-m3 \
    --gpu-memory-utilization 0.2 \
    --max-model-len 8192 \
    --host 0.0.0.0 --port 8001
