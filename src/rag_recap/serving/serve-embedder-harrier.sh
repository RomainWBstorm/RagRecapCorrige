vllm serve ~/models/embedder/harrier-oss-v1-0.6b \
    --served-model-name harrier-oss-v1-0.6b \
    --runner pooling \
    --gpu-memory-utilization 0.3 \
    --max-model-len 8192 \
    --host 0.0.0.0 --port 8001
