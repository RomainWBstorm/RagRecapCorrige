llama-server -m ~/models/gemma4-26b-a4b/llama/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --alias gemma-4-26b \
    -ngl 99 -fa on \
    --parallel 1 --ctx-size 8196 \
    -ctk q8_0 -ctv q8_0 \
    --model-draft ~/models/gemma4-26b-a4b/llama/mtp-gemma-4-26B-A4B-it.gguf \
    --spec-type draft-mtp --spec-draft-n-max 4 \
    --host 0.0.0.0 --port 8000