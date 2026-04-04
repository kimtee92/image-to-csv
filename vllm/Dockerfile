FROM vllm/vllm-openai:cu130-nightly

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir git+https://github.com/huggingface/transformers.git xxhash
