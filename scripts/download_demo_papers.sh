#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

# 1) Attention Is All You Need
curl -L "https://arxiv.org/pdf/1706.03762.pdf" -o "data/raw/attention_is_all_you_need.pdf"

# 2) FlashAttention
curl -L "https://arxiv.org/pdf/2205.14135.pdf" -o "data/raw/flashattention.pdf"

# 3) vLLM / PagedAttention
curl -L "https://arxiv.org/pdf/2309.06180.pdf" -o "data/raw/vllm_pagedattention.pdf"

echo "Done. Files saved to data/raw/"