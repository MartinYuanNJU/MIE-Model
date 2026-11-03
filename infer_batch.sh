export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"
export MASTER_PORT=22300
MIN_PIXELS=784 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model xxx/outputs/grpo/20251020/v0-20251020-153307/checkpoint-400 \
    --model_type qwen2_5_vl \
    --use_hf true \
    --system "" \
    --infer_backend vllm \
    --val_dataset xxx/data/moredata/test_data_v6.jsonl \
    --max_batch_size 16 \
    --max_new_tokens 1024 \
    --result_path xxx/outputs/grpo/20251020/v0-20251020-153307/400.jsonl
