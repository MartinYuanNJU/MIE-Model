export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"
export MASTER_PORT=22355
MIN_PIXELS=784 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model xxx/models/qwen-2.5-vl-7b \
    --model_type qwen2_5_vl \
    --use_hf true \
    --train_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --system "" \
    --torch_dtype bfloat16 \
    --dataset xxx/data/moredata/qwen_sft_v4_zero_.jsonl \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 8 \
    --max_length 2096 \
    --output_dir xxx/outputs/sft/$(date '+%Y%m%d') \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2
