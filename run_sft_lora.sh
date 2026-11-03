export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"
export MASTER_PORT=22355
MIN_PIXELS=784 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model xxx/models/qwen-2.5-vl-7b \
    --model_type qwen2_5_vl \
    --use_hf true \
    --train_type lora \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --target_modules all-linear \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --dataset xxx/data/moredata/qwen_sft_v4.jsonl \
    --split_dataset_ratio 0.0 \
    --torch_dtype bfloat16 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 8 \
    --max_length 2096 \
    --system "" \
    --output_dir xxx/outputs/sft/$(date '+%Y%m%d') \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero2
