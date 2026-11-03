export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"
export MASTER_PORT=22355
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
MIN_PIXELS=784 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model xxx/models/qwen-2.5-vl-7b \
    --model_type qwen2_5_vl \
    --adapters xxx/outputs/sft/checkpoint-2408 \
    --ref_adapters xxx/outputs/sft/checkpoint-2408 \
    --use_hf true \
    --reward_funcs crossmodalre \
    --train_type lora \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --target_modules all-linear \
    --lora_rank 64 \
    --lora_alpha 128 \
    --torch_dtype bfloat16 \
    --dataset xxx/data/moredata/train_grpo.jsonl \
    --max_completion_length 1024 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy no \
    --save_total_limit 2 \
    --logging_steps 4 \
    --output_dir xxx/outputs/grpo/$(date '+%Y%m%d') \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 6 \
    --num_generations 8 \
    --temperature 1.0 \
    --split_dataset_ratio 0.0 \
    --system "" \
    --num_iterations 1 \
    --beta 0.001 \
    --async_generate false \
    --deepspeed zero2
