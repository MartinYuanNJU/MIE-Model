export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="$no_proxy,.byteintl.net"
export MASTER_PORT=22355
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIN_PIXELS=784 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model xxx/outputs/sft/20251014/v0-20251014-154726/checkpoint-561 \
    --model_type qwen2_5_vl \
    --use_hf true \
    --external_plugins xxx/swift/examples/train/grpo/plugin/plugin.py \
    --reward_funcs mmreformatr mmrelengthr mmreanswerr \
    --train_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --torch_dtype bfloat16 \
    --dataset xxx/data/moredata/train_grpo_561-050.jsonl \
    --dataset_shuffle false \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 300 \
    --eval_strategy no \
    --save_total_limit 6 \
    --logging_steps 4 \
    --output_dir xxx/outputs/grpo/$(date '+%Y%m%d') \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 0.8 \
    --split_dataset_ratio 0.0 \
    --system "" \
    --num_iterations 1 \
    --async_generate false \
    --deepspeed zero2 \
    --beta 0.001
