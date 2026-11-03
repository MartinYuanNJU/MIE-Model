#!/usr/bin/env bash
# SSGPF LoRA / QLoRA fine-tuning launcher (paper §III-D).
# Edit MODEL / DATA / OUTPUT paths below for your environment.
# Note: this snapshot trains multi_train_unilabel.json (ablation "-unilabel").
# For the main SSGPF ARSGP run, point --data_path to preprocess/data/multi_train.json
# and train a second adapter on preprocess/data/multi_ed.json for ETSGP/MED.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL_BASE="${MODEL_BASE:-xxx/models/llava-v1.5-7b}"
VISION_TOWER="${VISION_TOWER:-xxx/models/clip-vit-large-patch14-336}"
IMAGE_FOLDER="${IMAGE_FOLDER:-xxx/datasets/}"
DATA_PATH="${DATA_PATH:-$ROOT/preprocess/data/multi_train_unilabel.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/llava_lora_multi_train_unilabel}"
GPU="${GPU:-localhost:0}"

deepspeed --master_port=23500 --include "${GPU}" \
    "$ROOT/llava/train/train_mem.py" \
    --deepspeed "$ROOT/scripts/zero2.json" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 64 \
    --mm_projector_lr 2e-5 \
    --bits 4 \
    --model_name_or_path "$MODEL_BASE" \
    --version llava_llama_2 \
    --data_path "$DATA_PATH" \
    --validation_data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 12 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True
