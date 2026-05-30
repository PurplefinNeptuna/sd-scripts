#!/bin/bash
#
# Anima ADDifT Training Script for bellymove dataset
# Trains a LoRA to learn the transformation from source to target images
#
# Usage:
#   ./train_bellymove.sh              # Full training (100 steps)
#   ./train_bellymove.sh --smoke-test # Smoke test (1 step)
#

set -e

# Disable tokenizer parallelism warning to avoid spam during training
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Configuration
# ============================================================

# Paths
PRETRAINED_MODEL="/mnt/storage2/Old_Data/Documents/Programs/StabilityMatrix/Data/Models/DiffusionModels/anima-base-v1.0.safetensors"
QWEN3_TEXT_ENCODER="/mnt/storage2/Old_Data/Documents/Programs/StabilityMatrix/Data/Models/TextEncoders/qwen_3_06b_base.safetensors"
VAE_PATH="/mnt/storage2/Old_Data/Documents/Programs/StabilityMatrix/Data/Models/VAE/qwen_image_vae.safetensors"
DATASET_CONFIG="datasets/bellymove/dataset.toml"

# Output
OUTPUT_DIR="outputs/bellymove"
OUTPUT_NAME="bellymove"

# Network
NETWORK_MODULE="networks.lora_anima"
NETWORK_DIM=4
NETWORK_ALPHA=4

# Training
MAX_STEPS=1000
LEARNING_RATE=5e-5
OPTIMIZER="AdamW8bit"
LR_SCHEDULER="cosine_with_restarts"
LR_SCHEDULER_NUM_CYCLES=3
LR_WARMUP_STEPS=100

# Precision
MIXED_PRECISION="bf16"

# Loss weighting (logit_normal for ADDifT)
WEIGHTING_SCHEME="logit_normal"
LOGIT_MEAN=-0.2
LOGIT_STD=1.5

# Network dropout to prevent overfitting
NETWORK_DROPOUT=0.01

# Reverse pair settings for bidirectional learning
REVERSE_WEIGHT=1.0
REVERSE_MULTIPLIER=-1.0

# Save settings
SAVE_EVERY_N_STEPS=100

# ============================================================
# Parse arguments
# ============================================================

SMOKE_TEST=false
for arg in "$@"; do
    case $arg in
        --smoke-test)
            SMOKE_TEST=true
            MAX_STEPS=1
            echo "[INFO] Smoke test mode enabled: 1 step"
            ;;
    esac
done

# ============================================================
# Create output directory
# ============================================================
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Anima ADDifT Training - bellymove"
echo "============================================================"
echo ""
echo "  Pretrained model:    $PRETRAINED_MODEL"
echo "  Qwen3 text encoder:  $QWEN3_TEXT_ENCODER"
echo "  VAE:                 $VAE_PATH"
echo "  Dataset config:      $DATASET_CONFIG"
echo ""
echo "  Output dir:          $OUTPUT_DIR"
echo "  Output name:         $OUTPUT_NAME"
echo "  Network:             $NETWORK_MODULE (dim=$NETWORK_DIM, alpha=$NETWORK_ALPHA)"
echo ""
echo "  Max steps:           $MAX_STEPS"
echo "  Learning rate:       $LEARNING_RATE"
echo "  Optimizer:           $OPTIMIZER"
echo "  Scheduler:           $LR_SCHEDULER (num_cycles=$LR_SCHEDULER_NUM_CYCLES, warmup=$LR_WARMUP_STEPS steps)"
echo ""
echo "  Loss weighting:      $WEIGHTING_SCHEME (logit_mean=$LOGIT_MEAN, logit_std=$LOGIT_STD)"
echo "  Network dropout:     $NETWORK_DROPOUT"
echo "  Bidirectional:       enabled (reverse_weight=$REVERSE_WEIGHT, reverse_multiplier=$REVERSE_MULTIPLIER)"
echo "  Save every N steps:  $SAVE_EVERY_N_STEPS"
echo ""
echo "  Mixed precision:     $MIXED_PRECISION"
echo "  Gradient checkpointing: enabled"
echo "  Latent caching:      enabled"
echo "  TensorBoard logging: enabled (logs/)"
echo ""
echo "============================================================"
echo ""

# ============================================================
# Training command
# ============================================================

uv run anima_train_addift.py \
    --dataset_config="$DATASET_CONFIG" \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --qwen3="$QWEN3_TEXT_ENCODER" \
    --vae="$VAE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --output_name="$OUTPUT_NAME" \
    --save_model_as=safetensors \
    --network_module="$NETWORK_MODULE" \
    --network_dim="$NETWORK_DIM" \
    --network_alpha="$NETWORK_ALPHA" \
    --network_train_unet_only \
    --cache_text_encoder_outputs \
    --addift_cache_conditioning_latents \
    --learning_rate="$LEARNING_RATE" \
    --optimizer_type="$OPTIMIZER" \
    --lr_scheduler="$LR_SCHEDULER" \
    --lr_scheduler_num_cycles=$LR_SCHEDULER_NUM_CYCLES \
    --lr_warmup_steps=$LR_WARMUP_STEPS \
    --max_train_steps=$MAX_STEPS \
    --mixed_precision="$MIXED_PRECISION" \
    --gradient_checkpointing \
    --attn_mode=xformers \
    --split_attn \
    --vae_chunk_size=64 \
    --vae_disable_cache \
    --save_precision=bf16 \
    --seed=42 \
    --max_data_loader_n_workers=1 \
    --log_with tensorboard \
    --logging_dir logs \
    --add_reverse_pairs \
    --reverse_weight=$REVERSE_WEIGHT \
    --reverse_multiplier=$REVERSE_MULTIPLIER \
    --network_dropout=$NETWORK_DROPOUT \
    --weighting_scheme="$WEIGHTING_SCHEME" \
    --logit_mean=$LOGIT_MEAN \
    --logit_std=$LOGIT_STD \
    --save_every_n_steps=$SAVE_EVERY_N_STEPS

echo ""
echo "============================================================"
echo "  Training completed!"
echo "  Model saved to: $OUTPUT_DIR/${OUTPUT_NAME}.safetensors"
echo "============================================================"
