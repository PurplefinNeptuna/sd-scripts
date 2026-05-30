#!/bin/bash
#
# Anima ADDifT Training Script for bloat dataset
# Trains a LoRA to learn the stomach enlargement effect after overeating
#
# Usage:
#   ./train_bloat.sh              # Full training (1180 steps)
#   ./train_bloat.sh --smoke-test # Smoke test (1 step)
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
DATASET_CONFIG="datasets/bloat/dataset.toml"

# Output
OUTPUT_DIR="outputs/bloat"
OUTPUT_NAME="bloat"

# Network
NETWORK_MODULE="networks.lora_anima"
NETWORK_DIM=4
NETWORK_ALPHA=4

# Training
MAX_STEPS=500
LEARNING_RATE=2e-5
OPTIMIZER="AdamW8bit"
LR_SCHEDULER="constant"

# Precision
MIXED_PRECISION="bf16"

# Loss weighting
WEIGHTING_SCHEME="cosmap"
TIMESTEP_SAMPLING="uniform"

# Network dropout to prevent overfitting
NETWORK_DROPOUT=0.01

# Save settings
SAVE_EVERY_N_STEPS=50

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
echo "  Anima ADDifT Training - bloat"
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
echo "  Scheduler:           $LR_SCHEDULER"
echo ""
echo "  Loss weighting:      $WEIGHTING_SCHEME"
echo "  Timestep Sampling:   $TIMESTEP_SAMPLING"
echo "  Network dropout:     $NETWORK_DROPOUT"
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
    --cache_latents \
    --addift_cache_conditioning_latents \
    --cache_latents_to_disk \
    --learning_rate="$LEARNING_RATE" \
    --optimizer_type="$OPTIMIZER" \
    --lr_scheduler="$LR_SCHEDULER" \
    --max_train_steps=$MAX_STEPS \
    --mixed_precision="$MIXED_PRECISION" \
    --gradient_checkpointing \
    --attn_mode=xformers \
    --split_attn \
    --save_precision=bf16 \
    --seed=42 \
    --max_data_loader_n_workers=1 \
    --log_with tensorboard \
    --logging_dir logs \
    --network_dropout=$NETWORK_DROPOUT \
    --weighting_scheme="$WEIGHTING_SCHEME" \
    --timestep_sampling="$TIMESTEP_SAMPLING" \
    --addift_min_sigma=0.2 \
    --addift_max_sigma=0.9 \
    --add_reverse_pairs \
    --addift_pair_settings "datasets/bloat/pair_settings.json" \
    --save_every_n_steps=$SAVE_EVERY_N_STEPS

echo ""
echo "============================================================"
echo "  Training completed!"
echo "  Model saved to: $OUTPUT_DIR/${OUTPUT_NAME}.safetensors"
echo "============================================================"
