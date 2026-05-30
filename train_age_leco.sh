#!/bin/bash
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
DATASET_CONFIG="datasets/bloat/dataset_leco.toml"

# Output
OUTPUT_DIR="outputs/age"
OUTPUT_NAME="age_slider_anima"

# Network
NETWORK_MODULE="networks.lora_anima"
NETWORK_DIM=4
NETWORK_ALPHA=4

# Training
MAX_STEPS=2000
LEARNING_RATE=1e-4
OPTIMIZER="AdamW8bit"
LR_SCHEDULER="constant"

# Precision
MIXED_PRECISION="bf16"

# Network dropout to prevent overfitting
NETWORK_DROPOUT=0.01

# Reverse pair settings for bidirectional learning
REVERSE_WEIGHT=1.0
REVERSE_MULTIPLIER=1.0
ENABLE_REVERSE=true

# iLECO prompt pairs
ILECO_PROMPT_PAIRS="datasets/bloat/ileco_prompt_pairs_adult.json"

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
        --no-reverse)
            ENABLE_REVERSE=false
            echo "[INFO] Reverse pairs disabled"
            ;;
    esac
done

# ============================================================
# Create output directory
# ============================================================
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Anima iLECO Training - age"
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
echo "  Network dropout:     $NETWORK_DROPOUT"

if [ "$ENABLE_REVERSE" = true ]; then
    echo "  Bidirectional:       enabled (reverse_weight=$REVERSE_WEIGHT, reverse_multiplier=$REVERSE_MULTIPLIER)"
else
    echo "  Bidirectional:       disabled"
fi

echo "  Save every N steps:  $SAVE_EVERY_N_STEPS"
echo ""
echo "  Mixed precision:     $MIXED_PRECISION"
echo "  Gradient checkpointing: enabled"
echo "  Latent caching:      enabled (dataset-backed iLECO)"
echo "  TensorBoard logging: enabled (logs/)"
echo ""
echo "============================================================"
echo ""

# ============================================================
# Build the command array safely
# ============================================================

CMD=(
    uv run anima_train_leco.py
    --ileco_latent_source="dataset"
    --dataset_config="$DATASET_CONFIG"
    --pretrained_model_name_or_path="$PRETRAINED_MODEL"
    --qwen3="$QWEN3_TEXT_ENCODER"
    --vae="$VAE_PATH"
    --output_dir="$OUTPUT_DIR"
    --output_name="$OUTPUT_NAME"
    --save_model_as="safetensors"
    --network_module="$NETWORK_MODULE"
    --network_dim="$NETWORK_DIM"
    --network_alpha="$NETWORK_ALPHA"
    --network_train_unet_only
    --cache_text_encoder_outputs
    --cache_latents
    --cache_latents_to_disk
    --ileco_prompt_pairs="$ILECO_PROMPT_PAIRS"
)

if [ "$ENABLE_REVERSE" = true ]; then
    CMD+=(
        "--add_reverse_pairs"
        "--reverse_weight=$REVERSE_WEIGHT"
        "--reverse_multiplier=$REVERSE_MULTIPLIER"
    )
fi

CMD+=(
    "--learning_rate=$LEARNING_RATE"
    "--optimizer_type=$OPTIMIZER"
    "--lr_scheduler=$LR_SCHEDULER"
    "--max_train_steps=$MAX_STEPS"
    "--mixed_precision=$MIXED_PRECISION"
    "--gradient_checkpointing"
    "--attn_mode=xformers"
    "--split_attn"
    "--save_precision=bf16"
    "--seed=42"
    "--max_data_loader_n_workers=1"
    "--log_with" "tensorboard"
    "--logging_dir" "logs"
    "--network_dropout=$NETWORK_DROPOUT"
    "--save_every_n_steps=$SAVE_EVERY_N_STEPS"
    "--ileco_min_sigma=0.2"
    "--ileco_max_sigma=1.0"
    "--ileco_guidance_scale=5.0"
)

# ============================================================
# Execute
# ============================================================

"${CMD[@]}"

echo ""
echo "============================================================"
echo "  Training completed!"
echo "  Model saved to: $OUTPUT_DIR/${OUTPUT_NAME}.safetensors"
echo "============================================================"