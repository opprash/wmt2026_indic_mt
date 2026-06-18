#!/bin/bash
YAML_DIR=/base/rd1/sft/kokborok_yaml
LOG_DIR=/base/rd1/train_logs/kokborok
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/kokborok_2e-4_sft.yaml" > "$LOG_DIR/2e-4_sft.log" 2>&1
