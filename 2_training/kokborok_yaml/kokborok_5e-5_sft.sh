#!/bin/bash
YAML_DIR=/base/rd1/sft/kokborok_yaml
LOG_DIR=/base/rd1/train_logs/kokborok
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/kokborok_5e-5_sft.yaml" > "$LOG_DIR/5e-5_sft.log" 2>&1
