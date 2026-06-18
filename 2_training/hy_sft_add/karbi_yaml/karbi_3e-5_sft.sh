#!/bin/bash
YAML_DIR=/base/rd1/hy_sft_add/karbi_yaml
LOG_DIR=/base/rd1/train_logs_hy_add/karbi
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/karbi_3e-5_sft.yaml" > "$LOG_DIR/3e-5_sft.log" 2>&1
