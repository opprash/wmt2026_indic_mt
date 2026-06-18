#!/bin/bash
YAML_DIR=/base/rd1/sft_add/nagamese_yaml
LOG_DIR=/base/rd1/train_logs_add/nagamese
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/nagamese_2e-4_sft.yaml" > "$LOG_DIR/2e-4_sft.log" 2>&1
