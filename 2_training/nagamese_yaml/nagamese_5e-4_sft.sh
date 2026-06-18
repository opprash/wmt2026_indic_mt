#!/bin/bash
YAML_DIR=/base/rd1/sft/nagamese_yaml
LOG_DIR=/base/rd1/train_logs/nagamese
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/nagamese_5e-4_sft.yaml" > "$LOG_DIR/5e-4_sft.log" 2>&1
