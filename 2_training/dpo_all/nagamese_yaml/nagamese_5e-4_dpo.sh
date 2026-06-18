#!/bin/bash
YAML_DIR=/base/rd1/dpo_all/nagamese_yaml
LOG_DIR=/base/rd1/dpo_logs/nagamese
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup llamafactory-cli train "$YAML_DIR/nagamese_5e-4_dpo.yaml" > "$LOG_DIR/5e-4_dpo.log" 2>&1
