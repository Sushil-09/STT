#!/bin/bash

export MODEL_NAME="openai/whisper-small"
export DATA_DIR="data/processed_dataset"
export OUTPUT_DIR="whisper_finetuned_indian"

python run_speech_recognition_ctc.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --logging_strategy "steps" \
  --logging_steps 100 \
  --save_total_limit 2 \
  --fp16 \
  --group_by_length \
  --do_train \
  --do_eval
