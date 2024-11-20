#!/bin/bash

# Define parameters
LOG_PATH="/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs"
RESULT_PATH="/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results"
RAW_JSON_PATH="/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/raw_results.json"
METHOD="main_chat_only"

# Run experiments in parallel with different seeds and models
for SEED in 0 1 2; do
    for MODEL in "gpt-4o" "gpt-4-turbo-preview"; do
        python test_validator.py \
            --log_path "$LOG_PATH" \
            --result_path "$RESULT_PATH" \
            --raw_result_path "$RAW_JSON_PATH" \
            --output_file "text_${MODEL}_seed_${SEED}.json" \
            --method "$METHOD" \
            --model "$MODEL" &
    done
done

# Wait for all parallel tasks to complete
wait

echo "All experiments have completed."
