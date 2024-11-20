from validation_agent.validator import *
from load_annotations import AnnotationLoader
import pandas as pd
import json
from tqdm import tqdm
import argparse
import os

def validate_task_id(task_id, annotation_loader, config):
    try: 
        intent = annotation_loader.get_intent(task_id)
        main_chat_sequence = annotation_loader.get_high_level_trajectory(task_id)
        screenshot_paths = annotation_loader.get_screenshot_paths(task_id)
    except Exception as e:
        return {"error": f"Unable to load workflow, failed with: {e}"}
        
    
    method = config.get("method", None)
    model = config.get("model", "gpt-4o")
    
    validation_result = {}
    if method == "main_chat_only":
        validation_result = validate_task_text(main_chat_sequence, intent, model)
    elif method == "nested_chat_only":
        pass
    elif method == "DOM_tree_only":
        pass
    elif method == "screenshots_only":
        # validate_task_vision(screenshot_paths, intent, model)
        pass
    elif method == "screenshot_final_response":
        pass 
    else:
        print("Error! No method specified!")
    
    return validation_result

def summarize_validator(original_df):
    # Ensure the necessary columns exist
    if "pred_score" in original_df.columns and "score" in original_df.columns:
        # Initialize counters
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        # Iterate through the DataFrame and compare predictions to actual scores
        for index, row in original_df.iterrows():
            actual = row["score"]
            predicted = row["pred_score"]
            
            if predicted == 1 and actual == 1:
                true_positive += 1
            elif predicted == 0 and actual == 0:
                true_negative += 1
            elif predicted == 1 and actual == 0:
                false_positive += 1
            elif predicted == 0 and actual == 1:
                false_negative += 1
        
        # Calculate rates and accuracy
        total = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / total
        tp_rate = true_positive / total
        tn_rate = true_negative / total
        fp_rate = false_positive / total
        fn_rate = false_negative / total

        # Print the summary
        print(f"Accuracy: {accuracy:.2%}")
        print(f"True Positive Rate: {tp_rate:.2%}")
        print(f"True Negative Rate: {tn_rate:.2%}")
        print(f"False Positive Rate: {fp_rate:.2%}")
        print(f"False Negative Rate: {fn_rate:.2%}")
    else:
        print("Columns 'pred_score' or 'score' are missing in the DataFrame.")

def configure_experiment():
    parser = argparse.ArgumentParser(description="Configure and run experiments")
    
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log files")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result files")
    parser.add_argument("--raw_json_path", type=str, required=True, help="Path to raw results JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output JSON file")
    parser.add_argument("--method", type=str, required=True, help="Validation method (e.g., 'main_chat_only')")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (default: gpt-4o)")

    args = parser.parse_args()

    # Create configuration dictionary
    config = {
        "LOG_PATH": args.log_path,
        "RESULT_PATH": args.result_path,
        "RAW_JSON_PATH": args.raw_json_path,
        "OUTPUT_FILE": os.path.join("/Users/ruhana/Agent-E/test/evaluator_results", args.output_file),
        "method": args.method,
        "model": args.model,
    }

    return config

def main():
    # Parse arguments and configure the experiment
    config = configure_experiment()
    output_file = config["OUTPUT_FILE"]

    # Load annotations
    original_annotation = AnnotationLoader(config["LOG_PATH"], config["RESULT_PATH"])

    # Load the human-annotated data
    with open(config["RAW_JSON_PATH"], 'r') as f:
        json_data = json.load(f)
    original_df = pd.json_normalize(json_data)

    # Add prediction columns
    if "pred_score" not in original_df.columns:
        original_df["pred_score"] = None
        original_df["pred_reason"] = None

    # Iterate through each task_id and validate
    for index, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Processing predictions"):
        if row['pred_score'] is None:
            task_id = row["task_id"]
            validation_result = validate_task_id(task_id, original_annotation, config)
            
            # Update the DataFrame with the results
            original_df.at[index, "pred_reason"] = validation_result.get("pred_rationale", None)
            original_df.at[index, "pred_score"] = validation_result.get("pred_task_completed", None)
        else:
            pass # task_id is already complete
        
        if index % 20:
            original_df.to_json(output_file, orient="records", indent=4)
            summarize_validator(original_df)

    # Save the updated DataFrame to a JSON file
    original_df.to_json(output_file, orient="records", indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
    
    # EXAMPLE:
    # python test_validator.py \
    # --log_path "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs" \
    # --result_path "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results" \
    # --raw_json_path "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/raw_results.json" \
    # --output_file "hehe.json" \
    # --method "main_chat_only" \
    # --model "gpt-4o"