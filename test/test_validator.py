from validation_agent.validator import *
from load_annotations import AnnotationLoader
import pandas as pd
import json
from tqdm import tqdm

def validate_task_id(task_id, annotation_loader, config):
    intent = annotation_loader.get_intent(task_id)
    main_chat_sequence = annotation_loader.get_high_level_trajectory(task_id)
    screenshot_paths = annotation_loader.get_screenshot_paths(task_id)
    
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
        total = len(original_df)
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

# Call the annotation_loader
LOG_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs"
RESULT_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results"
original_annotation = AnnotationLoader(LOG_PATH, RESULT_PATH)

# Set experiment configurations!
base_path = "/Users/ruhana/Agent-E/test/evaluator_results/"
output_file = f"{base_path}/test.json"
config = {"method": "main_chat_only",
          "model": "gpt-4-turbo-preview"}

# Load the human annotated data
raw_result_path = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/raw_results.json"
with open(raw_result_path, 'r') as f:
    json_data = json.load(f)
original_df = pd.json_normalize(json_data)

# Add two new columns
original_df["pred_score"] = None
original_df["pred_reason"] = None

#original_df = original_df[30:40]

# Iterate through each task_id and validate
for index, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Processing predictions"):
    task_id = row["task_id"]
    validation_result = validate_task_id(task_id, original_annotation, config)
    
    # Update the DataFrame with the results
    original_df.at[index, "pred_reason"] = validation_result["pred_rationale"]
    original_df.at[index, "pred_score"] = validation_result["pred_task_completed"]

    row = original_df.loc[index]
    print(row[['task_id', 'intent', 'start_url', 'score', 'pred_score', 'reason', 'pred_reason']])

     # Save the updated DataFrame to a new JSON file
    if index % 20:
        original_df.to_json(output_file, orient="records", indent=4)
        
        # Give quick summary
        summarize_validator(original_df)
