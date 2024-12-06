
import sys
import os
# Add project home directory in the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import pandas as pd
from test.validation_agent.validator import *
from load_annotations import AnnotationLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

def print_confusion(gpt_eval):
    true_positive = gpt_eval[(gpt_eval['pred_score'] == 1) & (gpt_eval['score'] == 1)]
    true_negative = gpt_eval[(gpt_eval['pred_score'] == 0) & (gpt_eval['score'] == 0)]
    false_positive = gpt_eval[(gpt_eval['pred_score'] == 1) & (gpt_eval['score'] == 0)]
    false_negative = gpt_eval[(gpt_eval['pred_score'] == 0) & (gpt_eval['score'] == 1)]
    
    # Calculate the accuracy
    accuracy = (gpt_eval['pred_score'] == gpt_eval['score']).mean()
    
    # Calculate recall, precision, and F1 score
    recall = recall_score(gpt_eval['score'], gpt_eval['pred_score'])
    precision = precision_score(gpt_eval['score'], gpt_eval['pred_score'])
    f1 = f1_score(gpt_eval['score'], gpt_eval['pred_score'])
    
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"True Positive (TP): {len(true_positive)}")
    print(f"True Negative (TN): {len(true_negative)}")
    print(f"False Positive (FP): {len(false_positive)}")
    print(f"False Negative (FN): {len(false_negative)}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

def validate_task_id(task_id, annotation_loader, config):
    intent = annotation_loader.get_intent(task_id)
    main_chat_sequence = annotation_loader.get_high_level_trajectory(task_id)
    screenshot_paths = annotation_loader.get_screenshot_paths(task_id)
    
    method = config.get("method", None)
    model = config.get("model", "gpt-4o")
    
    validation_result = {}
    if method == "main_chat_only":
        validation_result = validate_task_text(main_chat_sequence, intent, model=model)
    elif method == "nested_chat_only":
        pass # TODO
    elif method == "DOM_tree_only":
        pass
    elif method == "screenshots_only":
        #validate_task_vision(screenshot_paths, intent)
        pass # TODO
    elif method == "screenshot_final_response":
        pass # TODO
    else:
        print("Error! No method specified!")
    return validation_result

def validate_all_annotation(annotation_loader, config):
    result_dfs = []
    task_id_list = range(0, 643, 5)
    
    for task_id in tqdm(task_id_list, desc="Processing tasks", unit="task"):
        original_results = annotation_loader.get_result(task_id)
        pred_result = validate_task_id(task_id, annotation_loader, config)
        
        #
        pred_score = pred_result.get("pred_task_completed")
        pred_reason = str(pred_result.get("pred_reasoning_questions"))
        
        # Store new results
        original_results['pred_score'] = pred_score
        original_results['pred_reason'] = pred_reason
        intent = original_results['intent'].squeeze()
        score = original_results['score'].squeeze()
        print(f"Task ID: {task_id}\t true_score: {score}\t pred_score: {pred_score}")
        result_dfs.append(original_results)
    
    all_results = pd.concat(result_dfs)
    return all_results
    
    
# call the annotation_loader
LOG_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs"
RESULT_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results"
original_annotation = AnnotationLoader(LOG_PATH, RESULT_PATH)
config = {"method": "main_chat_only",
            "model": "gpt-4-vision-preview"}

# Test a single task_id
# print(validate_task_id(task_id=0, 
#                        annotation_loader=original_annotation, 
#                        config=config)) 

gpt_eval_df = validate_all_annotation(annotation_loader=original_annotation, 
                        config=config)
print_confusion(gpt_eval_df)
    
    

