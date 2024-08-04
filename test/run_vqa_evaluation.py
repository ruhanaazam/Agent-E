from ast import Raise
import json
from typing import Any
from xmlrpc.client import Boolean
from pandas import isnull
from tqdm import tqdm
from validation_agent import validator
from validation_agent.utils import (get_chat_sequence, 
                                    get_screenshot_paths, 
                                    build_screenshot_prompt_sequence,
                                    build_text_prompt_sequence
                                    )
from summary_vqa import summarize_eval_method, printUnlabeled
from test_utils import robust_json_loader
from PIL import Image
import os
import pandas as pd
import pickle

def get_example()->list[str]:
    # # # # load full dataset
    # #task_file = "/Users/ruhana/Desktop/Agent-E/test/tasks/webvoyager_split/webvoyager_sampled_train_80.0.json"
    # task_file = "/Users/ruhana/Desktop/Agent-E/test/tasks/webvoyager_split/webvoyager_sampled_reason.json"
    # with open(task_file, 'r') as file:
    #     tasks = json.load(file)
    
    # all_examples: list[str] = []
    # for task in tqdm(tasks, desc="Building example prompt..."):
    #     task_id = task['task_id']
    #     intent = task['intent']
    #     start_url = task['start_url']
    #     score = task['score'] # TODO: 
    #     vision_score = task['vqa_score'] # this will need to be done later TODO: this needs to be in the dataset
    #     text_score = task['text_score'] # this will need to be done later TODO: this needs to be in the dataset
    #     reason = task['reason']
    #     # Set relevant paths
    #     chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/execution_logs_{task_id}.json"
    #     screenshot_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/snapshots"

    #     # String or screenshot sequence
    #     text_prompt_start = "Text Observation:"
    #     vision_prompt_start = "Visual Observation:"
    #     text_seq = [text_prompt_start] + get_user_chat(chat_path)
    #     screenshot_seq = get_screenshot_paths(screenshot_path)
    #     final_response = get_final_response(chat_path)
        
    #     # Make text and screenshot examples gpt4 compatible
    #     text_prompt = build_text_prompt_sequence(text_seq)
    #     screenshot_prompt = build_text_prompt_sequence([vision_prompt_start]) + build_screenshot_prompt_sequence(screenshot_seq) + build_text_prompt_sequence([final_response])

    #     # Get solution and intent prompt
    #     intent_prompt = build_text_prompt_sequence(["Task:", f"Startin at the url {start_url}, {intent}"])
        
    #     answer = "text"
    #     if vision_score == text_score:
    #         answer = "either"
    #     elif vision_score == score:
    #         answer = "vision"
        
    #     answer_str =f"""{{ 
    #                     "rationale": {reason},
    #                     "mode": {answer}
    #                 }}""" # TODO: Make sure this is consistent
    #     answer_prompt = build_text_prompt_sequence(["Solution:", answer_str])

        
    #     # Put it together 
    #     one_example: list[str] = intent_prompt + text_prompt + screenshot_prompt + answer_prompt
    #     all_examples = all_examples + one_example
        
    # # Save 
    # num = "reasoning"
    # with open(f'/Users/ruhana/Desktop/Agent-E/test/tasks/webvoyager_split/web_voyager_example_prompt_{num}.pkl', 'wb') as file:
    #     pickle.dump(all_examples, file)
    #     print(f"saved web_voyager_example_prompt...")
        
    # Load 
    num = "reasoning"
    with open(f'/Users/ruhana/Desktop/Agent-E/test/tasks/webvoyager_split/web_voyager_example_prompt_{num}.pkl', 'rb') as file:
        all_examples = pickle.load(file)
    return all_examples
    

def printSizes(state_seq):
    sizes = []
    for screenshot_path in state_seq:
        file_size_mb = os.path.getsize(screenshot_path['path_to_screenshot']) / (1024 * 1024)
        sizes.append(file_size_mb)
       
        with Image.open(screenshot_path['path_to_screenshot']) as img:
                width, height = img.size
        print(screenshot_path, file_size_mb, f"{width}x{height}")
    print(f"Total size: {sum(sizes)} MB")
    return

def get_user_chat(file_path: str):
    # Load chat log
    with open(file_path, 'r') as file:
        raw_chat = json.load(file)
    agent = list(raw_chat.keys())[-1]
    chat = raw_chat[agent]
        
    # Get only user output from the chat
    chat_sequence: list[str] = []
    for item in chat:        
        role = item.get('role', None)
        message = item.get('content', None)
        if role == "user":
            chat_sequence.append(message)
            
    # Append final statement from the planner
    if chat[-1].get("role") == "assistant":
        content = chat[-1].get('content', None)
        try:
            content = robust_json_loader(content)
            message = "The closing statement:" + content.get('final_response', "") + "."
            chat_sequence.append(message)
        except Exception as e:
            print(f"Exception getting user chat, likely due to unexpected formatting of {content}: {e} ")
            return chat_sequence # add without final message if there is a parsing issue...
    return chat_sequence

def get_final_response(file_path: str):
    # Load chat log
    with open(file_path, 'r') as file:
        raw_chat = json.load(file)
    agent = list(raw_chat.keys())[-1]
    chat = raw_chat[agent]
        
    # Append final statement from the planner
    message = ""
    if chat[-1].get("role") == "assistant":
        content = chat[-1].get('content', None)
        try:
            content = robust_json_loader(content)
            message = "The closing statement:" + content.get('final_response', "") + "."
        except Exception as e:
            print(f"Exception getting user chat, likely due to unexpected formatting of {content}: {e} ")# add without final message if there is a parsing issue...
    return message

def get_nested_chat(file_path:str)-> list[str]:
    # get all nested chats files
    files = os.listdir(file_path)
    nested_chat_files = [f for f in files if f.startswith("nested_chat")]
    
    #open each json file
    chat_sequence:list[str] = []
    for nested_chat_name in nested_chat_files:
        full_file_path = f"{file_path}/{nested_chat_name}"
        with open(full_file_path, 'r') as file:
            chat = json.load(file)
        
        for item in chat:
            message = item.get('content', None)
            if message:
                chat_sequence.append(message)
    return chat_sequence

def evaluate_tasks(tasks: list[Any], type:str, save_file:str="./temp_results.json"):
    if type=="smart":
        example_seq = get_example()
        
    for task in tqdm(tasks, desc=f"Evaluating with {type}..."):
        task_id = task['task_id']
        intent = task['intent']
        score = task['score']
        start_url = task['start_url']

        # Evaluate using VQA
        if True:
            try:
                # Evaluate
                if type=="vision" and ("vqa_raw_response" not in task or isnull(task['vqa_score'])): # # Evaluate using vision
                    # Get screenshots 
                    screenshot_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/snapshots"
                    chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/execution_logs_{task_id}.json"
                    screenshot_seq = get_screenshot_paths(screenshot_path)
                    #final_response = get_final_response(chat_path)
                    
                    # Run evaluation
                    #vision_result = validator.validate_task_vision(screenshot_seq, task, final_response=final_response)
                    vision_result = validator.validate_task_vision(screenshot_seq, task)
                    
                    # Add results to task dictionary
                    vision_result.pop('task_description', None)
                    vision_result.pop('pred_raw_response', None)
                    task["vqa_score"] = vision_result.get("pred_task_completed", None)
                    task["vqa_reason"] = vision_result.get("pred_rationale", None)
                    task["vqa_raw_response"] = json.dumps(vision_result)
                
                elif type=="text" and ("text_raw_response" not in task or isnull(task['text_score'])): # # Evaluate using chat text
                    # Get user chat log
                    chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/execution_logs_{task_id}.json"
                    text_seq = get_user_chat(chat_path)
                    
                    # Get nested chat log
                    # chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/"
                    # text_seq = get_nested_chat(chat_path)
                    
                    # Run evaluation
                    text_result = validator.validate_task_text(text_seq, task)
                    text_result.pop('task_description', None)
                    text_result.pop('pred_raw_response', None)
                    
                    # Add results to task dictionary
                    task["text_score"] = text_result.get("pred_task_completed", None)
                    task["text_reason"] = text_result.get("pred_rationale", None)
                    task["text_raw_response"] = json.dumps(text_result)
                    
                elif type=="text_vision" and ("text_vision_score" not in task or isnull(task['text_vision_reason'])): # Evalute with chat + vision 
                    # Gather (text & visions) observations
                    chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/execution_logs_{task_id}.json"
                    screenshot_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/snapshots"
                    text_seq = get_user_chat(chat_path)
                    screenshot_seq = get_screenshot_paths(screenshot_path)
                    
                    # Call evaluator
                    text_vision_result = validator.validate_task_text_vision(text_sequence=text_seq, vision_seqence=screenshot_seq, task=intent,)
                    
                    # Add results to dataframe
                    task["text_vision_score"] = text_vision_result["pred_task_completed"]
                    task["text_vision_reason"] = text_vision_result["pred_rationale"]
                    print(text_vision_result)
                
                elif type=="smart" and ("smart_score" not in task or isnull(task['smart_reason'])):
                    # Gather (text & vision) observations
                    chat_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/execution_logs_{task_id}.json"
                    screenshot_path = f"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_{task_id}/snapshots"
                    text_seq = get_user_chat(chat_path)
                    screenshot_seq = get_screenshot_paths(screenshot_path)
                    #final_response = get_final_response(chat_path) # TODO: ADD LATER
                    
                    # Run evaluator 
                    full_task = f"Startin at the url {start_url}, {intent}"
                    in_context_result = validator.validate_task_smart(text_sequence=text_seq, vision_seqence=screenshot_seq, task=full_task, example_sequence=example_seq)
                    
                    # Add results to dataframe
                    task["smart_score"] = in_context_result["pred_task_completed"]
                    task["smart_reason"] = in_context_result["pred_rationale"]
                    task["smart_raw_response"] = json.dumps(in_context_result)
                    
                # Print results -- More robust now, to none values...
                text_score, vqa_score, text_vision_score, smart_score = task.get("text_score", -1), task.get("vqa_score", -1), task.get("text_vision_score", -1), task.get("smart_score", -1)
                text_score = -1 if text_score is None else text_score
                vqa_score = -1 if vqa_score is None else vqa_score
                text_vision_score = -1 if text_vision_score is None else text_vision_score
                smart_score = -1 if smart_score is None else smart_score
                text_score, vqa_score, text_vision_score, smart_score = float(text_score), float(vqa_score), float(text_vision_score), float(smart_score)
                print(f'Task ID: {task_id}, Intent: {intent}\n Text Score: {text_score}, VQA Score: {vqa_score}, Vision+Text: {text_vision_score}, In-Content: {smart_score}, Manual Score: {score}')
        
                # # Write updated tasks back to the JSON file after each successful update
                test_results = save_file #"/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/raw_results_v2.json"
                
                with open(test_results, 'w') as file:
                    json.dump(tasks, file, indent=4)
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
                continue
    return tasks


def main():
    # Get all webvoyager tasks
    base = "text_vision"
    evaluation_name= f"{base}_eval_gpt4_o_full.json"
    save_file = f"/Users/ruhana/Desktop/Agent-E/test/evaluations/{evaluation_name}"

    task_file = f"/Users/ruhana/Desktop/Agent-E/test/evaluations/{evaluation_name}"
    #task_file = "/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/raw_results.json"

    with open(task_file, 'r') as file:
        tasks = json.load(file)

    # # Get subset of tasks if you want here...
    #tasks = [tasks[221]]
    #tasks = [tasks[i] for i in range(0, len(tasks)//30)] # will make about 30

    # # # Run evaluation
    tasks = evaluate_tasks(tasks, save_file=save_file, type=base)

    # # Open file to read
    task_file = save_file
    with open(task_file, 'r') as file:
        tasks = json.load(file)

    # Print a summary    
    df = pd.DataFrame(tasks)
    summarize_eval_method(df, base)
    printUnlabeled(df, base)

    # Existing Error cases: 
    # (Text) Json parsing issue (text): 599, 632
    # (ALL) No such file or directory: '/Users/ruhana/Desktop/Agent-E/ruhana_notes_observations/save_results/baseline_annotated/log_full/logs_for_task_246/execution_logs_246.json'
    # (Vision) No screenshots: 471, 480, 528, 529, 562, 569, 593, 596, 597, 600, 601, 615, 630, 638, 639, 640
    # (Vision) issue compressing image/ openai error, image too large: 117, 119

# WAYYY TO SLOW....
# there are two parts 
# 