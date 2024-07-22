from .prompts import (
    prompt__validate_action, 
    prompt__validate_with_vision__intro, prompt__validate_with_vision__close, 
    prompt__validate_with_text__intro, prompt__validate_with_text__close, 
    prompt__validate_with_text_vision__intro, prompt__validate_with_text_vision__close
)
from .utils import (
    _fetch_openai_completion,
    fetch_openai_vision_completion,
    load_screenshot_for_state,
    build_screenshot_prompt_sequence,
    build_text_prompt_sequence
)
from typing import Dict, Any, List
import json
import argparse
import os
from test_utils import robust_json_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_images", type=str, help="Path to task demo folder")
    parser.add_argument("--is_action", action="store_true", help="If TRUE, then eval on action")
    parser.add_argument("--is_task_completion", action="store_true", help="If TRUE, then eval on task completion success")
    parser.add_argument("--use_vqa", action="store_true", help="If TRUE, then use VQA on task completion success")
    parser.add_argument("--requested_action", type=str, help="Action requested for action validation")
    parser.add_argument("--task", type=str, help="Description of the task for task completion validation")
    return parser.parse_known_args()

def validate_action(init_state: Dict[str, Any], requested_action: Dict[str, Any], resultant_state: Dict[str, Any]) -> Dict[str, str]:
    ## Simple validator function of an action that takes as input the initial state, the requested action, and the resultant state, and determines if it succeeded.
    path_to_screenshot_before, encoded_image_before = load_screenshot_for_state(init_state)
    path_to_screenshot_after, encoded_image_after = load_screenshot_for_state(resultant_state)
    prompt: str = prompt__validate_action(requested_action["action"])
    pred_raw_response: str = fetch_openai_vision_completion(prompt, [encoded_image_before, encoded_image_after])

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_taken']
    except:
        pred_rationale = None
        pred_is_met = None

    return {
        # metadata
        "init_state_id" : init_state["id"],
        "action_id" : requested_action["id"],
        "path_to_screenshot_before": path_to_screenshot_before,
        "path_to_screenshot_after": path_to_screenshot_after,
        # gt
        "requested_action": requested_action["action"],
        # preds
        "pred_rationale": pred_rationale,
        "pred_action_taken" : pred_is_met,
        "pred_raw_response": pred_raw_response,
    }

def validate_task_vision(state_seq: List[Any], task: str) -> Dict[str, str]:
    ## Simple validator function that takes as input the sequence of states and the task, and determines if it succeeded.
    prompt_sequence = build_screenshot_prompt_sequence(state_seq)
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_with_vision__intro(task)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_with_vision__close()
        }]
    }
    # Feed (S, S', S'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_sequence + [close_prompt]
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["visual_questions"]
    except:
        pred_rationale = None
        pred_is_met = None
        pred_questions = None

    return {
        # metadata
        "task_description": task,
        # preds
        "pred_visual_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    }
    
def validate_task_text_vision(text_result: Dict[str,str], vision_result: Dict[str,str], task: str) -> Dict[str, str]:
    try:
        # you need to put the prompt together ruhana
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__validate_with_text_vision__intro(task)
            }] # type: ignore
        }
        
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__validate_with_text_vision__close()
            }] # type: ignore
        }
        
        def was_complete_helper(was_complete: bool)-> str:
            if not was_complete:
                return "not"
            return ""
        
        text_prompt = f"The text evaluator believes the task was {was_complete_helper(text_result['pred_task_completed'])} completed. This is because {text_result['pred_rationale'].lower()}. To better understand the text validators', here is a dictionary of question and answers which the text validator has answered: {text_result['pred_reasoning_questions']}."
        vision_prompt = f"The vision evaluator believes the task was {was_complete_helper(vision_result['pred_task_completed'])} completed. This is because {vision_result['pred_rationale'].lower()}. To better understand the text validators', here is a dictionary of question and answers which the text validator has answered: {vision_result['pred_visual_questions']}."
        prompt_sequence = build_text_prompt_sequence([text_prompt, vision_prompt])
        
        messages: List[str] = [intro_prompt] + prompt_sequence + [close_prompt]
        pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-turbo-preview', temperature=0.0)
        
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["reasoning_questions"]
    except:
        pred_rationale = None
        pred_is_met = None
        pred_raw_response = None
        pred_questions = None

    return {
        # metadata
        "task_description": task,
        "pred_reasoning_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    } # type: ignore    
    
def validate_task_raw_vision_text(text_seq: List[Any], vision_seq: List[Any], task: str) -> Dict[str, str]:
    try:
        vision_result: Dict[str, str] = validate_task_vqa(vision_seq, task=task)
        text_result: Dict[str, str] = validate_task_text(text_seq, task=task)
        result: Dict[str, str] = validate_task_vqa_text(text_result=str(text_result), vision_result=str(vision_result), task=task)
        return result
        
    except:
        return {
        # metadata
        "task_description": task,
        "pred_rationale": None,
        "pred_task_completed" : None,
        "pred_raw_response": None
    } # type: ignore    

def validate_task_text(state_seq: List[Any], task: str) -> Dict[str, str]:
    ## Simple validator function that takes as input the sequence of states and the task, and determines if it succeeded.
    prompt_sequence = build_text_prompt_sequence(state_seq)
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_with_text__intro(task)
        }] # type: ignore
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_with_text__close()
        }] # type: ignore
    }  
    
    # Feed (S, S', S'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_sequence + [close_prompt]
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-turbo-preview', temperature=0.0)

    # Evaluate
    try:
        pred_json = robust_json_loader(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["reasoning_questions"]
    except:
        pred_rationale = None
        pred_is_met = None
        pred_questions = None

    return {
        # metadata
        "task_description": task,
        "pred_reasoning_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    } # type: ignore
    
def main(args): # type: ignore
    is_action: bool = args.is_action
    is_task_completion: bool = args.is_task_completion
    use_vqa: bool = args.use_vqa
    path_to_images: str = args.path_to_images
    task: str = args.task
    requested_action: str = args.requested_action
    
    assert sum([is_action, is_task_completion]) == 1, "Must specify EXACTLY ONE of --is_action or --is_task_completion"
    
    # Execute eval
    output_file_name: str = ""
    if is_action:
        init_state = {"id":0,"path_to_screenshot":f"{path_to_images}/0.png"}
        resultant_state = {"id": 2, "path_to_screenshot":f"{path_to_images}/1.png"}
        requested_action = {"id": 1, "action": requested_action}

        out = validate_action(init_state, requested_action, resultant_state)
    elif is_task_completion:
        state_seq = []
        file_num = 0
        filelist = [filename for filename in os.listdir(path_to_images) if filename.endswith(".png")]
        filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for file in filelist:
            if file.endswith(".png"):
                state_seq.append({"id":file_num, "path_to_screenshot": os.path.join(path_to_images,file)})
                file_num+=1
        if use_vqa:
            print("Using VQA")
            out = validate_task_vqa(state_seq, task)
        else:
            print("Without VQA")
            out = validate_task(state_seq, task)
    else:
        raise ValueError("Must specify either --is_action or --is_task_completion")
    return out

if __name__ == "__main__":
    args, __ = parse_args()
    print(main(args))