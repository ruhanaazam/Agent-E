from random import seed
from .prompts import (
    prompt_validate_with_vision_intro, prompt_validate_with_vision_close, 
    prompt_validate_with_text_intro, prompt_validate_with_text_close, 
    prompt_validate_with_text_vision_intro, prompt_validate_with_text_vision_close,
    prompt_validate_with_vision_final_response_intro, prompt_validate_with_vision_final_response_close,
    prompt_classifier_intro, prompt_classifier_close
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
from datetime import datetime
from test_utils import robust_json_loader
from ae.utils.logger import logger

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

def validate_task_vision(state_seq: List[Any], task: str, final_response: str| None = None) -> Dict[str, str]:
    ## Simple validator function that takes as input the sequence of states and the task, and determines if it succeeded.
    prompt_sequence = build_screenshot_prompt_sequence(state_seq)
    date_message = f"{datetime.now().strftime('%d %B %Y')}" 

    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt_validate_with_vision_intro(task, date_message)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt_validate_with_vision_close()
        }]
    }
    
    # Feed (S, S', S'', ...) -- i.e. all screenshots at once
    if final_response: 
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt_validate_with_vision_final_response_intro(task, date_message)
            }]
        }
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt_validate_with_vision_final_response_close()
            }]
        }
        final_response_formatted = build_text_prompt_sequence([final_response])
        messages: List[str] = [intro_prompt] + prompt_sequence + final_response_formatted +[close_prompt]
    else: 
        messages: List[str] = [intro_prompt] + prompt_sequence + [close_prompt]
        #print(f"model: gpt-4-turbo-preview")
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0, seed=1234) #gpt-4-vision-preview
    logger.info(f"Raw Validator Response: {pred_raw_response}")

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["visual_questions"]
    except Exception as e:
        logger.error(f"Error in parsing: {pred_raw_response}. Exception given: {e}")
        pred_rationale = None
        pred_is_met = True #default answer -- basically an answer given at random
        pred_questions = None
        pred_raw_response: f"Error in parsing: {pred_raw_response}. Exception given: {e}"

    return {
        # metadata
        "task_description": task,
        # preds
        "pred_visual_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    }
    
def validate_task_text_vision(text_sequence: List[Any], vision_seqence: List[Any], task: str) -> Dict[str, str]:
    try:
        # you need to put the prompt together ruhana
        date_message = f"{datetime.now().strftime('%d %B %Y')}" 
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt_validate_with_text_vision_intro(task, date_message)
            }] # type: ignore
        }
        
        close_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt_validate_with_text_vision_close()
            }] # type: ignore
        }
        
        text_prompt= build_text_prompt_sequence(["Text Observations:"]) + build_text_prompt_sequence(text_sequence)
        vision_prompt = build_text_prompt_sequence(["Vision Observations:"]) + build_screenshot_prompt_sequence(vision_seqence)
        messages: List[str] = [intro_prompt] + text_prompt + vision_prompt + [close_prompt]
        pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4o', temperature=0.0, seed=1234)
        logger.info(f"Raw Validator Response: {pred_raw_response}")
        
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["reasoning_questions"]
    except Exception as e:
        logger.error(f"Error in parsing: {pred_raw_response}. Exception given: {e}")
        pred_rationale = None
        pred_is_met = True #default answer -- basically an answer given at random
        pred_questions = None
        pred_raw_response: f"Error in parsing: {pred_raw_response}. Exception given: {e}"

    return {
        # metadata
        "task_description": task,
        "pred_reasoning_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    } # type: ignore   
     
def validate_task_raw_vision_text(text_seq: List[Any], vision_seq: List[Any], task: str) -> Dict[str, str]:
    vision_result: Dict[str, str] = validate_task_vqa(vision_seq, task=task)
    text_result: Dict[str, str] = validate_task_text(text_seq, task=task)
    result: Dict[str, str] = validate_task_vqa_text(text_result=str(text_result), vision_result=str(vision_result), task=task)
    return result

def validate_task_text(state_seq: List[Any], task: str) -> Dict[str, str]:
    ## Simple validator function that takes as input the sequence of states and the task, and determines if it succeeded.
    
    date_message = f"{datetime.now().strftime('%d %B %Y')}" 
    prompt_sequence = build_text_prompt_sequence(state_seq)
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt_validate_with_text_intro(task, date_message)
        }] # type: ignore
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt_validate_with_text_close()
        }] # type: ignore
    }  
    
    # Feed (S, S', S'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_sequence + [close_prompt]
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4o', temperature=0.0, seed=1234) # model='gpt-4-turbo-preview'
    logger.info(f"Raw Validator Response: {pred_raw_response}")

    # Evaluate
    try:
        pred_json = robust_json_loader(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["reasoning_questions"]
    except Exception as e:
        logger.error(f"Error in parsing: {pred_raw_response}. Exception given: {e}")
        pred_rationale = None
        pred_is_met = True #default answer -- basically an answer given at random
        pred_questions = None
        pred_raw_response: f"Error in parsing: {pred_raw_response}. Exception given: {e}"

    return {
        # metadata
        "task_description": task,
        "pred_reasoning_questions": pred_questions,
        "pred_rationale": pred_rationale,
        "pred_task_completed" : pred_is_met,
        "pred_raw_response": pred_raw_response
    } # type: ignore