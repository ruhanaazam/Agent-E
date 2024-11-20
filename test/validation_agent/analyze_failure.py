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

# TODO: modify the prompt(s)


def analyze_failure_vision(state_seq: List[Any], task: str, model: str, final_response: str| None = None) -> Dict[str, str]:
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
    pred_raw_response: str = _fetch_openai_completion(messages, model=model, temperature=0.0, seed=1234)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["visual_questions"]
    except Exception as e:
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

def analyze_failure_text(state_seq: List[Any], task: str, model: str) -> Dict[str, str]:
    ## Simple validator function that takes as input the sequence of states (a list of text) and the task, and determines if it succeeded.
    
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
    pred_raw_response: str = _fetch_openai_completion(messages, model=model, temperature=0.0, seed=1234) # model='gpt-4-turbo-preview'

    # Evaluate
    try:
        pred_json = robust_json_loader(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        pred_questions: List[Any] = pred_json["reasoning_questions"]
    except Exception as e:
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