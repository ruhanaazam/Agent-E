from validation_agent.validator import *
from load_annotations import AnnotationLoader

# call the annotation_loader
LOG_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs"
RESULT_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results"
original_annotation = AnnotationLoader(LOG_PATH, RESULT_PATH)

# # given a certain task_id, get the screenshot sequence and main_chat sequence
# task_id = 0
# intent = original_annotation.get_intent(task_id)
# main_chat_sequence = original_annotation.get_high_level_trajectory(task_id)
# screenshot_paths = original_annotation.get_screenshot_paths(task_id)

# #validation_result = validate_task_text(main_chat_sequence, intent)
# validation_result = validate_task_vision(screenshot_paths, intent)
# print(validation_result['pred_task_completed'], validation_result['pred_rationale'])


# todo: need a better way to configure the gpt version and method ***
def validate_task_id(task_id, annotation_loader, config):
    intent = annotation_loader.get_intent(task_id)
    main_chat_sequence = annotation_loader.get_high_level_trajectory(task_id)
    screenshot_paths = annotation_loader.get_screenshot_paths(task_id)
    
    method = config.get("method", None)
    model = config.get("model", "gpt-4o")
    
    validation_result = {}
    if method == "main_chat_only":
        validation_result = validate_task_text(main_chat_sequence, intent)
    elif method == "nested_chat_only":
        pass
    elif method == "DOM_tree_only":
        pass
    elif method == "screenshots_only":
        #validate_task_vision(screenshot_paths, intent)
        pass
    elif method == "screenshot_final_response":
        pass 
    else:
        print("Error! No method specified!")
    
    return validation_result
   
task_id = 0 
config = {"method": "main_chat_only",
          "model": "gpt-4-vision-preview"}
print(validate_task_id(task_id, original_annotation, config))