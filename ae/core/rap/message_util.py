import json

# Parse and load the prior plan execution
def load_main_chat(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    message_json = list(json_data.values())[0]
    return message_json

def format_main_chat(message_json):
    # Load each message
    fomatted_messages = []
    for message in message_json:
        role = message.get("role", "")
        content = message.get("content", "")
        fomatted_messages.append(f"{role}: {content}")
    return message_json

def get_last_plan(message_json):
    # get index of all the validator calls
    validator_index = []
    message_json[-1].get("name")
    for i, item in enumerate(message_json):
        if item.get("name") == 'validator_agent':
            validator_index.append(i)
    
    
    last_plan_index = 1
    if len(validator_index) > 1:
        last_plan_index = validator_index[-2] + 1
    
    last_plan = message_json[last_plan_index].get("content", "")
    return last_plan

    
    