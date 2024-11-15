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