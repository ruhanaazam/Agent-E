from audioop import reverse
from email import message
import json
from typing import Dict, Any, List
import logging

def parse_response(message: str) -> Dict[str, Any]:
    """
    Parse the response from the browser agent and return the response as a dictionary.
    """
    # Parse the response content
    json_response = {}
    raw_messgae = message
    message = message.replace("\n", "\\n") # type: ignore
    # replace all \\n 
    message = message.replace("\\n", "")
    #if message starts with ``` and ends with ``` then remove them
    if message.startswith("```"):
        message = message[3:]
    if message.endswith("```"):
        message = message[:-3]
    if message.startswith("json"):
        message = message[4:]
    
    message = message.strip()
    try:
        json_response = json.loads(message)
    except:
        # If the response is not a valid JSON, try pass it using string matching. 
        #This should seldom be triggered
        if ("plan" in message and "next_step" in message):
            start = message.index("plan") + len("plan")
            end = message.index("next_step")
            json_response["plan"] = message[start:end].replace('"', '').strip()
        if ("next_step" in message and "terminate" in message):
            start = message.index("next_step") + len("next_step")
            end = message.index("terminate")
            json_response["next_step"] = message[start:end].replace('"', '').strip()
        if ("terminate" in message and "final_response" in message):
            start = message.index("terminate") + len("terminate")
            end = message.index("final_response")
            matched_string=message[start:end].replace('"', '').strip()
            if ("yes" in matched_string):
                json_response["terminate"] = "yes"
            else:
                json_response["terminate"] = "no"
            
            start=message.index("final_response") + len("final_response")
            end=len(message)-1
            json_response["final_response"] = message[start:end].replace('"', '').strip()

        elif ("terminate" in message):
            start = message.index("terminate") + len("terminate")
            end = len(message)-1
            matched_string=message[start:end].replace('"', '').strip()
            if ("yes" in matched_string):
                json_response["terminate"] = "yes"
            else:
                json_response["terminate"] = "no"
        logging.error(f"Error parsing JSON response {raw_messgae}")
        # TODO: add robust parser here
    return json_response

def getLastValidationMessage(messages):
    """
    Given a list of messages from a chat, will find the last message from the validator_agent 
    """
    for message in reversed(messages): 
        role = message.get("name", None)

        # Return the response given by the validator
        if role == "validator_agent":
            content = message.get("content", None)
            content_json=parse_response(content)
            return content_json
    return None

def getLastPlannerMessage(messages):
    """
    Given a list of messages from a chat, will find the last message from the planner_agent 
    """
    for message in reversed(messages): 
        role = message.get("name", None)

        # Return the parsed response given by the planner
        if role == "planner_agent":
            content = message.get("content", None)
            content_json=parse_response(content)
            return content_json
    return None

def isPlanValid(messages):
    """
    Given a list of messages from a chat, will determine if the latest plan has been validated.
    """
    content = getLastValidationMessage(messages)
    if content:
        return content.get("valid_plan", None) == "yes"
    return False

def isTerminate(messages):
    """
    Given a list of messages from a chat, will determine if the task should be terminated
    """
    content = getLastPlannerMessage(messages)
    if content:
        return content.get("terminate", None) == "yes"
    return False

def group_manager_error_check(messages:List[Dict[Any, Any]]):
    # There are generally issues in these cases:
    # The validator agent is not called after the planner terminates
    # the planner is not called after the validator agent says a plan is wrong
    # the user agent is not called after the validator agent says the plan is correct
    # all these outcomes usually lead to the user agent being called twice in a row when the valiator agent should be called
    
    # Checks if group manager ordering has been broken
    for i in range(len(messages)-1):
        role_0: str = messages[i].get("name", None)
        role_next: str = messages[i+1].get("name", None)
        
        # check user is not called twice in a row
        if role_0 == "user" and role_next == "user":
            return True
        
        if role_0 == "planner_agent" and role_next == "user":
            content = messages[i].get("content", None)
            content_json=parse_response(content)
            if content_json.get("terminate", None) == "yes":
                print(f"Error with message {i}")
                return True # validator was not called!
                
    return False