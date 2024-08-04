from ast import Raise
from string import Template
from typing import Optional, List, Dict, Any, Union
from ae.core.post_process_responses import final_reply_callback_planner_agent as print_message_as_planner  # type: ignore
import autogen  # type: ignore
from ae.core.memory.static_ltm import get_user_ltm
from ae.core.prompts import LLM_PROMPTS
from datetime import datetime
from autogen import Agent  # type: ignore
from autogen import ConversableAgent, AssistantAgent # type: ignore
from ae.utils.logger import logger
import json

class PlannerAgent(AssistantAgent):
    def __init__(self, config_list, user_proxy_agent:ConversableAgent): # type: ignore
        """
        Initialize the PlannerAgent and store the AssistantAgent instance
        as an instance attribute for external access.

        Parameters:
        - config_list: A list of configuration parameters required for AssistantAgent.
        - user_proxy_agent: An instance of the UserProxyAgent class.
        """
        user_ltm = self.__get_ltm()
        system_message = LLM_PROMPTS["PLANNER_AGENT_PROMPT"]
        print(f">>> Planner system_message: {system_message}")
        if user_ltm: #add the user LTM to the system prompt if it exists
            user_ltm = "\n" + user_ltm
            system_message = Template(system_message).substitute(basic_user_information=user_ltm)
        system_message = system_message + "\n" + f"Today's date is {datetime.now().strftime('%d %B %Y')}" 
        
        super().__init__(name="planner_agent", 
                         system_message=system_message, 
                         llm_config={
                            "config_list": config_list,
                            "cache_seed": None,
                            "temperature": 0.0,
                            "seed": 1234
                            }
                        )

    def __get_ltm(self):
        """
        Get the the long term memory of the user.
        returns: str | None - The user LTM or None if not found.
        """
        return get_user_ltm()
    
    def generate_reply(self, messages: Optional[List[Dict[str, Any]]] = None,
                   sender: Optional["Agent"] = None,
                   **kwargs: Any) -> Union[str, Dict[str, Any], None]:
        raw_output = super().generate_reply(messages, sender, **kwargs)
        
        # Check that output is formatted as json
        try:
            json.loads(raw_output)
        except Exception as e:
            logger.warning(f"Planner outputted a non-json object: {raw_output}")
            logger.info("Attempting to reformat plan into json object...")
            
            # try to reformat plan
            formatted_plan = self._fix_plan_format(raw_output)
            return formatted_plan
            
        return raw_output
    
    
    async def a_generate_reply(self, messages: Optional[List[Dict[str, Any]]] = None,
                           sender: Optional["Agent"] = None,
                           **kwargs: Any) -> Union[str, Dict[str, Any], None]:
        raw_output = await super().a_generate_reply(messages, sender, **kwargs)
        
        # Check that output is formatted as json
        try:
            json.loads(raw_output)
        except Exception as e:
            logger.warning(f"Planner outputted a non-json object: {raw_output}")
            logger.info("Attempting to reformat plan into json object...")
            
            # try to reformat plan
            formatted_plan = self._fix_plan_format(raw_output)
            return formatted_plan
            
        return raw_output
    
    def _fix_plan_format(self, message: str)-> str:
        json_response = {}
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
        
        if "plan" in message and "next_step" in message and "terminate" in message:
            json_to_string = json.dumps(json_response)
            return json_to_string
        elif "terminate" in message and "final_response" in message:
            json_to_string = json.dumps(json_response)
            return json_to_string 
        else:
             raise Exception(f"Cannot format plan: {message}")