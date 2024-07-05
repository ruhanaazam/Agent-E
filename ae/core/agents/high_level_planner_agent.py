from string import Template
from ae.core.post_process_responses import final_reply_callback_planner_agent as print_message_as_planner  # type: ignore
import autogen  # type: ignore
from ae.core.memory.static_ltm import get_user_ltm
from ae.core.skills.get_user_input import get_user_input
from ae.core.prompts import LLM_PROMPTS
from datetime import datetime
from autogen import Agent  # type: ignore
from autogen import ConversableAgent # type: ignore
from autogen import OpenAIWrapper  # type: ignore
class PlannerAgent:
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
        
        self.agent = autogen.AssistantAgent(
            name="planner_agent",
            system_message=system_message,
            llm_config={
                "config_list": config_list,
                "cache_seed": None,
                "temperature": 0.0
            },
        )

    def __get_ltm(self):
        """
        Get the the long term memory of the user.
        returns: str | None - The user LTM or None if not found.
        """
        return get_user_ltm()

