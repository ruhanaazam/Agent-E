import asyncio
import json
import os
import time
import tempfile
import traceback
from string import Template
from time import time_ns
from typing import Any

import autogen  # type: ignore
from autogen.agentchat.groupchat import GroupChatManager
import nest_asyncio  # type: ignore
import openai

#from autogen import Cache
from dotenv import load_dotenv

from ae.config import SOURCE_LOG_FOLDER_PATH
from ae.core.agents.browser_nav_agent import BrowserNavAgent
from ae.core.agents.high_level_planner_agent import PlannerAgent  
from ae.core.agents.validator_agent import ValidationAgent
from ae.core.prompts import LLM_PROMPTS
import logging
from ae.utils.logger import logger
from ae.utils.autogen_sequential_function_call import UserProxyAgent_SequentialFunctionExecution
from ae.utils.response_parser import getLastPlannerMessage, isTerminate, group_manager_error_check
from ae.core.skills.get_url import geturl
import nest_asyncio # type: ignore
from ae.core.post_process_responses import final_reply_callback_planner_agent as print_message_from_planner  # type: ignore
from autogen.agentchat.contrib.capabilities import transform_messages, transforms

nest_asyncio.apply()  # type: ignore

class AutogenWrapper:
    """
    A wrapper class for interacting with the Autogen library.

    Args:
        max_chat_round (int): The maximum number of chat rounds.

    Attributes:
        number_of_rounds (int): The maximum number of chat rounds.
        agents_map (dict): A dictionary of the agents that are instantiated in this autogen instance.

    """

    def __init__(self, max_chat_round: int = 100):
        self.number_of_rounds = max_chat_round

        self.agents_map: dict[str, UserProxyAgent_SequentialFunctionExecution | autogen.AssistantAgent | autogen.ConversableAgent ] | None = None

        self.config_list: list[dict[str, str]] | None = None
        self.chat_logs_dir: str = SOURCE_LOG_FOLDER_PATH
        self.task_start_time = time.time()
    

    @classmethod
    async def create(cls, agents_needed: list[str] | None = None, max_chat_round: int = 100):
        """
        Create an instance of AutogenWrapper.

        Args:
            agents_needed (list[str], optional): The list of agents needed. If None, then ["user", "browser_nav_executor", "planner_agent", "browser_nav_agent"] will be used.
            max_chat_round (int, optional): The maximum number of chat rounds. Defaults to 50.

        Returns:
            AutogenWrapper: An instance of AutogenWrapper.

        """
        print(f">>> Creating AutogenWrapper with {agents_needed} and {max_chat_round} rounds.")
        if agents_needed is None:
            agents_needed = ["user", "browser_nav_executor", "planner_agent", "browser_nav_agent", "validator_agent"]
            
        # Create an instance of cls
        self = cls(max_chat_round)
        load_dotenv()
        os.environ["AUTOGEN_USE_DOCKER"] = "False"

        autogen_model_name = os.getenv("AUTOGEN_MODEL_NAME")
        if not autogen_model_name:
            autogen_model_name = "gpt-4-turbo"
            logger.warning(f"Cannot find AUTOGEN_MODEL_NAME in the environment variables, setting it to default {autogen_model_name}.")

        autogen_model_api_key = os.getenv("AUTOGEN_MODEL_API_KEY")
        if autogen_model_api_key is None:
            logger.warning("Cannot find AUTOGEN_MODEL_API_KEY in the environment variables.")
            if not os.getenv('OPENAI_API_KEY'):
                logger.error("Cannot find OPENAI_API_KEY in the environment variables.")
                raise ValueError("You need to set either AUTOGEN_MODEL_API_KEY or OPENAI_API_KEY in the .env file.")
            else:
                autogen_model_api_key = os.environ['OPENAI_API_KEY']
        else:
            logger.info(f"Using model {autogen_model_name} for AutoGen from the environment variables.")
        model_info = {'model': autogen_model_name, 'api_key': autogen_model_api_key}

        if os.getenv("AUTOGEN_MODEL_BASE_URL"):
            model_info["base_url"] = os.getenv("AUTOGEN_MODEL_BASE_URL") # type: ignore

        env_var: list[dict[str, str]] = [model_info]
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
            json.dump(env_var, temp)
            temp_file_path = temp.name
        self.config_list = autogen.config_list_from_json(env_or_file=temp_file_path, filter_dict={"model": {autogen_model_name}}) # type: ignore
        
        # Intialize inividual agents
        self.agents_map = await self.__initialize_agents(agents_needed) 
        
        planner_agent = self.agents_map["planner_agent"]
        self.manager = self.agents_map["manager"]
              
        def trigger_nested_chat(manager: autogen.ConversableAgent) -> bool:
            print("Checking trigger_nested_chat()...")
            
            messages = manager.chat_messages[planner_agent] # this chat is shared
            
            if group_manager_error_check(messages):
                logging.error(f"Group manager did not maintain order!, {messages}")
                raise Exception("Group manager did not maintain order!")

            # Get the last message from the planner
            lastPlanMessage = getLastPlannerMessage(messages)
            if not lastPlanMessage:
                return False # Either plan or validation is missing
            
            next_step = lastPlanMessage.get('next_step', None)
            plan = lastPlanMessage.get('plan', None)
        
            if plan is not None:
                print_message_from_planner("Plan: "+ str(plan))
            if next_step is None: 
                print_message_from_planner("Received no next step response, terminating..") 
                return False
            return True

        def get_url() -> str:
            return asyncio.run(geturl())

        def my_custom_summary_method(sender: autogen.ConversableAgent,recipient: autogen.ConversableAgent, summary_args: dict ) : # type: ignore
            messages_str_keys = {str(key): value for key, value in sender.chat_messages.items()} # type: ignore
            self.__save_chat_log(list(messages_str_keys.values())[0]) # type: ignore
            last_message=recipient.last_message(sender)["content"] # type: ignore
            if not last_message or last_message.strip() == "": # type: ignore
                return "I received an empty message. Try a different approach."
            elif "##TERMINATE TASK##" in last_message:
                last_message=last_message.replace("##TERMINATE TASK##", "") # type: ignore
                last_message=last_message+" "+  get_url() # type: ignore
                print_message_from_planner("Response: "+ last_message) # type: ignore
                return last_message #  type: ignore
            return recipient.last_message(sender)["content"] # type: ignore
        
        def reflection_message(recipient, messages, sender, config): # type: ignore
            last_message=getLastPlannerMessage(messages)
            next_step = last_message.get('next_step', None)
            if next_step is None: 
                print ("Message to nested chat returned None")
                return None
            else:
                next_step = next_step.strip() +" " + get_url() # type: ignore
                print(f"Message to nested chat: {next_step}")
                return next_step # type: ignore

        print(f">>> Registering nested chat. Available agents: {self.agents_map}")
        self.agents_map["user"].register_nested_chats( # type: ignore
            [
                {
            "sender": self.agents_map["browser_nav_executor"],
            "recipient": self.agents_map["browser_nav_agent"],
            "message":reflection_message,  
            "max_turns": self.number_of_rounds,
            "summary_method": my_custom_summary_method,
                }   
            ],
            trigger=trigger_nested_chat, # type: ignore
        )

        return self

    def set_logger_to_debug(self, file_name: str):
        '''
        This function configers the logger (at the level of the autogen wrapper).
        Parameter:
            file_name (str): Name of the file where the logs will be saved
        '''
        logging.basicConfig(
            level=logging.DEBUG, # change level here or use set_log_level() to change it
            format="[%(asctime)s] %(levelname)s {%(filename)s:%(lineno)d}  - %(message)s", filename=file_name, filemode='a'
        )
        return 
    
    def set_logger_to_info(self, file_name: str):
        '''
        This function configers the logger (at the level of the autogen wrapper).
        Parameter:
            file_name (str): Name of the file where the logs will be saved
        '''
        logging.basicConfig(
            level=logging.INFO, # change level here or use set_log_level() to change it
            format="[%(asctime)s] %(levelname)s {%(filename)s:%(lineno)d}  - %(message)s", filename=file_name, filemode='a'
        )
        return 
    
    def get_chat_logs_dir(self) -> str|None:
        """
        Get the directory for saving chat logs.

        Returns:
            str|None: The directory path or None if there is not one

        """
        return self.chat_logs_dir

    def set_chat_logs_dir(self, chat_logs_dir: str):
        """
        Set the directory for saving chat logs.

        Args:
            chat_logs_dir (str): The directory path.

        """
        self.chat_logs_dir = chat_logs_dir


    def __save_chat_log(self, chat_log: list[dict[str, Any]]):
        chat_logs_file = os.path.join(self.get_chat_logs_dir() or "", f"nested_chat_log_{str(time_ns())}.json")
        # Save the chat log to a file
        with open(chat_logs_file, "w") as file:
            json.dump(chat_log, file, indent=4)


    async def __initialize_agents(self, agents_needed: list[str]):
        """
        Instantiate all agents with their appropriate prompts/skills.

        Args:
            agents_needed (list[str]): The list of agents needed, this list must have user_proxy in it or an error will be generated.

        Returns:
            dict: A dictionary of agent instances.

        """
        agents_map: dict[str, UserProxyAgent_SequentialFunctionExecution  | autogen.ConversableAgent]= {}
        self.agents_needed = agents_needed
                
        user_delegate_agent = await self.__create_user_delegate_agent()
        agents_map["user"] = user_delegate_agent
        agents_needed.remove("user")
        
        browser_nav_executor = self.__create_browser_nav_executor_agent()
        agents_map["browser_nav_executor"] = browser_nav_executor
        agents_needed.remove("browser_nav_executor")

        # make validator agent either way but set to modaility to none if it is not 
        validator_agent = self.__create_validator_agent()
        agents_map["validator_agent"] = validator_agent
        if "validator_agent" not in agents_needed:
               agents_map["validator_agent"] = validator_agent
               validator_agent.modality = "none"
        
        for agent_needed in agents_needed:
            if agent_needed == "browser_nav_agent":
                browser_nav_agent: autogen.ConversableAgent = self.__create_browser_nav_agent(agents_map["browser_nav_executor"] )
                agents_map["browser_nav_agent"] = browser_nav_agent
                agents_map["validator_agent"] = validator_agent
            elif agent_needed == "planner_agent":
                planner_agent = self.__create_planner_agent(user_delegate_agent)
                agents_map["planner_agent"] = planner_agent
            elif agent_needed == "validator_agent":
                continue
            else:
                raise ValueError(f"Unknown agent type: {agent_needed}")
        
        # Initalize group chat manager 
        manager_agent: GroupChatManager = self.__create_group_chat_manager_agent(agents_map)
        agents_map["manager"] = manager_agent
        
        # TODO: Add handling of long context length, currently getting this error from the code below:
        # # Tranformation to shorten context lenght
        # context_handling = transform_messages.TransformMessages(
        #     transforms=[
        #         transforms.MessageTokenLimiter(max_tokens=128000)
        #     ]
        # )
        
        # # Apply transformation to each agent
        # for agent_name in agents_map.keys():
        #     if agent_name == "browser_nav_executor":
        #         continue
        #     else:
        #         context_handling.add_to_agent(agents_map[agent_name])
                
        
        return agents_map


    async def __create_user_delegate_agent(self) -> autogen.ConversableAgent:
        """
        Create a ConversableAgent instance.

        Returns:
            autogen.ConversableAgent: An instance of ConversableAgent.

        """
        def is_planner_termination_message(x: dict[str, str])->bool: # type: ignore
            '''
            Determines if the current task is completed.
            This function is called before each user agent call. 
            
            Parameter:
                x: The last message posted by an agent.
            
            Return:
                Boolean determining if the intent was completed.
            '''
            print(f"Checking is_planner_termination_message()...")
            if "validator_agent" not in self.agents_needed:
                # just check if termination was called
                # should be planner agent here
                name = x.get("name", None)
                if name == "planner_agent":
                    content:Any = x.get("content", "")
                    if "\"terminate\": \"yes\"" in content:
                        return True 
            else:
                if x.get("name", None) != "validator_agent":
                    return False 
                    # This might be the place to throw an error... if should_terminate is false, then this is bad, the user agent should never been called...
                else:
                    content:Any = x.get("content", "")
                    should_terminate = ("The task was completed" in content)
                    print(f"Should terminate ... {should_terminate}")
                    if not should_terminate:
                        raise Exception('User agent called after validator agent before termination!')
                    return should_terminate # type: ignore
            return False 
        task_delegate_agent = autogen.ConversableAgent(
            name="user",
            llm_config=False, 
            system_message=LLM_PROMPTS["USER_AGENT_PROMPT"],
            is_termination_msg=is_planner_termination_message, # type: ignore
            human_input_mode="NEVER", 
            max_consecutive_auto_reply=self.number_of_rounds,
        )
        return task_delegate_agent

    def __create_browser_nav_executor_agent(self):
        """
        Create a UserProxyAgent instance for executing browser control.

        Returns:
            autogen.UserProxyAgent: An instance of UserProxyAgent.

        """
        def is_browser_executor_termination_message(x: dict[str, str])->bool: # type: ignore
             tools_call:Any = x.get("tool_calls", "")
             if tools_call :
                return False
             else:
                return True
        
        browser_nav_executor_agent = UserProxyAgent_SequentialFunctionExecution(
            name="browser_nav_executor",
            is_termination_msg=is_browser_executor_termination_message,
            human_input_mode="NEVER",
            llm_config=None,
            max_consecutive_auto_reply=self.number_of_rounds,
            code_execution_config={
                                "last_n_messages": 1,
                                "work_dir": "tasks",
                                "use_docker": False,
                                },
        )
        print(">>> Created browser_nav_executor_agent:", browser_nav_executor_agent)
        return browser_nav_executor_agent

    def __create_browser_nav_agent(self, user_proxy_agent: UserProxyAgent_SequentialFunctionExecution) -> autogen.ConversableAgent:
        """
        Create a BrowserNavAgent instance.

        Args:
            user_proxy_agent (autogen.UserProxyAgent): The instance of UserProxyAgent that was created.

        Returns:
            autogen.AssistantAgent: An instance of BrowserNavAgent.

        """
        browser_nav_agent = BrowserNavAgent(self.config_list, user_proxy_agent) # type: ignore
        #print(">>> browser agent tools:", json.dumps(browser_nav_agent.agent.llm_config.get("tools"), indent=2))
        return browser_nav_agent.agent

    def __create_planner_agent(self, assistant_agent: autogen.ConversableAgent):
        """
        Create a Planner Agent instance. This is mainly used for exploration at this point

        Returns:
            autogen.AssistantAgent: An instance of PlannerAgent.
        """
        planner_agent = PlannerAgent(self.config_list, assistant_agent) # type: ignore
        return planner_agent
    
    def __create_validator_agent(self,):
        # get log directory for current task
        log_dir = self.get_chat_logs_dir()
        
        # intialize the validator agent
        validator_agent = ValidationAgent(
            name="validator_agent",
            log_dir=log_dir,
            human_input_mode="NEVER",
            task_start_time=self.task_start_time
        )
        return validator_agent
    
    def __create_group_chat_manager_agent(self, agents_map: dict[str, UserProxyAgent_SequentialFunctionExecution  | autogen.ConversableAgent]) -> GroupChatManager:
        # Intialize group chat and group chat manager
        user_agent = agents_map["user"]
        planner_agent = agents_map["planner_agent"]
        validator_agent = agents_map["validator_agent"]
        
        def state_transition(last_speaker, groupchat):
            print(f"Speaker is {last_speaker.name}")
            messages = groupchat.messages
            involk_validator = (validator_agent.modality != "none")
            
            # After task has been validated, called user or planner agent
            if last_speaker is validator_agent:
                message = messages[-1]
                content = message.get("content", None)
                if "The task was completed" in content:
                    return user_agent # The user agent is expected to terminate the task
                else:
                    return planner_agent # Continue planning with new feedback
                
            # Call validation when planner outputs terminate flag
            shouldTerminate = isTerminate(messages) # checks the last planner message for a termination message
            if last_speaker is planner_agent and shouldTerminate and involk_validator:
                return validator_agent
            if last_speaker is user_agent and shouldTerminate and involk_validator: 
               return validator_agent 
            
            # Until task is terminated, go between validation and user agent
            if last_speaker is user_agent and not shouldTerminate: 
                return planner_agent
            if last_speaker is planner_agent:
                return user_agent
            return None
        
        groupchat = autogen.GroupChat(
            agents=[user_agent, planner_agent, validator_agent], 
            messages=[], 
            max_round=12,
            speaker_selection_method=state_transition,
            )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat, 
            llm_config={
                    "config_list": self.config_list,
                    "cache_seed": None,
                    "temperature": 0.0,
                    "seed": 1234
                },
            ) 
        return manager

    async def process_command(self, command: str, current_url: str | None = None) -> autogen.ChatResult | None:
        """
        Process a command by sending it to one or more agents.

        Args:
            command (str): The command to be processed.
            current_url (str, optional): The current URL of the browser. Defaults to None.

        Returns:
            autogen.ChatResult | None: The result of the command processing, or None if an error occurred. Contains chat log, cost(tokens/price)

        """
        # start tracking the start time of the current task
        self.task_start_time = time.time()
        
        current_url_prompt_segment = ""
        if current_url:
            current_url_prompt_segment = f"Current Page: {current_url}"

        prompt = Template(LLM_PROMPTS["COMMAND_EXECUTION_PROMPT"]).substitute(command=command, current_url_prompt_segment=current_url_prompt_segment)
        logger.info(f"Prompt for command: {prompt}")
        #with Cache.disk() as cache:
        # try:
        if self.agents_map is None:
            raise ValueError("Agents map is not initialized.")
        print(self.agents_map["browser_nav_executor"].function_map) # type: ignore
        
        # Clear chat history for each (new) command
        for agent in self.agents_map.values():
            agent.clear_history()
        
        result=await self.agents_map["user"].a_initiate_chat( # type: ignore
            self.manager,
            max_turns=self.number_of_rounds,
            clear_history=True,
            message=prompt,
            silent=False,
            cache=None,
        )
        # reset usage summary for all agents after each command            for agent in self.agents_map.values():
        if hasattr(agent, "client") and agent.client is not None:
                agent.client.clear_usage_summary() # type: ignore
        return result
        # except openai.BadRequestError as bre:
        #     logger.error(f"Unable to process command: \"{command}\". {bre}")
        #     traceback.print_exc()

