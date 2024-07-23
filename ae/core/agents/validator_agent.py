from errno import EXDEV
from autogen.agentchat import ConversableAgent
from typing import List, Dict, Optional, Any, Union
import asyncio

import sys
from ae.config import PROJECT_TEST_ROOT
sys.path.append(PROJECT_TEST_ROOT)
from validation_agent.validator import validate_task_text, validate_task_vision # type: ignore
from validation_agent.utils import get_screenshot_paths, get_chat_sequence, get_intent # type: ignore

class ValidationAgent(ConversableAgent):
    def __init__(self, name: str, modality: str="text",  log_dir: str | None=None, **kwargs):
        """
        Initialize the validation agent. This is a custom conversation agent.
        """ 
        self.modality:str = modality
        self.log_dir = log_dir
        self.screenshot_directory =  f"{log_dir}/snapshots" if log_dir else None
        super().__init__(name, **kwargs)
        return 
    
    async def a_generate_reply(self, messages: Optional[List[Dict[str, Any]]] = None,
                           sender: Optional["Agent"] = None,
                           **kwargs: Any) -> Union[str, Dict[str, Any], None]:
        
        # Call the synchronous generate_reply method
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, self.generate_reply, messages, sender, **kwargs)
        return reply
    
    def generate_reply(self, 
                       messages: Optional[List[Dict[str, Any]]] = None, 
                       sender: Optional["Agent"] = None, **kwargs: Any) -> Union[str, Dict, None]:
        
        '''
        Overides the generate reply function to prevent auto reply. 
        Instead only use user agent's messages and final plan to validate the plan. 
        '''
        
        #system_message = LLM_PROMPTS["VALIDATOR_AGENT_PROMPT"]
        #print(f">>> Planner system_message: {system_message}")
        #system_message = system_message + "\n" + f"Today's date is {datetime.now().strftime('%d %B %Y')}" 
        
        # Get the intent from messages
        messages = list(self.chat_messages.values())[0] # Note: This can probably be done in a better way
        intent = get_intent(messages=messages) #ignore
        
        # Evaluate the intent
        score_dict = {}
        if self.modality == "text":            
            # Get the relevant chat sequence to validate
            state_seq = get_chat_sequence(messages)  # type: ignore
            score_dict = validate_task_text(state_seq, intent) # type: ignore
            # TODO: limit the state_seq to be between now and last validation
            
        if self.modality == "vision":
            if not self.screenshot_directory:
                raise Exception("Screenshot directory is not set in validation_agent. Cannot proceed with vision-based evaluation.")
            screenshot_seq = get_screenshot_paths(self.screenshot_directory) # type: ignore
            score_dict = validate_task_vision(screenshot_seq, intent) # type: ignore
             # TODO: limit the state_seq to be between now and last validation
            
            
        #if self.modality == "test_vision":  
            # TODO: Implement text & vision self-validator
            
        
        # TODO: Play around with fixing this text response.
        if score_dict["pred_task_completed"]:
            response = "The task was completed successfully."   
        else:
            response = f"The task was not completed succesfully. {score_dict['pred_rationale']} Please come up with a new plan which is different than the previous attempted plan(s). This plan should take into account and avoid issue(s) from prior plan(s). You are allowed to attempt build plans which seemed less likely to work previously."   
        
        print(f"Validator Raw Response: {score_dict}")
        return {"content": response}
    
    def get_modality(self,)-> str:
        return self.modality
    
    def set_modality(self, new_modality: str):
        assert (new_modality in ["text", "vision", "text_vision"]), f"{new_modality} is not a valid modality for the validator"
        self.modality=new_modality
        return

    def get_screenshot_directory(self,):
        return self.screenshot_directory
    
    def set_screenshot_directory(self, screenshot_directory:str ):
        self.screenshot_directory = screenshot_directory
        return

