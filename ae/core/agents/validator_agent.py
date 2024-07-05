from autogen.agentchat import ConversableAgent
from typing import List, Dict, Optional, Any, Union
import asyncio

import sys
from ae.config import PROJECT_TEST_ROOT
sys.path.append(PROJECT_TEST_ROOT)
from validation_agent.validator import validate_task_text

def get_state_sequence(messages: Optional[List[Dict[str, Any]]]):
    '''
    Creates a list with each user message followed by the final statement from the planner.
    '''
    # Get only user output from the chat
    chat_sequence: list[str] = []
    for item in messages:        
        role = item.get('role', None)
        message = item.get('content', None)
        if role == "user":
            chat_sequence.append(message)
    
    # Append final statement from the planner
    if messages[-1].get("role") == "assistant":
        content = messages[-1].get('content', None)
        try:
            content = json.loads(content)
            message = "The closing statement:" + content["final_response"]
            chat_sequence.append(message)
        except Exception as e:
            chat_sequence.append(content)
    return chat_sequence

def getIntent(messages: Optional[List[Dict[str, Any]]])-> str | None:
    '''
    Get the intent from the user agent's message
    '''
    message:str|None = messages[0]
    try:
        content: str | None = message.get("content", None)
        start:int = content.find('\"') + 1
        end:int = content.find('\"', start)
        intent:str = content[start:end]
        return intent
    except:
        print("No intent found in the chat messages.")
    return None

class ValidationAgent(ConversableAgent):
    def __init__(self, name: str, **kwargs):
        """
        Initialize the validation agent. This is a custom conversation agent.
        """ 
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
        messages = list(self.chat_messages.values())[0] # This can probably be done in a better way
        intent = getIntent(messages=messages)
        
        # Get the relevant chat sequence to validate
        state_seq = get_state_sequence(messages)
        score_dict = validate_task_text(state_seq, intent)
        
        # TODO: Play around with fixing this text response.
        response = "The task was completed successfully."
        if score_dict["pred_task_completed"]:
            response = "The task was completed successfully."   
        else:
            response = f"The task was not completed succesfully. {score_dict['pred_rationale']} Please try again."   
        
        print("Validator called successfully")
        return {"content": response}



