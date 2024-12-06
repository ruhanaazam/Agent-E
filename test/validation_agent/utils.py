### Subset of helper functions from eclair-agents
import json
import subprocess
import time
import openai
import base64
import traceback
from typing import Dict, Any, Tuple, List
from PIL import Image
import os

SYSTEM_PROMPT: str = "You are a helpful assistant that automates digital workflows."

def encode_image(path_to_img: str):
    """Base64 encode an image"""
    with open(path_to_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_screenshot_for_state(state: Dict[str, Any]) -> Tuple[str, str]:
    path_to_screenshot: str = state["path_to_screenshot"]
    encoded_image: str = encode_image(path_to_screenshot)
    return path_to_screenshot, encoded_image

def fetch_openai_vision_completion(
    prompt: str, base64_images: List[str], model: str, **kwargs
) -> str:
    """Helper function to call OpenAI's Vision API. Handles rate limit errors and other exceptions"""
    messages: List[Any] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
                for img in base64_images
            ]
            + [{"type": "text", "text": prompt}],
        },
    ]
    return _fetch_openai_completion(messages, model=model, **kwargs)

def _fetch_openai_completion(messages: List[Any], model: str, **kwargs) -> str | None:
    """Helper function to call OpenAI's Vision API. Handles rate limit errors and other exceptions"""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            model=model,
            max_tokens=4096,
            **kwargs,
        )
    except openai.RateLimitError:
        print("Rate limit exceeded -- waiting 1 min before retrying")
        time.sleep(60)
        return _fetch_openai_completion(messages, model, **kwargs)
    except openai.APIError as e:
        traceback.print_exc()
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
        return None
    return response.choices[0].message.content

def build_screenshot_prompt_sequence(state_seq: List[Any]) -> List[str]:
    # Loop through states
    prompt_sequence: List[str] = []
    for item in state_seq:
        path_to_screenshot, encoded_image = load_screenshot_for_state(item)
        prompt_sequence.append({
            "role": "user", 
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            }],
        })
    return prompt_sequence

def build_text_prompt_sequence(state_seq: List[Any]) -> List[str]:
    prompt_sequence: List[str] = []
    for item in state_seq:
        prompt_sequence.append({
            "role": "user", 
            "content": [{
                "type": "text",
                "text": str(item)
            }],
        }) # type: ignore
    return prompt_sequence

def compress_png(file_path:str, max_size_mb:int=20, max_height:int =2048, max_width:int=768, reduce_factor:float=0.9)-> bool:
    short_side_limit:int= min(max_height, max_width)
    long_side_limit:int= max(max_height, max_width)
    
    try:
        # get image size and dimensions
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024.0)  # type: ignore
        # with Image.open(file_path) as img:
        #         width, height = img.size
        
        while file_size_mb >= max_size_mb:  #or min(width, height) >= short_side_limit or max(width, height) >= long_side_limit:
            print(f"Compressing {file_path} (Initial Size: {file_size_mb:.2f} MB)")
            with Image.open(file_path) as img:
                width, height = img.size
                new_width = int(width * reduce_factor)
                new_height = int(height * reduce_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.save(file_path, optimize=True)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024.0)  # type: ignore
                print(f"Resized to: {new_width}x{new_height}, Size: {file_size_mb:.2f} MB")
        #print(f"Final Size of {file_path}: {file_size_mb:.2f} MB")
        return file_size_mb < max_size_mb
    except Exception as e:
        print(f"Error compressing {file_path}: {e}")
        return False

def list_items_in_folder(path: str)-> list[str]:
    # TODO : remove try except, that way error is caught elsewhere
    # TODO: consider logging the error somewhere
    try:
        items = os.listdir(path)
        items_with_mtime = [(item, os.path.getmtime(os.path.join(path, item))) for item in items]
        items_with_mtime.sort(key=lambda x: x[0])
        sorted_items = [item for item, mtime in items_with_mtime]
        return sorted_items
    except FileNotFoundError:
        print(f"The path {path} does not exist.")
        return []
    except NotADirectoryError:
        print(f"The path {path} is not a directory.")
        return []
    except PermissionError:
        print(f"Permission denied to access {path}.")
        return []
    
def get_screenshot_paths(file_path:str, compress:bool=True)-> list[str]:
    '''
    Will get all screenshots in a folder and compress them.
    '''
    # Get all screenshots for a task
    screenshot_names: list[str]= list_items_in_folder(file_path)     
   
    # # Check that screenshots exist for evaluation
    if len(screenshot_names) == 0:
        print("Warning: no screenshots found!")
    
    # Load and compress screenshots
    state_seq: list[str] = []
    for screenshot_name in screenshot_names:
        screenshot_path = f"{file_path}/{screenshot_name}"
        if compress:
            compress_sucessful = compress_png(screenshot_path)
            if not compress_sucessful:
                print(f"Screenshot {screenshot_path} not added due to issue with compressing image...")
            else:
               state_seq.append({"path_to_screenshot": f"{screenshot_path}"}) # type: ignore
        else: 
            state_seq.append({"path_to_screenshot": f"{screenshot_path}"}) # type: ignore
    return state_seq

def get_chat_sequence(messages: List[Dict[str, str]]):
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
    final_message = get_final_response(messages=messages)
    if final_message:
        chat_sequence.append(final_message)
    return chat_sequence

def get_intent(messages: List[Dict[str, str]])-> str | None:
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

def get_final_response(messages: List[Dict[str, str]]):
    # Append final statement from the planner
    message = None
    for message in reversed(messages):
        #if message.get("role") == "assistant" or message.get("name") == "planner_agent":
        content = message.get('content', None)
        if content and "final_response" in content:
            try:
                content = json.loads(content)
                message = "The closing statement: " + content.get('final_response', "") + "."
                return message
            except Exception as e:
                print(f"Exception getting user chat, likely due to unexpected formatting of {content}: {e} ")# add without final message if there is a parsing issue...
    return message

def robust_json_loader(json_str):
    '''
    This function will correct any incorrectly formatted json objects then load it.
    (Can be helpful when you have json objects created by an LLM)
    '''
    try:
        json_object = json.loads(json_str)
    except Exception as e:
        model_name = os.environ['AUTOGEN_MODEL_NAME']
        print(f"Json formatting issue... Calling {model_name} to reformat...")

        #call gpt-4 to fix
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],)
        prompt = f"Your job is to correct a string to be a perfect json object so it can be parsed. The given string is almost is json. Please help me make this string into proper json format. Please only respond with a plain text and nothing else: {json_str} "
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.0
        )
        new_json_str:str = response.choices[0].message.content
        json_object = json.loads(new_json_str)
    return json_object