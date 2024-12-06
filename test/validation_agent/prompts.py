prompt_validate_action: str = lambda task_action: f"""# Task
You are an RPA bot that navigates digital UIs like a human. Your job is to validate that a certain action was successfully taken.

# Action
The action that was supposed to be taken was: "{task_action}"

# Question

The first screenshot shows the digital UI BEFORE the action was supposedly taken.
The second screenshot shows the digital UI AFTER the action was supposedly taken.

Given the change between the screenshots, was the action successfully taken? Be lenient and assume that the action was taken if the UI is "close enough" to the expected UI.

Answer in the JSON format:
{{
    "rationale": <rationale>,
    "was_taken": <true/false>
}}

Answer:"""

prompt_validate_with_vision_intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of screenshots.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Here are the screenshots of the workflow:"""

prompt_validate_with_vision_close: str = lambda : f"""
# Instructions

Given what you observe in the previous sequence of screenshots, was the workflow successfully completed? 
To determine this, derive few visual questions from the task description that upon answering will help decide if the workflow was successfully completed.
If the workflow is asking a question, consider it completed successfully if you could deduce the answer to the question by viewing the screenshots. 
If the workflow was completed successfully, then set `was_completed` to `true`.
Also, provide the visual questions and their answers as part of the response.

Provide your answer as a JSON dictionary with the following format:
{{
    "visual_questions": <list of visual questions and their answers>,
    "rationale": <rationale>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""

prompt_validate_with_text_intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of actions.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application. After each step, a description of the action is given.

# Workflow Demonstration

You are given the following sequence of actions which summarizes each part of the demonstration of the workflow. 
The actions are presented in chronological order.

Here are the actions from the workflow:"""

prompt_validate_with_text_close: str = lambda : f"""
# Instructions

Given what you observe in the previous sequence of action descriptions, was the workflow successfully completed? 
To determine this, derive few questions from the task description that upon answering will help decide if the workflow was successfully completed.
If the workflow is asking a question, consider it completed successfully if you could deduce the answer to the question by viewing the sequence of actions. 
If the workflow was completed successfully, then set `was_completed` to `true`. Please only set `was_completed` to false if you are absolutely certain the solution provided for the task is wrong. Some workflows will only contain the answer in a small portion of the workflow or in the final answer only. Keep this in mind.
Also, provide the questions and their answers as part of the response.

Provide your answer as a JSON dictionary with the following format:
{{
    "reasoning_questions": <list of questions and their answers>,
    "rationale": <rationale>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""

prompt_validate_with_text_vision_intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, based off the evaluations of a text evaluator and vision evaluator. The text evaluator is exclusively given text information about the workflow. The vision evaluator is exclusively given screenshots throughout the workflow. You are given the outcome based of each of these evaluator. Note there are certain situations where one evaluators works better than the other:

Text Evaluator Advantages: The text evaluator includes a summary of every action of the workflow with no missing parts. This means every step of the way is captured with no gaps.
Text Evaluator  Disadvantages: The text evaluator only summarizes each step based off the site DOM. These summaries are sometimes inaccurate as some actions cannot be verified by the DOM. Thus, occasionally the text evaluator assumes a task was done correctly when in fact it was not. 
Vision Evaluator Advantages: The vision evaluator has an accurate depiction of the website with a single screenshot. Unlike the text evaluator, it is always an accurate depiction of the current state of the website.
Vision Evaluator Disadvantages: The vision evaluator frequently is provided with little to no screenshots, thus in many cases it does not have enought information to evaluate the workflow.


# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application. The text evaluator and vision evaluator give their assesment of wheather the task was successfully completed as well as their reasoning.
"""

prompt_validate_with_text_vision_close: str = lambda : f"""
# Instructions

Given what you the assesements given by the text and vision evaluator, was the workflow successfully completed? 
To determine this, derive few questions from the task description that upon answering will help decide if the workflow was successfully completed.
If the workflow was completed successfully, then set `was_completed` to `true` otherwise set to `false`. In cases where many screenshots are missing and it cannot be confirmed if a task was done successfully, rely on the text evaluator. Rely on the visual evaluator when it can absolutely confirm the text evaluator is wrong. With the information given, please determine if the the intial task was completed by the workflow.


Provide your answer as a JSON dictionary with the following format:
{{
    "reasoning_questions": <list of questions and their answers>,
    "rationale": <rationale>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""

prompt_validate_with_vision_final_response_intro: str = lambda : f""""""  # TODO

prompt_validate_with_vision_final_response_close: str = lambda : f"""""" # TODO