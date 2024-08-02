prompt__validate_action: str = lambda task_action: f"""# Task
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

prompt__validate_with_vision__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of screenshots.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Here are the screenshots of the workflow:"""

prompt__validate_with_vision__close: str = lambda : f"""
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

prompt__validate_with_vision_final_response__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of screenshots and text response.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration. 
The screenshots are presented in chronological order. After the screenshots, you are give a text response.

Here are the screenshots of the workflow:"""

prompt__validate_with_vision_final_response__close: str = lambda : f"""
# Instructions

Given what you observe in the previous sequence of screenshots followed by a text response from an actor, was the workflow successfully completed? 
To determine this, derive few visual questions from the task description that upon answering will help decide if the workflow was successfully completed.
If the workflow is asking a question, consider it completed successfully if you could deduce the answer to the question by viewing the screenshots and find the answer in the final response. 
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

prompt__validate_with_text__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, as depicted by the following sequence of actions.

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application. After each step, a description of the action is given.

# Workflow Demonstration

You are given the following sequence of actions peformed on a browser, which summarizes each part of the demonstration of the workflow. 
The actions are presented in chronological order.

Here are the actions from the workflow:"""

prompt__validate_with_text__close: str = lambda : f"""
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

prompt__validate_with_text_vision__intro: str = lambda task_descrip: f"""# Task
Your job is to decide whether the workflow was successfully completed, based off the text and vision observations. Below are some considerations.

# Considerations

1. Text Evaluator Advantages: The text evaluator includes a summary of every action of the workflow with no missing parts. This means every step of the way is captured with no gaps.
2. Text Evaluator  Disadvantages: The text observations only summarizes each step based off the site DOM. Depending on the website, the text observations may be inaccurate. This is highly dependent on the website the task that is being performed. For complex websites, with many filter, widgets and other highly interactive features, the text observations are frequently wrong. In such cases, it is advised to rely on the vision observations. 
3. Vision Evaluator Advantages: The vision evaluator has an accurate depiction of the website with a single screenshot. Unlike the text evaluator, it is always an accurate depiction of the current state of the website.
4. Vision Evaluator Disadvantages: The vision evaluator frequently is provided with little to no screenshots, thus in many cases it does not have enought information to evaluate the workflow.
5. You are advised to use both vision and text information to evaluate the task. 
6. It is advised that you cross check the text observations with the vision to make sure they are accurate. 
7. If there are too many missing screenshots, please rely on the text evaluator. 

# Workflow

The workflow is: "{task_descrip}"

# User Interface

The workflow was executed within the web application. Below is the sequence of text observations and vision observations:
"""

prompt__validate_with_text_vision__close: str = lambda : f"""
# Instructions

Given what you the assesemented by the text and vision observations, was the workflow successfully completed? 
To determine this, derive few questions from the task description that upon answering will help decide if the workflow was successfully completed.
If the workflow was completed successfully, then set `was_completed` to `true` otherwise set to `false`. Remember you can use both types observations to make your decision. Additionally, remember each type of observations can be cross checked with one another (i.e. if the text evaluator claims to type into the text box but the screenshots show that the wrong text is typed into the text box, you are able to tell the text observations are inaccurate.)


Provide your answer as a JSON dictionary with the following format:
{{
    "reasoning_questions": <list of questions and their answers>,
    "rationale": <rationale>,
    "was_completed": <true/false>
}}

Please write your JSON below:
"""


prompt__classifier__intro: str = """
# Task
You are a classification model which determine the best method for evaluating a task which was performed on a web browser. The task can be evaluated in two ways 1) using text only or 2) using vision only. You are given start_url, task, a set of text observations and a set of visual observations (i.e. screenshots). Given these pieces of information, which method of evaluation is the most likely to be accurate?

# Considerations
There are certain situations where one evaluators works better than the other:

1. The text observations only summarizes each step based off the site DOM. Depending on the website, these summaries are sometimes inaccurate. This is highly dependent on the website the task that is being performed. For complex websites, with many filter, widgets and other highly interactive features, the text observations are frequently wrong. In such cases, it is advised to used the vision observations. For simple websites which contain static text and are less interactive, text observations will work well.
2. The vision observation contains a sequence of screenshots throuhgout the workflow. Although, much of the time screenshots are missing throughout the workflow. This can make it difficult to accurately evaluate the workflow. Try your best to determine if many screenshots are missing screenshots. In cases where many screenshots are missing, we recommend using the text evaluator. 
3. To check if the sequence of screenshots are reliable, you can try to cross check the visual observations are consistent with the text observations. 
4. Note although the text observations can seem detailed, they can have misleading information! It is recommended if a website has many filters and widgets to use the vision obvervations. 

# Example
To get an idea of in which cases text observations are better and which cases visual observations are better, here are some examples:
"""

prompt__classifier__close: str = """
# Instructions

Given the start_url, task, text observations and vision observations, which set of observations are more informative for evaluating the task? 
If the text observations are better to perform evaluation, then set `mode` to `text` otherwise set to `vision`.

Provide your answer as a JSON dictionary with the following format:
{{
    "reasoning_questions": <list of questions and their answers>,
    "rationale": <rationale>,
    "mode": <text/vision>
}}

Please write your JSON below:
"""

# RUHANA U DID NOT INCLUDE THE START URL!!!!!!

# write out some specific examples of good and bad...
# one where the text example has sufficient information

# one where there are missing screenshots

# one where the website dom is difficult to understand.

# you can see in the image that there are filters being used, these are typically difficult. These websites typically have inaccurate information 