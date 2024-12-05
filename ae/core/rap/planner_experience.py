from ae.core.rap.message_util import load_main_chat, format_main_chat, get_last_plan

class PlannerExperience:
    def __init__(self,):
        """
        Constructor to initialize width and height of the rectangle.
        """
        self.results_folder = "/Users/ruhana/Agent-E/new_ruhana_notes/All/results/results_for_test_full_text"
        self.log_folder = "/Users/ruhana/Agent-E/new_ruhana_notes/All/logs/test_results_for_full_text/"
        return 
    
    def main_chat_path(self, example_id) -> str: 
            return f"{self.log_folder}/logs_for_task_{example_id}/execution_logs_{example_id}.json"
        
    def result_path(self, example_id) -> str:
            return f"{self.results_folder}/results_for_task_{example_id}.json"

    def build_few_shot_prompt(self, strategy:str, example_id=None) -> str:
        """
        """
        prompt = ""
        if strategy == "initial_plan_example":
            main_chat_list = load_main_chat(self.main_chat_path(example_id))
            intial_plan = get_last_plan(main_chat_list)
            intent = main_chat_list[0].get("content", "")
            
            directions = "Use the example below to help accomplish the provided task."
            prompt = f'''{directions}
                Example 1:
                Task: {intent}
                Solution: {intial_plan}
            '''
        elif strategy == "full_chat_example":
            main_chat_list = load_main_chat(self.main_chat_path(example_id))
            formatted_main_chat = format_main_chat(main_chat_list)
            direction = "Please use this plan as a guideline"
            prompt = f"Please use this plan as a guideline:{formatted_main_chat}"
        elif strategy == "rag-self-reflection":
            reflections_folder = f"" # TODO: get the tip folder
            tips = "Try to be helpful" # TODO: get the tip
            direction = f"\n When planning consider the following tips:\n"
            prompt = direction + tips
        return prompt