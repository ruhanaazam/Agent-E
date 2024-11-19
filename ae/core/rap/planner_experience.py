from ae.core.rap.message_util import load_main_chat, format_main_chat, get_last_plan

class PlannerExperience:
    def __init__(self,):
        """
        Constructor to initialize width and height of the rectangle.
        """
        return 
    
    def main_chat_path(self, example_id) -> str: 
            #return f"/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs/logs_for_task_{example_id}/execution_logs_{example_id}.json"
            return f"/Users/ruhana/Agent-E/new_ruhana_notes/All/logs/test_results_for_full_text/logs_for_task_{example_id}/execution_logs_{example_id}.json"
        
    def result_path(self, example_id) -> str:
            #return f"/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results/results_for_task_{example_id}.json"
            return f"/Users/ruhana/Agent-E/new_ruhana_notes/All/results/results_for_test_full_text/results_for_task_{example_id}.json"

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
        return prompt