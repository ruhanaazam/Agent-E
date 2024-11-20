import os
import json
import pandas as pd
from validation_agent.utils import get_screenshot_paths

class AnnotationLoader:
    def __init__(self, log_path, results_path):
        self.log_path = log_path
        self.results_path = results_path
        
        self.result_prefix = "test_result_"
        self.log_prefix = "logs_for_task_"
        return 

    def get_task_ids(self,):
        result_files = os.listdir(self.results_path)
        task_ids = [file.replace(self.result_prefix, "").replace(".json", "") for file in result_files if "json" in file]
        task_ids = [int(ids) for ids in task_ids]
        return sorted(task_ids)
    
    def load_json(self, file_path):
        """
        Load a JSON file and flatten it using pandas' json_normalize function.

        :param file_path: Path to the JSON file.
        :return: A flattened DataFrame.
        """
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return pd.json_normalize(json_data)
    
    def load_nested_chat(self, file_path):
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        message_json = json_data
        
        # first message is the subtask in the plan
        # last message is summarizes actions from the low-level browser
        task = message_json[0].get("content")
        final_response = message_json[-1].get("content")
        
        message_json=message_json[0:-1]

        
        # Load each message as state and action
        # intial state is the task, the rest are tool call responses
        # (subtask, action0), (response1, action1), (response2, action2).... (final_tool_response, None)
        trajectory = []
        for i in range(0, len(message_json), 2):
            if i == ((len(message_json))//2)*2: #last loop
                state = message_json[i]["content"]
                action = None
            else:
                state = message_json[i]["content"]
                action = message_json[i+1]["tool_calls"][0]["function"]["name"] #this may need to be more precise
            trajectory.append((state, action))
        return trajectory
    
    def get_result(self, task_id):
        # Load results file(s)
        result_path = os.path.join(self.results_path, f"{self.result_prefix}{task_id}.json")
        result_df = self.load_json(result_path)
        return result_df
    
    def get_intent(self, task_id):
        return self.get_result(task_id)['intent']
    
    def get_high_level_trajectory(self, task_id):
        '''
        Load a single trajectory
        '''        
        # Load high-level planner chat
        chat_path = os.path.join(self.log_path, f"{self.log_prefix}{task_id}/execution_logs_{task_id}.json")
        chat_df = self.load_main_chat(chat_path)
        return chat_df 
    
    def get_low_level_trajectory(self, task_id):
        '''
        Load a single trajectory
        '''
        # Load results file(s)
        result_path = os.path.join(self.results_path, f"{self.result_prefix}{task_id}.json")
        result_df = self.load_json(result_path)
        task = result_df['intent']
        score = result_df['score']
        
        # Load low-level planner chat
        nested_file_base_path = os.path.join(self.log_path, f"{self.log_prefix}{task_id}")
        nested_files = os.listdir(nested_file_base_path)
        
        trajectories = []
        for file in nested_files:
            if "nested_chat_log" in file:
                full_file = os.path.join(nested_file_base_path, file)
                trajectory = self.load_nested_chat(full_file)
                trajectories = trajectories + trajectory
        return trajectories
    
    def get_screenshot_paths(self, task_id):
        screenshot_directory = os.path.join(self.log_path, f"{self.log_prefix}{task_id}/snapshots")
        screenshot_path_list = get_screenshot_paths(screenshot_directory)
        return screenshot_path_list     
    
    def get_train_test_split(self, split=0.8):
        # TODO: unfinished function
        task_ids = self.get_task_ids()
        
        for task_id in task_ids:
            low_level = self.get_low_level_trajectory(task_id)
            high_level = self.get_high_level_trajectory(task_id)
        return 
   
    def get_all_annotation_results(self,):
        task_ids = self.get_task_ids()
        
        result_df_list = []
        for task_id in task_ids:
            try:
                result_path = os.path.join(self.results_path, f"{self.result_prefix}{task_id}.json")
                result_df = self.load_json(result_path)
                
                # Add token count from main_chat
                high_level_chat = self.get_high_level_trajectory(task_id)
                high_level_chat = self.format_main_chat(high_level_chat)
                total_generated_tokens = sum(len(string.split()) for string in high_level_chat[1:]) - len(high_level_chat[1:]) 
                result_df["total_main_chat_token"] = total_generated_tokens
                result_df["main_chat_length"] = len(high_level_chat)

                # Add trajectory length from fine-grained actions
                low_level_trajectory = self.get_low_level_trajectory(task_id)
                total_trajectory_length = len(low_level_trajectory)
                result_df["total_trajectory_length"] = total_trajectory_length
                result_df_list.append(result_df)
            except Exception as err: 
                print(err)
                continue
        all_results = pd.concat(result_df_list)
        return all_results
        
    def load_main_chat(self, file_path: str):
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        message_json = list(json_data.values())[0]
        return message_json
    
    def format_main_chat(self, message_json):
        # Load each message
        fomatted_messages = []
        for message in message_json:
            role = message.get("role", "")
            content = message.get("content", "")
            fomatted_messages.append(f"{role}: {content}")
            # TODO: might be better to remove the json formatting message, migth be more intuitive
        return fomatted_messages
        
        
    def load_main_chat_seqence(self, task_id: int):
        # Load the main chat as json
        chat_path = os.path.join(self.log_path, f"{self.log_prefix}{task_id}/execution_logs_{task_id}.json")
        with open(chat_path, 'r') as f:
            json_data = json.load(f)
        message_json = list(json_data.values())[0]
        
        # # Load each message
        # fomatted_messages = []
        # for message in message_json:
        #     role = message.get("role", "")
        #     content = message.get("content", "")
        #     fomatted_messages.append(f"{role}: {content}")
        #     # TODO: might be better to remove the json formatting message, migth be more intuitive
        return message_json

class AnnotationLoaderForEmbedding():
    def __init__(self, log_path, results_path):
        self.log_path = log_path
        self.results_path = results_path
        
        self.result_prefix = "test_result_"
        self.log_prefix = "logs_for_task_"
        return 
    
    def get_task_ids(self,):
        result_files = os.listdir(self.results_path)
        task_ids = [file.replace(self.result_prefix, "").replace(".json", "") for file in result_files if "json" in file]
        task_ids = [int(ids) for ids in task_ids]
        return sorted(task_ids)
    
    def load_json(self, file_path):
        """
        Load a JSON file and flatten it using pandas' json_normalize function.

        :param file_path: Path to the JSON file.
        :return: A flattened DataFrame.
        """
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return pd.json_normalize(json_data)
    
    def get_high_level_trajectory(self, task_id):
        # Load results file(s)
        result_path = os.path.join(self.results_path, f"{self.result_prefix}{task_id}.json")
        result_df = self.load_json(result_path)
        score = result_df['score'][0]
        
        # Load high-level planner chat
        chat_path = os.path.join(self.log_path, f"{self.log_prefix}{task_id}/execution_logs_{task_id}.json")
        with open(chat_path, 'r') as f:
            json_data = json.load(f)
        message_json = list(json_data.values())[0]
        
        # fill dataframe for the task
        df = pd.DataFrame(columns=['task', 'step', 'output', 'history'])
        time_step = 0
        task = message_json[0].get("content", "")
        for i in range(0, len(message_json), 2):
            user_message = message_json[i]
            planner_message = message_json[i+1].get('content')
            history = str(message_json[0:i]).replace('"', '""')
            
            #add to dataframe here
            new_row = pd.DataFrame([{
                'task': task,
                'step': time_step,
                'output': planner_message,
                "history": history,
                'reward': score
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            time_step +=1
            
        return df
    
    def build_csv(self, file_name):
        task_ids = self.get_task_ids()
        
        df_list = []
        for task_id in task_ids:
            df = self.get_high_level_trajectory(task_id)
            df_list.append(df)
        full_df = pd.concat(df_list, ignore_index=True)
        full_df.to_csv(file_name, index=False)
        return 
    
def main():
    LOG_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/logs"
    RESULT_PATH = "/Users/ruhana/Agent-E/ruhana_notes/baseline_annotated/original_annotations/All/results"
    
    annon = AnnotationLoaderForEmbedding(LOG_PATH, RESULT_PATH)
    csv_name = "agent_e_webvoyager_rollouts.csv"
    print(annon.build_csv(csv_name))
    # double check the log and result exits 
    
    # now we look for specific example task_id runs, we want to improve on!, and double check that issues persists,
    # the goal here is to find trajectories that are long and successfull
    
main()
    
    