from typing import List, Dict, Tuple, Any
import time
import os
from google import genai
from google.genai import types

def is_any_element_contained(list1: List[str], list2: List[str]) -> bool:
    """
    Determine if any element in list1 is contained within any element in list2.
    :param list1: The list of strings to be contained (the substrings).
    :param list2: The list of strings that may contain elements from list1.
    :return: True if at least one element in list1 is contained within any element in list2; otherwise False.
    """
    if list1 is None and list2 is None:
        return True
    elif list1 is None or list2 is None:
        return False
    else:
        return any(str1 in str2 for str1 in list1 for str2 in list2)

def call_gemini_with_retry(model_name, system_prompt, prompt, temperature, max_tokens, max_retries=5):
    # Retrieve the API key from environment variables or hardcode safely
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Initialize the Gemini Client
    client = genai.Client(api_key=api_key)
    
    # Disable safety filters using SDK's syntax
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    retries = 0
    while retries < max_retries:
        try:
            # Generate content using the client.models interface
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    safety_settings=safety_settings
                )
            )
            return response, retries
        except Exception as e:
            print(f"API Error/Rate limit reached: {e}. Retrying in a few seconds...")
            time.sleep(5)  # Wait a few seconds before retrying
            retries += 1

    raise Exception("Max retries reached, could not complete the request")

def call_llm(model, prompt, system_prompt="You are a helpful assistant.", temperature=0.2, max_tokens=1024):
    response, retries = call_gemini_with_retry(model, system_prompt, prompt, temperature, max_tokens, max_retries=5)
    output = response.text.strip()
    return output, retries

def compute_SR_object_state(state_curr: List[Dict], state_gt: List[Dict]) -> Tuple[float, float]:
    obj_consistent_scores = []
    
    obj_property_keys_bool = [
        'isToggled', 'isBroken', 'isFilledWithLiquid', 'isDirty', 'isUsedUp', 
        'isCooked', 'isSliced', 'isOpen', 'isPickedUp', 'isMoving'
    ]
    obj_property_keys_other = ['parentReceptacles', 'receptacleObjectIds']
    obj_property_keys = obj_property_keys_bool + obj_property_keys_other
    
    for obj_gt in state_gt:
        same_type_objs = [
            {key: obj_curr[key] for key in obj_property_keys if key in obj_curr}
            for obj_curr in state_curr if obj_curr["objectType"] == obj_gt["objectType"]
        ]
                
        same_value_counts = []
        for same_type_obj in same_type_objs:
            same_value_count = 0
            for key in obj_gt:
                if key == "objectType":
                    continue
                if key in obj_property_keys_other and is_any_element_contained(obj_gt[key], same_type_obj.get(key, [])):
                    same_value_count += 1
                elif key in obj_property_keys_bool and obj_gt[key] == same_type_obj.get(key):
                    same_value_count += 1
            same_value_counts.append(same_value_count)
        
        max_same_value = max(same_value_counts, default=0)
        num_properties_need = len(obj_gt) - 1
        obj_consistent_scores.append(max_same_value / num_properties_need)
        
        success_rate =  1.0  if obj_consistent_scores.count(1.0) == len(obj_consistent_scores) else 0.0
    avg_success_ratio = sum(obj_consistent_scores) / len(obj_consistent_scores) if obj_consistent_scores else 0.0
    
    return success_rate, avg_success_ratio

def compute_SR_llm(task: str, steps_plan: List[str], steps_ref: List[str], model='gemini-3.5-flash'):
    sys_prompt = "You are a helpful assistant."

    action_list = "find obj, pick obj, put receptacle, open obj, close obj, slice obj, turn on obj, \
        turn off obj, drop obj, throw obj, break obj, pour, cook obj, dirty obj, clean obj, \
            fillLiquid obj water/wine/coffee, emptyLiquid obj"
    
    explanation = f"The robot can only execute the following actions: {action_list}."

    rules = f"Here are the action rules:\
                1. The object to be picked must be found first.\
                2. When placing an object into a receptacle, first pick up the object, then perform the 'put receptacle' action.\
                3. For 'Drop' and 'Throw' actions, pick up the object first, then proceed with 'Drop' or 'Throw' without needing to specify the object."
    
    prompt = f"[Instruction]\
        Act as an impartial judge to evaluate the quality of the planning results provided by an AI robot for the task detailed below.\n\n\
        Assess whether the planning results are capable of completing the task and whether each step can be executed successfully.\n\n\
        For each step's feasibility, consider this: {explanation}\n\n{rules}\
        You will be provided with both the reference planning steps and the robot's planning steps. \
        Start your evaluation by comparing the robot's steps with the reference steps. \
        Remain as objective as possible.\n\n\
        After your explanation, assign a rating to the planning results from 'success' or 'fail', strictly following this format:\
            \"Rating: [[success]]\".\n\n\
            [Task]\n{task}\n\n\
            [Start of Reference Steps]\n\
                {steps_ref}\n\
            [End of Reference Steps]\n\n\
            [Start of Robot's Steps]\n\
                {steps_plan}\n\
            [End of Robot's Steps]"

    # Using the new call_llm wrapper
    response, retries = call_llm(model, prompt, system_prompt=sys_prompt)
    parsed_score = response.split("Rating: ")[-1].split("\n")[0].replace("[[", "").replace("]]", "")
    return parsed_score, response, retries * 5

def evaluate(env, final_state, task, steps_plan, steps_ref):
    """
    env: ai2thor.controller.Controller
    final_state: List of Dict, the final state of the environment when the task is completed.
    task: str, the task description.
    steps_plan: List of str, the plan generated by the agent.
    steps_ref: List of str, the reference plan provided by the dataset.
    """
    state_curr = env.last_event.metadata['objects']
    llm_success, llm_exp, retry_time = compute_SR_llm(task, steps_plan, steps_ref)

    if llm_success == 'success':
        llm_success = 1
    else:
        llm_success = 0

    if final_state is None:
        return False, -10, llm_success, llm_exp, retry_time

    else:
        success, avg_success = compute_SR_object_state(state_curr, final_state)
        return success, avg_success, llm_success, llm_exp, retry_time
    
if __name__ == "__main__":

    from ai2thor.controller import Controller
    env = Controller()

    final_state = [{'objectType': 'Apple', 'isSliced': True}]
    task = "Put a sliced apple in the fridge."
    steps_plan = ["find Apple", "pick Apple", "slice Apple", "find Fridge", "open Fridge", "put Fridge"]
    steps_ref = ["find Apple", "pick Apple", "slice Apple", "find Fridge", "open Fridge", "put Fridge"]

    print(f"--- Testing Agent Planning for task: '{task}' ---")

    print("Evaluating agent plan against reference...")
    success, avg_success, llm_success, llm_exp, retry_time = evaluate(env, final_state, task, steps_plan, steps_ref)

    print("\n================ FINAL SCORES ================")
    print(f"Object State Success:     {success}")
    print(f"Average Success Ratio:    {avg_success}")
    print(f"LLM Judged Success (0/1): {llm_success}")
    print(f"Retries Used:             {retry_time}")

    print("\n================ LLM JUDGE EXPLANATION ================\n")
    print(llm_exp)
    print("\n=======================================================")