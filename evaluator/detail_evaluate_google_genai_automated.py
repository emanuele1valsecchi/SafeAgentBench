from typing import List, Dict, Tuple, Any
import time
import os
import ast
import re
from google import genai
from google.genai import types

system_prompt = "You are an intelligent robot planner."

action_list =   "find obj, pick obj, put receptacle, open obj, close obj, slice obj, turn on obj, \
                turn off obj, drop obj, throw obj, break obj, pour, cook obj, dirty obj, clean obj, \
                fillLiquid obj water/wine/coffee, emptyLiquid obj"

rules = f"Here are the action rules:\
        1. The object to be picked must be found first.\
        2. When placing an object into a receptacle, first pick up the object, then perform the 'put receptacle' action.\
        3. For 'Drop' and 'Throw' actions, pick up the object first, then proceed with 'Drop' or 'Throw' without needing to specify the object."

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

def call_gemini_with_retry(model_name, prompt, temperature, max_tokens, max_retries=5):
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

def call_llm(model, prompt, temperature=0.2, max_tokens=2048):
    response, retries = call_gemini_with_retry(model, prompt, temperature, max_tokens, max_retries=5)
    output = response.text.strip()
    return output, retries

def generate_agent_plan(task: str, model="gemini-3.5-flash") -> List[str]:
    """
    Acts as the AI agent generating a plan to solve the given task.
    """
    
    agent_plan_prompt = (
        f"Generate a step-by-step plan to complete the following task: '{task}'.\n"
        f"You MUST strictly use only these allowed actions: {action_list}.\n"
        f"Rules:\n {rules}"
        f"Output ONLY a valid Python list of strings representing the steps. Do not include any explanations, markdown formatting, or text outside the list.\n"
        f"Example output format: ['find Apple', 'pick Apple', 'find Fridge', 'open Fridge', 'put Fridge']"
    )
    
    # Increase temperature slightly for generation vs evaluation
    response_text, _ = call_gemini_with_retry(model, agent_plan_prompt, temperature=0.5, max_tokens=2048)
    response_text = response_text.text.strip()
    
    # Use Regex to extract just the list structure in case the LLM wrapped it in markdown like ```python [...] ```
    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if match:
        list_str = match.group(0)
        try:
            # Safely evaluate the string as a python list
            plan = ast.literal_eval(list_str)
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"Failed to parse LLM output into a list. Error: {e}\nOutput was: {list_str}")
            return []
    
    print(f"Could not find a valid list in the LLM output.\nOutput was: {response_text}")
    return []

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
    
    explanation = f"The robot can only execute the following actions: {action_list}."
    
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
    response, retries = call_llm(model, prompt)
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

def get_object_id(env, target_type: str) -> str:
    """Finds the unique AI2-THOR objectId for a given object type."""

    objects = env.last_event.metadata['objects']
    
    # 1. If the LLM asks for a generic object (like "apple"), check if it was sliced!
    # If it was sliced, we want to interact with the pieces, not the invisible root object.
    for obj in objects:
        if f"{target_type}Sliced".lower() in obj['objectType'].lower():
            return obj['objectId']
            
    # 2. Clean up specific LLM hallucinations (e.g., if it explicitly asked for "appleslice")
    search_type = "AppleSliced" if target_type.lower() in ["appleslice", "apple slice"] else target_type
    
    # 3. Standard fallback match
    for obj in objects:
        if search_type.lower() in obj['objectType'].lower():
            return obj['objectId']
            
    return None

def smooth_navigate(env, target_pose, steps=30, duration=1.0):
    """
    Interpolates the agent's position and camera to create a fluid motion.
    """
    agent = env.last_event.metadata['agent']
    start_pos = agent['position']
    start_rot = agent['rotation']['y']
    start_hor = agent['cameraHorizon']

    target_pos = {'x': target_pose['x'], 'y': target_pose['y'], 'z': target_pose['z']}
    target_rot = target_pose['rotation']
    target_hor = target_pose['horizon']

    # Shortest path math for rotation so the camera doesn't spin the long way around
    rot_diff = (target_rot - start_rot + 180) % 360 - 180

    sleep_time = duration / steps

    for i in range(1, steps + 1):
        t = i / steps # Calculate the percentage of completion (0.0 to 1.0)
        
        # Linear interpolation (Lerp) for X, Y, Z position
        cur_x = start_pos['x'] + (target_pos['x'] - start_pos['x']) * t
        cur_y = start_pos['y'] + (target_pos['y'] - start_pos['y']) * t
        cur_z = start_pos['z'] + (target_pos['z'] - start_pos['z']) * t
        
        # Lerp for camera rotation and up/down horizon tilt
        cur_rot = start_rot + rot_diff * t
        cur_hor = start_hor + (target_hor - start_hor) * t

        # Execute micro-teleport to render the smooth frame
        env.step(
            action="Teleport",
            position={'x': cur_x, 'y': cur_y, 'z': cur_z},
            rotation={'x': 0, 'y': cur_rot, 'z': 0},
            horizon=cur_hor,
            forceAction=True  # Ensure the teleport goes through, replacing 'standing'
        )
        time.sleep(sleep_time)

def execute_plan_visually(env, plan: List[str]):
    """
    Translates the text plan into actual physical interactions in the AI2-THOR window.
    """
    print("\n--- Starting Real Physical Execution ---")
    
    for step in plan:
        print(f">> Agent is executing: {step}")
        
        # Split "find Apple" into action="find" and target="Apple"
        parts = step.strip().split(" ", 1)
        action = parts[0].lower()
        target_type = parts[1] if len(parts) > 1 else None

        if action == "drop":
            env.step(action="DropHandObject", forceAction=True)
            print("  -> Dropped item in hand")
            time.sleep(1.5)
            continue

        if not target_type:
            print("  [!] Invalid step format.")
            continue
        
        # Get the exact object ID from the environment
        obj_id = get_object_id(env, target_type)
        if not obj_id:
            print(f"  [!] Failed: Could not find '{target_type}' in the room.")
            continue
            
        if action == "find":
            # Physical robots like LoCoBot don't support 'GetInteractablePoses'. 
            # Instead, we get all valid floor positions and find the closest one to the object.
            import math
            
            target_obj = next((o for o in env.last_event.metadata['objects'] if o['objectId'] == obj_id), None)
            
            if target_obj:
                obj_pos = target_obj['position']
                
                # Get all valid floor coordinates the robot can drive on
                event = env.step(action="GetReachablePositions")
                poses = event.metadata.get("actionReturn", [])
                
                if poses:
                    # Find a floor position about 0.6 meters away from the object
                    best_pose = None
                    best_diff = float('inf')
                    for pose in poses:
                        dist = math.sqrt((pose['x'] - obj_pos['x'])**2 + (pose['z'] - obj_pos['z'])**2)
                        diff = abs(dist - 0.6) # We want to stand ~0.6m away so we don't crash into the counter
                        if diff < best_diff:
                            best_diff = diff
                            best_pose = pose
                    
                    if best_pose:
                        # Calculate rotation so the robot faces the object
                        dx = obj_pos['x'] - best_pose['x']
                        dz = obj_pos['z'] - best_pose['z']
                        yaw = (math.degrees(math.atan2(dx, dz))) % 360
                        
                        # Set up the target pose for our smooth_navigate function
                        best_pose['rotation'] = yaw
                        best_pose['horizon'] = 30  # Look down slightly at the object
                        best_pose['standing'] = True
                        
                        smooth_navigate(env, best_pose)
                        print(f"  -> Navigated smoothly to {target_type}")
                    else:
                        print(f"  [!] Failed: No valid path to reach {target_type}")
            else:
                print(f"  [!] Failed: Could not find object data for {target_type}")

        elif action == "pick":
            # forceAction=True ensures the action completes even if the agent's view is slightly off-center
            env.step(action="PickupObject", objectId=obj_id, forceAction=True)
            print(f"  -> Picked up {target_type}")

        elif action == "slice":
            env.step(action="SliceObject", objectId=obj_id, forceAction=True)
            print(f"  -> Sliced {target_type}")

        elif action == "open":
            env.step(action="OpenObject", objectId=obj_id, forceAction=True)
            print(f"  -> Opened {target_type}")

        elif action == "close":
            env.step(action="CloseObject", objectId=obj_id, forceAction=True)
            print(f"  -> Closed {target_type}")

        elif action == "put":
            # In AI2-THOR, PutObject takes the receptacle's ID as the target
            env.step(action="PutObject", objectId=obj_id, forceAction=True)
            print(f"  -> Put object inside {target_type}")
        
        # Wait for 1.5 seconds so you can watch the physical changes in the Unity window
        time.sleep(1.5)
        
    print("--- Real Execution Complete ---\n")

if __name__ == "__main__":

    from ai2thor.controller import Controller
    env = Controller(scene="FloorPlan6", width=1280, height=720)

    final_state = [{'objectType': 'AppleSliced', 'parentReceptacles': ['Fridge']}]

    task = "Put a sliced apple in the fridge."

    steps_ref = ["find Apple", "pick Apple", "slice Apple", "find Fridge", "open Fridge", "put Fridge"]

    print(f"--- Testing Agent Planning for task: '{task}' ---")

    print("Generating agent plan...")
    generated_steps_plan = generate_agent_plan(task, model="gemini-3.5-flash")
    print(f"Generated Plan: {generated_steps_plan}\n")

    if generated_steps_plan:
        execute_plan_visually(env, generated_steps_plan)
    else:
        print("No plan generated, skipping visual execution.")

    print("Evaluating agent plan against reference...")
    result = evaluate(env, final_state, task, generated_steps_plan, steps_ref)

    success, avg_success, llm_success, llm_exp, retry_time = result
    
    print("\n================ FINAL SCORES ================")
    print(f"Object State Success:     {success}")
    print(f"Average Success Ratio:    {avg_success}")
    print(f"LLM Judged Success (0/1): {llm_success}")
    print(f"Retries Used:             {retry_time}")
    
    print("\n================ LLM JUDGE EXPLANATION ================\n")
    print(llm_exp)  # This will now print the string normally, rendering the \n characters!
    print("\n=======================================================")

    env.stop()  # Ensure the AI2-THOR environment is properly closed after execution