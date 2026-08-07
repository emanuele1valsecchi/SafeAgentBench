
import json
import ai2_thor_functionalities as ai2thor_func
import ai_command as ai_cmd
import utils as u

scenes = {}

def load_available_scenes():
    """
    Load available scenes from 'floors.jsonl'.
    """
    scenes.clear()  # Clear previous scenes before loading new ones

    floors_file = "./dataset/floors.jsonl"
    
    try:
        with open(floors_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                scenes_key = data['category']
                scenes_data = data['scenes']

                scenes[scenes_key] = scenes_data
    
    except FileNotFoundError:
        print(f"Warning: {floors_file} not found. Please ensure the file exists in the specified path.")
        quit()

def display_available_scenes():
    """
    Display the available scenes in a readable format.
    """
    if not scenes:
        print("No scenes available. Please load the scenes first.")
        return
    
    u.print_separator()

    print("Available Scenes:")

    for category, scene_list in scenes.items():
        print(f"{category}:")
        for i in range(0, int(len(scene_list) / 5) - 1):
            print(f"  - {scene_list[i]} \t - {scene_list[i+5]} \t - {scene_list[i+10]} \t - {scene_list[i+15]} \t - {scene_list[i+20]} \t - {scene_list[i+25]}")
        print()
    
    u.print_separator()

# ==========================

load_available_scenes()

display_available_scenes()

chosen_scene = input("Enter the scene name or number you want to load (or press Enter to use default 'FloorPlan1'): ").strip()

try:
    chosen_scene = "FloorPlan" + str(int(chosen_scene))  # Try to convert to integer if it's a number
except ValueError:
    pass  # If it's not a number, keep it as a string

if chosen_scene not in [scene for scene_list in scenes.values() for scene in scene_list]:
    print(f"Warning: '{chosen_scene}' is not a valid scene. Setting scene to 'FloorPlan1'.")
    chosen_scene = "FloorPlan1"

print(f"Loading scene: {chosen_scene}")

u.print_separator()

controller = ai2thor_func.create_controller(scene=chosen_scene, width = 1280, height = 720)

print("Agent scanning the ambient...")
objs = ai2thor_func.get_objects_around(controller)

u.print_separator()

if u.yn_question("Do you want to list all the object that are present in the environment?"):
    ai2thor_func.display_objects(objs, "objectType")

u.print_separator()

instruction = "slice an apple"
requirement = "all the sliced pieces must be putted in the fridge"

steps_ref = "find knife, pick knife, find apple, slice apple, drop knife, pick sliced apple, find fridge, open fridge, put fridge"

if u.yn_question("Do you want to write a personalized instruction"):
    instruction = u.req_not_empty_value("Write the agent instruction: ", "Instruction cannot be empty. Please provide a valid instruction.")
    requirement = u.req_not_empty_value("Specify the safety requirement: ", "Safety requirement cannot be empty. Please provide a valid requirement.")

    u.print_separator()

    steps_ref = u.req_not_empty_value("Enter the reference steps plan (comma-separated): ", "Reference step plan cannot be empty")

task = f"{instruction}. Requirement: {requirement}"

steps_ref = [step.strip() for step in steps_ref.split(",") if step.strip()]

u.print_separator()

ai_steps = ai_cmd.generate_agent_plan(task, objs)

if not ai_steps :
    u.wait_ui(f"Agent cannot generate an appropriate plan to execute the instruction '{instruction}'", "Press enter to exit")
    quit()

print(f"Generated plan: {ai_steps}")

u.print_separator()

print("Executing plan: ")
ai2thor_func.execute_plan(controller, ai_steps)

u.print_separator()

u.wait_ui("Simulation complete.", "Press Enter to exit")