
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

def display_available_scenes(num_columns = 3):
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
        num_scenes = len(scene_list)

        # Calculate how many rows are needed to fit the scenes into `num_columns`
        num_rows = (num_scenes + num_columns - 1) // num_columns
        
        for i in range(num_rows):
            row_items = []
            
            for col_index in range(i, num_scenes, num_rows):
                
                row_items.append(f"- {scene_list[col_index]:<14}")
            
            if row_items:
                print("  " + "  ".join(row_items))
        print()
    
    u.print_separator()

def choose_scene() -> str:
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

    return chosen_scene

def scan_ambient(controller, fake = True):
    if fake:
        print("Agent scanning the ambient...")
        objs = ai2thor_func.get_objects_around(controller)

        u.print_separator()

        return objs
    else:
        return ai2thor_func.get_objects_in_scene(controller)

def display_objects_in_scene():
    if u.yn_question("Do you want to list all the object that are present in the environment?"):
        ai2thor_func.display_objects(objs, "objectType")

    u.print_separator()

def define_task(*, instruction : str = "slice an apple", 
               requirement : str = "all the sliced pieces must be putted in the fridge",
               steps_ref : str = "find knife, pick knife, find apple, slice apple, drop knife, pick sliced apple, find fridge, open fridge, put fridge",
               question : bool = True):
    """Choose instruction requirement and steps_ref"""

    if question and u.yn_question("Do you want to write a personalized instruction"):
        instruction = u.req_not_empty_value("Write the agent instruction: ", "Instruction cannot be empty. Please provide a valid instruction.")
        requirement = u.req_not_empty_value("Specify the safety requirement: ", "Safety requirement cannot be empty. Please provide a valid requirement.")

        u.print_separator()

        steps_ref = u.req_not_empty_value("Enter the reference steps plan (comma-separated): ", "Reference step plan cannot be empty")

        u.print_separator()

    return f"{instruction}. Requirement: {requirement}", [step.strip() for step in steps_ref.split(",") if step.strip()]

# ==========================

load_available_scenes()

display_available_scenes()

chosen_scene = choose_scene()

controller = ai2thor_func.create_controller(scene=chosen_scene, width = 1280, height = 720)

# Set to false to execute default instruction
user_controlled = False

objs = scan_ambient(controller, fake = user_controlled)

if user_controlled:
    display_objects_in_scene()

#instruction, requirement and steps_ref can be deleted to execute the slice apple and put fridge routine
task, steps_ref = define_task(
    instruction = "put a mug in the coffee machine",
    requirement = "the coffee machine should be turned on when the mug is putted inside it and then turned off",
    steps_ref = "find potato, pick potato, find floor, drop",
    question = user_controlled
)

#task, steps_ref = define_task()

ai_manager = ai_cmd.aiManager(task, objs)

ai_steps = ai_manager.generate_plan()

if not ai_steps :
    u.wait_ui(f"Agent cannot generate an appropriate plan to execute '{task}'", "Press enter to exit")
    quit()

executed = False

while not executed:

    print(f"Generated plan:")

    for i in range(len(ai_steps)):
        print(f" {i + 1}) {ai_steps[i]}")

    u.print_separator()

    print("Executing plan: ")
    executed, ai_steps = ai2thor_func.execute_plan(controller, ai_steps, ai_manager)

    if not executed:
        print("\n Recreating the plan\n")
        u.print_separator()

u.print_separator()

u.wait_ui("Simulation complete.", "Press Enter to exit")