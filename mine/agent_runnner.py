
import json
import ai2_thor_functionalities as ai2thor_func
import ai_command as ai_cmd
import utils as u
import rye

scenes = {}

def load_pre_defined_setup() -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Load predefined scene and task from 'kitchen_tasks_and_constraints.jsonl'"""

    try:
        with open("./dataset/kitchen_tasks_and_constraints.json", "r") as f:
            pre_defined_list = json.load(f)
    except:
        print("Error in loading pre defined tasks. Please ensure the file exists in the specified path")

    print("Available pre defined use case:")

    for i, use_case in enumerate(pre_defined_list):
        print(f"{i+1}) Scene: {use_case['scene_name']}\
              \n   Instruction: {use_case['instruction']}\
              \n   Requirement: {use_case['requirement']}\
              \n   Reference steps: {use_case['reference_steps']}\
              \n   Reelay Expression: {use_case['reelay_expression']}\n")

    u.print_separator()

    scenes.clear()

    chosen_scene = None
    chosen_instruction = None
    chosen_requirement = None
    chosen_reference_steps = None
    chosen_reelay_expression = None

    if u.yn_question(f"Do you want to load a pre defined use case?"):
        use_case = u.req_not_empty_value("Inser the use case number to load: ").strip()

        try:
            use_case = int(use_case) - 1

            if use_case > len(pre_defined_list) or use_case < 0:
                raise Exception
    
            chosen_case = pre_defined_list[use_case]
    
            chosen_scene = chosen_case['scene_name']
            chosen_instruction = chosen_case['instruction']
            chosen_requirement = chosen_case['requirement']
            chosen_reference_steps = chosen_case['reference_steps']
            chosen_reelay_expression = chosen_case['reelay_expression']

            print(f"\nLoading use case: {(use_case + 1)}\
                  \n  Scene: {chosen_scene}\
                  \n  Instruction: {chosen_instruction}\
                  \n  Requirement: {chosen_requirement}\
                  \n  Reference steps: {chosen_reference_steps}\
                  \n  Reelay Expression: {chosen_reelay_expression}")

            u.print_separator()

            print("Start simulation")

            u.print_separator()
        except:
            print("The input is not valid, no pre defined use case will be loaded")
    else:
        u.wait_ui(end_message = "Press enter to manually configure the use case")

        u.print_separator()

    return chosen_scene, chosen_instruction, chosen_requirement, chosen_reference_steps, chosen_reelay_expression

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
# Set to True to execute 
user_controlled = True

chosen_scene = "FloorPlan1"
chosen_reelay_expression = ""

if user_controlled:
    chosen_scene, chosen_instruction, chosen_requirement, chosen_reference_steps, chosen_reelay_expression = load_pre_defined_setup()

    if not chosen_scene:
        load_available_scenes()

        display_available_scenes()

        chosen_scene = choose_scene()
    else:
        user_controlled = False
else:
    chosen_instruction = "put a mug in the coffee machine"
    chosen_requirement = "the coffee machine should be turned on when the mug is putted inside it and then turned off"
    chosen_reference_steps = "find potato, pick potato, find floor, drop"

controller = ai2thor_func.create_controller(scene=chosen_scene, width = 1280, height = 720)

objs = scan_ambient(controller, fake = user_controlled)

if user_controlled:
    display_objects_in_scene()

#instruction, requirement and steps_ref can be deleted to execute the slice apple and put fridge routine
task, steps_ref = define_task(
    instruction = chosen_instruction,
    requirement = chosen_requirement,
    steps_ref = chosen_reference_steps,
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
        u.print_separator()
        print("\n Recreating the plan\n")
        u.print_separator()

u.print_separator()

if chosen_reelay_expression:
    u.wait_ui("Simulation complete.", "Press Enter to execute the rye analysis")

    rye_manager = rye.RyeManager()

    rye_manager.analysis(chosen_reelay_expression)

    u.print_separator()

else:
    u.wait_ui("Simulation complete.", "Press Enter to close the program")

u.wait_ui( end_message = "Press Enter to exit the program")

controller.stop()