#https://ai2thor.allenai.org/

from ai2thor.controller import Controller
import time

# Return a controller object with the specified parameters
def create_controller(agentMode = "default", visibilityDistance = 100, scene = "FloorPlan1", 
                      gridSize = 0.1, snapToGrid = False, rotationStepDegrees = 1,
                      renderDepthImage = True, renderInstanceSegmentation = True, 
                      width = 1280, height = 720, fieldOfView = 90):
    return Controller(
        agentMode=agentMode,
        visibilityDistance=visibilityDistance,
        scene=scene,

        gridSize=gridSize,
        snapToGrid=snapToGrid,
        rotationStepDegrees=rotationStepDegrees,

        renderDepthImage=renderDepthImage,
        renderInstanceSegmentation=renderInstanceSegmentation,

        width=width,
        height=height,
        fieldOfView=fieldOfView
    )

# Get the agent's reachable position in the scene.
def get_agent_reachable_position(controller: Controller):
    return controller.step(action="GetReachablePositions").metadata["actionReturn"]

def display_agent_reachable_position(controller: Controller):
    print("Agent's reachable positions:")
    for pos in get_agent_reachable_position(controller):
        print(f"{pos}")

def get_objects_in_scene(controller: Controller, **kwargs):
    """Get objects in scene, if **kwargs is passed only the one respecting 'key:value' are returned.\n 
    If no object respects the filter 'key:value' an emtpy list is returned """

    if not kwargs:
        return controller.last_event.metadata["objects"]

    fobjs = []

    for obj in controller.last_event.metadata["objects"]:
        if all(obj.get(k) == v for k, v in kwargs.items()):
            fobjs.append(obj)

    return fobjs

def find_object(controller: Controller, object_name: str):
    """Return the object with object_name reference in the scene if found, otherwise None"""

    if not object_name:
        return None
    
    objs = get_objects_in_scene(controller)

    for obj in objs:
        if obj['objectType'].lower() == object_name.lower():
            return obj

    return None

def display_objects_in_scene(controller: Controller, *args, **kwargs):
    """Display objects in the scene.\n
    Optionally specified the object characteristic to show in args\n
    Optionally filtered by **kwargs"""

    for obj in get_objects_in_scene(controller, **kwargs):
        if args:
            for objk, objd in obj.items():
                if objk in args:
                    print(f"{objk}: {objd}")
        else:
            for objk, objd in obj.items():
                print(f"{objk}: {objd}")
        print()

def display_visible_objects_in_scene(controller: Controller, *args):
    """Display only the visible objects in the scene, optionally specify the object characteristic to show"""
    display_objects_in_scene(controller, *args, visible=True)

def rotate_agent_smoothly(controller: Controller, direction, total_degrees=90, step = 10, sleep_between_steps=0.0001):
    """Rotate the agent smoothly by stepping through smaller rotation increments.\n
    - direction: 'left' or 'right'\n
    - total_degrees: how many degrees to rotate in total\n
    - step: degrees per step\n
    - sleep_between_steps: seconds to wait between steps for visible smoothness"""

    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")

    action = "RotateLeft" if direction == "left" else "RotateRight"
    remaining = abs(total_degrees)

    while remaining > 0:
        controller.step( action=action, degrees=step )

        if not controller.last_event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(f"Rotation failed: {controller.last_event.metadata.get('errorMessage')}")
        remaining -= step

        if remaining > 0:
            time.sleep(sleep_between_steps)

def rotate_agent_left_smoothly(controller: Controller, total_degrees=90, step=10, sleep_between_steps=0.0001):
    rotate_agent_smoothly(controller, "left", total_degrees, step, sleep_between_steps)

def rotate_agent_right_smoothly(controller: Controller, total_degrees=90, step=10, sleep_between_steps=0.0001):
    rotate_agent_smoothly(controller, "right", total_degrees, step, sleep_between_steps)

def display_visible_objects_around(controller: Controller, *args):
    """Display all the visible objects around the agent in the scene, optionally specify the object characteristic to show"""
    for i in range(4):
        display_visible_objects_in_scene(controller, *args)
        rotate_agent_left_smoothly(controller)

def execute_plan_visually(controller: Controller, plan: list[str]):
    for step in plan:
        parts = step.strip().split(" ", 1)

        action = parts[0].lower().strip()
        target = parts[1].lower().strip() if len(parts) > 1 else None

        obj = find_object(controller, target)
