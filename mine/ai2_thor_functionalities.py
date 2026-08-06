#https://ai2thor.allenai.org/

from ai2thor.controller import Controller
import ai2_thor_task as task
import time
import numpy as np
from scipy import spatial
import math
import networkx as nx

# === DEFAULT VALUES ===
SLEEP_BETWEEN_STEPS = 0.0001
CAMERA_HEIGHT_OFFSET = 0.675
TARGET_MAX_DISTANCE = 1.0

# === CREATION ===
def create_controller(agentMode = "default", visibilityDistance = 100, scene = "FloorPlan1", 
                      gridSize = 0.1, snapToGrid = False, rotationStepDegrees = 1,
                      renderDepthImage = True, renderInstanceSegmentation = True, 
                      width = 1280, height = 720, fieldOfView = 90):
    """Return a controller object with the specified parameters"""
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

# === POSITION ===
def get_agent_position(controller : Controller) -> dict:
    return controller.last_event.metadata['agent']['position']

def get_agent_reachable_positions(controller: Controller) -> list[dict]:
    """Get the agent's reachable position in the scene."""
    return controller.step(action="GetReachablePositions").metadata["actionReturn"]

def navigate_to(controller: Controller, target_position, steps = 60):
    """
    Interpolates the agent's position and camera to create a fluid motion.
    """

    agent = controller.last_event.metadata['agent']
    start_pos = agent['position']
    start_rot = agent['rotation']['y']
    start_hor = agent['cameraHorizon']

    target_rot = target_position['rotation']
    target_hor = target_position['horizon']

    # Shortest path math for rotation so the camera doesn't spin the long way around
    rot_diff = (target_rot - start_rot + 180) % 360 - 180

    for i in range(1, steps + 1):
        t = i / steps # Calculate the percentage of completion (0.0 to 1.0)
        
        # Linear interpolation (Lerp) for X, Y, Z position
        cur_x = start_pos['x'] + (target_position['x'] - start_pos['x']) * t
        cur_y = start_pos['y'] + (target_position['y'] - start_pos['y']) * t
        cur_z = start_pos['z'] + (target_position['z'] - start_pos['z']) * t
        
        # Lerp for camera rotation and up/down horizon tilt
        cur_rot = start_rot + rot_diff * t
        cur_hor = start_hor + (target_hor - start_hor) * t

        # Execute micro-teleport to render the smooth frame
        controller.step(
            action="Teleport",
            position={'x': cur_x, 'y': cur_y, 'z': cur_z},
            rotation={'x': 0, 'y': cur_rot, 'z': 0},
            horizon=cur_hor,
            forceAction=True  # Ensure the teleport goes through, replacing 'standing'
        )
        time.sleep(SLEEP_BETWEEN_STEPS)

# === AGENT MOVEMENT ===
def rotate_agent_smoothly(controller: Controller, direction, total_degrees=90, step = 10):
    """Rotate the agent smoothly by stepping through smaller rotation increments.

    Args:
        direction: 'left' or 'right'
        total_degrees: how many degrees to rotate in total
        step: degrees per step"""

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
            time.sleep(SLEEP_BETWEEN_STEPS)

def rotate_agent_left_smoothly(controller: Controller, total_degrees=90, step=10):
    rotate_agent_smoothly(controller, "left", total_degrees, step)

def rotate_agent_right_smoothly(controller: Controller, total_degrees=90, step=10):
    rotate_agent_smoothly(controller, "right", total_degrees, step)

def get_kdtree_reachable_positions(agent_reachable_positions : list[dict]) -> spatial._kdtree.KDTree:
    return spatial.KDTree(np.array([[p['x'], p['y'], p['z']] for p in agent_reachable_positions]))

def get_closest_reachable_position(agent_reachable_positions : list[dict], target_position : dict, nth : int = 1) -> dict:
    kdtree_reachable_positions = get_kdtree_reachable_positions(agent_reachable_positions)
    _, i = kdtree_reachable_positions.query([target_position['x'], target_position['y'], target_position['z']], k = nth + 1)
    return agent_reachable_positions[(i[nth - 1])]

def is_object_close(target : dict[str, str], target_max_dist = TARGET_MAX_DISTANCE) -> bool:
    return target['visible'] and target['distance'] < target_max_dist

def get_object_closest_position(controller: Controller, target : dict[str, str], target_max_dist=TARGET_MAX_DISTANCE) -> tuple[dict, float, float] | None:
    """
    Based on a target provided evaluates the closesest position and camera rotation near the object

    Args:
        controller: Ai2THOR controller
        target: the target object obtained by the controller metadata
        target_max_dist: the maximum distance that the agent has to have to the object

    Returns:
        tuple[dict, float, float]: Containing the closest position to the object, the rotation and horizon that the agent has to have to be close to the object and look at it
        None: if the agent can't move, shouldn't move (the object is already close and visible) an error occurred
    """

    agent_rpos = get_agent_reachable_positions(controller)
    if not agent_rpos: # Agent can't move
        return None
    
    target_pos = target['position'] #dict

    if is_object_close(target): # Agent is already close to the object
        return None

    reachable_pos_idx = 0

    max_attempts = 20

    if get_object_type(target) == 'Fridge' or get_object_type(target) == 'Microwave':
        target_max_dist *= 2

    # Try mutliple times to execute teleport (it can happen that it doesn't work)
    for i in range(max_attempts):
        reachable_pos_idx += 1

        if i == 2 and (get_object_type(target) == 'Fridge' or get_object_type(target) == 'Microwave'):
            reachable_pos_idx -= 2

        clos_pos = get_closest_reachable_position(agent_rpos, target_pos, reachable_pos_idx)

        # Evaluate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
        rot_angle = math.atan2(-(target_pos['x'] - clos_pos['x']), target_pos['z'] - clos_pos['z'])
        if rot_angle > 0:
            rot_angle -= 2 * math.pi

        rot_angle = -(180 / math.pi) * rot_angle  # in degrees

        # Evaluate the desired horizon angle
        camera_height = controller.last_event.metadata['agent']['position']['y'] + CAMERA_HEIGHT_OFFSET
        xz_dist = math.hypot(target_pos['x'] - clos_pos['x'], target_pos['z'] - clos_pos['z'])
        hor_angle = math.atan2((target_pos['y'] - camera_height), xz_dist)
        hor_angle = (180 / math.pi) * hor_angle  # in degrees
        hor_angle *= 0.9  # adjust angle for better view

        return clos_pos, rot_angle, -hor_angle

    return None

def build_navigation_graph(reachable_positions : list[dict], grid_size: float = 0.1) -> nx.Graph:
    """Builds a navigable graph from AI2-THOR reachable positions."""
    graph = nx.Graph()
    
    # 1. Add all points as nodes (using rounded tuples as unique keys)
    for p in reachable_positions:
        node_id = (p['x'], p['y'], p['z'])
        graph.add_node(node_id, pos=p)
        
    # 2. Connect adjacent nodes
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            # Calculate distance on the X/Z plane (ignore Y height)
            dist = math.dist([n1[0], n1[2]], [n2[0], n2[2]])
            
            # If points are next to each other (allowing for small float inaccuracies)
            if dist <= grid_size * 1.1:
                graph.add_edge(n1, n2, weight=dist)
                
    return graph

def get_path_to_position(controller: Controller, target_position: dict) -> list[dict]:
    """Finds the shortest path on the custom graph."""

    agent_pos = get_agent_position(controller)

    start_node = (agent_pos['x'], agent_pos['y'], agent_pos['z'])
    target_node = (target_position['x'], target_position['y'], target_position['z'])

    graph = build_navigation_graph(get_agent_reachable_positions(controller))
    
    try:
        # Calculate A* path
        path_nodes = nx.astar_path(graph, start_node, target_node)
        
        # Convert back to AI2-THOR dictionaries
        return [graph.nodes[n]['pos'] for n in path_nodes]
    
    except nx.NetworkXNoPath:
        print("No reachable path exists between these points.")
        return []

# === OBJECTS ===
def get_object_type(object) -> str:
    return object['objectType']

def get_object_id(object : dict) -> str:
    return object['objectId']

def find_object(controller: Controller, object_name: str) -> dict[str, str]:
    """Return the object with object_name reference in the scene if found, otherwise None"""

    if not object_name:
        return None
    
    objs = get_objects_in_scene(controller)

    for obj in objs:
        if get_object_type(obj).lower() == object_name.lower():
            return obj

    return None

def filter_objects_for(objects : list, **kwargs) -> list:
    """Filter the objects list passed accordingly to **kwargs 'key:value'.\n
    If no object respects the filter 'key:value' an empty list is returned.\n
    In case that the specified key:value is not valid, an empty list is returned"""
    
    fobjs = []

    for obj in objects:
        if all(obj.get(k) == v for k,v in kwargs.items()):
            fobjs.append(obj)

    return fobjs

def get_objects_in_scene(controller: Controller, **kwargs):
    """Get objects in scene, if **kwargs is passed only the one respecting 'key:value' are returned.\n 
    If no object respects the filter 'key:value' an emtpy list is returned\n
    In case that the specified key:value is not valid, an empty list is returned"""

    if not kwargs:
        return controller.last_event.metadata["objects"]

    return filter_objects_for(controller.last_event.metadata["objects"], **kwargs)

def display_objects(objects, *args: str):
    """Display objects.\n
    Optionally specified the object characteristic to show in args\n"""

    for obj in objects :
        if args:
            for objk, objd in obj.items():
                if objk in args:
                    print(f"{objk}: {objd}")
        else:
            for objk, objd in obj.items():
                print(f"{objk}: {objd}")
        print()

def get_visible_objects_in_scene(controller: Controller):
    return get_objects_in_scene(controller, visible=True)

def get_visible_objects_around(controller: Controller):
    objs = []

    for i in range(4):
        objs.extend(get_visible_objects_in_scene(controller))
        rotate_agent_left_smoothly(controller)

    return objs

# === TASK EXECUTION ===

def execute_plan(controller: Controller, plan: list[str]):
    """Execute the plan in the Ai2Thor environment
    
    Args:
        controller: the Ai2Thor controller
        plan: list of instructione that the embodied has to execute"""
    
    for step in plan:

        action, target = task.cmd_translation( step )

        if target:
            obj = find_object(controller, target)

            if not obj:
                print(f"Error: Object '{target}' not found in the environment\n\n")
                input()
                return

        print(f" -> {step} ")

        match action:
            case task.FIND:
                reach_object(controller, obj)

            case task.PICK:
                pick_up_object(controller, obj)

            case task.PUT:
                put_object(controller, obj)

            case task.DROP:
                drop_object(controller)

            case task.THROW:
                throw_object()

            case task.MOVEHELDBACK:
                move_held_object_back()

            case task.MOVEHELDLEFT:
                move_held_object_left()

            case task.MOVEHELDRIGHT:
                move_held_object_right()

            case task.MOVEHELDUP:
                move_held_object_up()

            case task.MOVEHELDDOWN:
                move_held_object_down()

            case task.POUR:
                rotate_held_object()

            case task.PUSH:
                directional_push_object()

            case task.PULL:
                direction_pull_object()

            case task.OPEN:
                open_object(controller, obj)

            case task.CLOSE:
                close_object(controller, obj)

            case task.BREAK:
                break_object()

            case task.COOK:
                cook_object()

            case task.SLICE:
                slice_object(controller, obj)

            case task.TURNON:
                toggle_object_on()

            case task.TURNOFF:
                toggle_object_off()

            case task.DIRTY:
                dirty_object()

            case task.CLEAN:
                clean_object()

            case task.FILLLIQUID:
                fill_object_with_liquid()

            case task.EMPTYLIQUID:
                empty_object_from_liquid()

            case _:
                print(f"Action '{action}' not allowed")
        
        time.sleep(SLEEP_BETWEEN_STEPS)

def reach_object(controller : Controller, obj : dict[str, str]):
    closest_position = get_object_closest_position(controller, obj)

    if not closest_position:
        return

    closest_position, rotation_angle, horizon_angle = closest_position

    path = get_path_to_position(controller, closest_position)

    for p in path:
        controller.step(
            action = "TeleportFull",
            position = p,
            rotation = rotation_angle,
            horizon = horizon_angle,
            standing = True
        )

        controller.step(action = "Done")

def pick_up_object(controller: Controller, object : dict):
    controller.step(action="PickupObject", objectId=get_object_id(object), forceAction=True)

def put_object(controller: Controller, object: dict):
    controller.step(action="PutObject", objectId=get_object_id(object), forceAction=True)

def drop_object(controller: Controller):
    controller.step(action="DropHandObject", forceAction=True)

def throw_object():
    pass

def move_held_object_back():
    pass

def move_held_object_left():
    pass

def move_held_object_right():
    pass

def move_held_object_up():
    pass

def move_held_object_down():
    pass

def rotate_held_object():
    pass

def directional_push_object():
    pass

def direction_pull_object():
    pass

def open_object(controller: Controller, object: dict):
    controller.step(action="OpenObject", objectId=get_object_id(object), forceAction=True)

def close_object(controller: Controller, object: dict):
    controller.step(action="CloseObject", objectId=get_object_id(object), forceAction=True)

def break_object():
    pass

def cook_object():
    pass

def slice_object(controller: Controller, object: dict):
    controller.step(action="SliceObject", objectId=get_object_id(object), forceAction=True)

def toggle_object_on():
    pass

def toggle_object_off():
    pass

def dirty_object():
    pass

def clean_object():
    pass

def fill_object_with_liquid():
    pass

def empty_object_from_liquid():
    pass