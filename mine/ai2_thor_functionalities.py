#https://ai2thor.allenai.org/

from ai2thor.controller import Controller
import ai2_thor_task as task
import time
import numpy as np
from scipy import spatial
import math
import networkx as nx
import custom_exceptions as ex

# === DEFAULT VALUES ===
SLEEP_BETWEEN_STEPS = 0.0001
CAMERA_HEIGHT_OFFSET = 0.675
TARGET_MAX_DISTANCE = 1.0
MAX_ATTEMPTS = 20 # NUmber of times an action is repeated before throwing an exception and closing the program

# === CREATION ===
def create_controller(agentMode = "default", visibilityDistance = 1.5, scene = "FloorPlan1", 
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

# === UTILS ===

def last_action_state(controller : Controller):
    return controller.last_event.metadata['lastActionSuccess']

def print_metadata(controller : Controller):
    for k, v in controller.last_event.metadata.items():
        print(f"\n{k} : {v}\n")

def print_object_info(object : dict[str, str], *args : str):
    for k, v in object.items():
        if not args:
            print(f"{k} : {v}")
        elif k in args:
            print(f"{k} : {v}")

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

def get_object_closest_position(controller: Controller, target : dict[str, str], target_max_dist=TARGET_MAX_DISTANCE, nth = 1) -> tuple[dict, float, float] | None:
    """
    Based on a target provided evaluates the closesest position and camera rotation near the object

    Args:
        controller: Ai2THOR controller
        target: the target object obtained by the controller metadata
        target_max_dist: the maximum distance that the agent has to have to the object
        nth: represent the 'yet another' closest point. By default, nth=1 means "give me the #1 closest point." If a spot is blocked, you could pass nth=2 to get the 2nd closest point, and so on

    Returns:
        tuple[dict, float, float]: Containing the closest position to the object, the rotation and horizon that the agent has to have to be close to the object and look at it
        None: if the agent can't move, shouldn't move (the object is already close and visible) an error occurred
    """

    agent_rpos = get_agent_reachable_positions(controller)
    if not agent_rpos: # Agent can't move
        return None, None, None
    
    target_pos = target['position'] # dict

    if is_object_close(target): # Agent is already close to the object
        return None, None, None

    clos_pos = get_closest_reachable_position(agent_rpos, target_pos, nth)
    
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
    hor_angle = -hor_angle # adjusting the direction that is the opposite of the one evaluated

    if hor_angle < -30:
        hor_angle = -30
    elif hor_angle > 60:
        hor_angle = 60

    return clos_pos, rot_angle, hor_angle

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
        path_nodes = [graph.nodes[n]['pos'] for n in path_nodes]

        return path_nodes[1:]
    
    except nx.NetworkXNoPath:
        return []

def teleport_to_free_position(controller : Controller):
    """Try to teleport the agent in a free position, without animation.
    This action should be used only if the agent is stucked in a position and should free itself
    
    Raises:
        Ai2THORException: if the teleport cannot be done in MAX_ATTEMPTS times"""

    for j in range(1, MAX_ATTEMPTS):
        free_position = get_closest_reachable_position(get_agent_reachable_positions(controller), get_agent_position(controller), j)
                    
        controller.step(
            action = "Teleport",
            position = free_position,
            standing = True
        )

        if last_action_state(controller):
            return
        
    raise ex.Ai2THORException(controller)

# === OBJECTS ===

def get_object_type(object : dict) -> str:
    return object['objectType']

def is_object_type(object : dict, object_type : str):
    return get_object_type(object).lower() == object_type.lower()

def is_object_close(target : dict[str, str], target_max_dist = TARGET_MAX_DISTANCE) -> bool:
    return target['visible'] and target['distance'] < target_max_dist

def get_object_id(object : dict) -> str:
    return object['objectId']

def get_object_by_type(controller: Controller, object_type: str) -> dict[str, str]:
    """Return the object with object_name reference in the scene if found, otherwise None"""

    if not object_type:
        raise ex.ObjectException(f"The object type given is not valid")
    
    objs = get_objects_in_scene(controller)

    for obj in objs:
        if is_object_type(obj, object_type):
            return obj

    return None

def get_object_by_id(controller : Controller, object_id : str) -> dict[str, str]:
    if not object_id:
        raise ex.ObjectException(f"The object id given is not valid")

    objs = get_objects_in_scene(controller)

    for obj in objs:
        if get_object_id(obj) == object_id:
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

def get_objects_in_scene(controller: Controller, **kwargs) -> list[dict]:
    """Access the scene metadata to scan for objects
        
        Args:
            controller: the Ai2THOR controller
            kwargs: can be None or a couple 'key:value'. If it is passed only the objects respecting 'key:value' are returned.\n 
                    If no object respects the filter 'key:value' an emtpy list is returned\n
                    In case that the specified key:value is not valid, an empty list is returned
            
        Returns:
            list: containing all the objects in the scene"""

    objects = controller.last_event.metadata['objects']

    if not kwargs:
        return objects

    return filter_objects_for(objects, **kwargs)

def display_objects(objects : list[dict], *args: str):
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

def get_objects_around(controller: Controller, **kwargs):
    """Returns the object in the scene and perform a fake scanning of the ambient to simulate the agent scanning the are

    Args:
        controller: the Ai2THOR controller
        kwargs: can be None or a couple 'key:value'. If it is passed only the objects respecting 'key:value' are returned.\n 
                If no object respects the filter 'key:value' an emtpy list is returned\n
                In case that the specified key:value is not valid, an empty list is returned
    """
    objs = get_objects_in_scene(controller, **kwargs)

    for i in range(4):
        rotate_agent_left_smoothly(controller)

    return objs

def get_object_parent_receptacles(object : dict):
    return object['parentReceptacles']

def is_object_interactable(object : dict):
    return object['visible'] and object['isInteractable']

def get_inherited_objects(controller : Controller, primary_object : dict[str, str] = None):
    objects = get_objects_in_scene(controller)

    inh_objs = []

    for obj in objects:
        if len(get_object_id(obj).split('|')) == 5:
            inh_objs.append(obj)

    if primary_object:
        for inh_obj in inh_objs:
            if get_object_id(primary_object) not in get_object_id(inh_obj):
                inh_objs.remove(inh_obj)

    return inh_objs if inh_objs else None

def get_agent_inventory(controller : Controller):
    return controller.last_event.metadata['inventoryObjects']

def get_agent_holded_object(controller : Controller):
    """
    Returns:
        inventory_object: if the agent is holding an object otherwise an exception is raised"""
    inventory_objects = get_agent_inventory(controller)

    if not inventory_objects:
        raise ex.HoldingObjectsException("The robot is not holding any object")
    elif len(inventory_objects) > 1:
        raise ex.HoldingObjectsException("To many objects in hand")

    return inventory_objects[0]
    
# === TASK EXECUTION ===

def execute_plan(controller: Controller, plan: list[str]) -> int:
    """Execute the plan in the Ai2Thor environment
    
    Args:
        controller: the Ai2Thor controller
        plan: list of instructione that the embodied has to execute
    """
    
    for step in plan:

        print(f"-> {step}")

        action = task.get_action_from_cmd( step )

        if not task.is_action(action):
            raise ex.BadActionFormat("Action not recognized by the agent")

        target = task.get_subjects_from_cmd( step )

        match action:
            case task.DROP | task.THROW | task.MOVEHELDBACK | task.MOVEHELDLEFT | task.MOVEHELDRIGHT | task.MOVEHELDUP | task.MOVEHELDDOWN:
                if target:
                    raise ex.BadActionFormat(f"Action '{action}' should not contain a target")
            case task.FIND | task.PICK | task.PUT | task.PUSH | task.PULL | task.OPEN | task.CLOSE | task.BREAK | task.COOK | task.SLICE | task.TURNON | task.TURNOFF | task.DIRTY | task.CLEAN:
                if ((not target) or (len(target) > 1)):
                    raise ex.BadActionFormat(f"Action '{action}' should contain a single target")
                else:
                    target = target[0]
                
                obj = get_object_by_type(controller, target)

                print_object_info(obj)

                if not obj:
                    raise ex.BadActionFormat(f"{action.capitalize()} target not found")
            case task.FILLLIQUID | task.EMPTYLIQUID:
                if (not target) or (len(target) != 2):
                    raise ex.BadActionFormat(f"Action '{action}' should contain two targets")

                obj = get_object_by_type(controller, target[0])

                liquid = target[1]

                if not obj:
                    raise ex.BadActionFormat(f"{action.capitalize()} target not found")

                if not liquid:
                    raise ex.BadActionFormat(f"To perform '{action}' a valid liquid must be specified")
                elif not task.is_liquid(liquid):
                    raise ex.BadActionFormat(f"The liquid '{liquid}' is not allowed, only available liquid are {task.get_available_liquids()}")
                

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
                throw_object(controller)

            case task.MOVEHELDBACK:
                move_held_object_back(controller)

            case task.MOVEHELDLEFT:
                move_held_object_left(controller)

            case task.MOVEHELDRIGHT:
                move_held_object_right(controller)

            case task.MOVEHELDUP:
                move_held_object_up(controller)

            case task.MOVEHELDDOWN:
                move_held_object_down(controller)

            case task.POUR:
                rotate_held_object(controller)

            case task.PUSH:
                directional_push_object(controller, obj)

            case task.PULL:
                direction_pull_object(controller, obj)

            case task.OPEN:
                open_object(controller, obj)

            case task.CLOSE:
                close_object(controller, obj)

            case task.BREAK:
                break_object(controller, obj)

            case task.COOK:
                cook_object(controller, obj)

            case task.SLICE:
                slice_object(controller, obj)

            case task.TURNON:
                toggle_object_on(controller, obj)

            case task.TURNOFF:
                toggle_object_off(controller, obj)

            case task.DIRTY:
                dirty_object(controller, obj)

            case task.CLEAN:
                clean_object(controller, obj)

            case task.FILLLIQUID:
                fill_object_with_liquid(controller, obj, liquid)

            case task.EMPTYLIQUID:
                empty_object_from_liquid(controller, obj)

            case _:
                print(f"Action '{action}' not allowed")
        
        time.sleep(SLEEP_BETWEEN_STEPS)

def resilient_execution(controller : Controller, **kwargs):
    controller.step(**kwargs)

    if not last_action_state(controller):
        for i in range(MAX_ATTEMPTS):
            teleport_to_free_position(controller)

            controller.step(**kwargs)

            if last_action_state(controller):
                break
        else:
            raise ex.Ai2THORException(controller)
    
    controller.step(action = "Done")

def reach_object(controller : Controller, obj : dict[str, str]):

    nth = 1

    for i in range(MAX_ATTEMPTS):

        closest_position, rotation_angle, horizon_angle = get_object_closest_position(controller, obj, nth)

        if (not closest_position):
            return

        path = get_path_to_position(controller, closest_position)

        for p in path:
            controller.step(
                action = "TeleportFull",
                position = p,
                rotation = {'x': 0, 'y': rotation_angle, 'z': 0},
                horizon = horizon_angle,
                standing = True
            )

            if not last_action_state(controller):
                if ( ex.Ai2THORException(controller).is_collision() ):
                    teleport_to_free_position(controller)
                
                if i == 10 :
                    nth -= 10
                else:
                    nth += 1
                
                break
            else:
                controller.step(action = "Done")
        else:
            break

def pick_up_object(controller: Controller, object : dict):

    if not is_object_close(object):
        raise ex.InteractionException("The object is not close to the agent")
    elif get_object_parent_receptacles(object) and ( not is_object_interactable(object) ):
        raise ex.InteractionException(f"Cannot interact with the object because it is contained in {get_object_parent_receptacles}")
    elif get_agent_inventory(controller):
        raise ex.HoldingObjectsException("Agent can only pick up one object at a time")

    resilient_execution(controller,
        action = "PickupObject",
        objectId = get_object_id(object),
        forceAction = False
    )

def put_object(controller: Controller, receptacle: dict):

    inventory_object = get_agent_holded_object(controller)

    controller.step(
        action="PutObject", 
        objectId=get_object_id(receptacle),
        forceAction=False
    )

    if not last_action_state(controller):

        # Try to put the object over the receptacle
        controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=get_object_id(receptacle),
            anywhere=False
        )

        position_above = controller.last_event.metadata['actionReturn']

        controller.step(
            action="PlaceObjectAtPoint",
            objectId=get_object_id(inventory_object),
            position = {
                "x": sum([tmp['x'] for tmp in position_above])/len(position_above),
                "y": sum([tmp['y'] for tmp in position_above])/len(position_above),
                "z": sum([tmp['z'] for tmp in position_above])/len(position_above)
            }
        )

        if last_action_state(controller):
            if get_object_id(receptacle) in get_object_by_id(controller, get_object_id(inventory_object))['parentReceptacles']:
                controller.step(action = "Done")
                return

        # Receptacle is full, so another one is searched in the environment
        
        recepts = get_objects_in_scene(controller, receptacle = True, objectType = receptacle['objectType'])

        for rec in recepts:
            if get_object_id(rec) != get_object_id(receptacle):
                reach_object(controller, rec)

                controller.step(
                    action="PutObject",
                    objectId=get_object_id(rec),
                    forceAction=False
                )

                if not last_action_state(controller):
                    continue
                else:
                    controller.step(action = "Done")
                    break
        else:
            raise ex.ReceptacleException(f"No {get_object_type(receptacle)} can hold the object in hand")
    else:
        controller.step(action = "Done")

def drop_object(controller: Controller):

    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "DropHandObject",
        forceAction = False
    )

def throw_object(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action="ThrowObject",
        moveMagnitude=1500.0,
        forceAction=False
    )

def move_held_object_back(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "MoveHeldObjectBack",
        moveMagnitude =0.1,
        forceVisible=True
    )

def move_held_object_left(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "MoveHeldObjectLeft",
        moveMagnitude =0.1,
        forceVisible=True
    )

def move_held_object_right(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "MoveHeldObjectRight",
        moveMagnitude =0.1,
        forceVisible=True
    )

def move_held_object_up(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "MoveHeldObjectUp",
        moveMagnitude =0.1,
        forceVisible=True
    )

def move_held_object_down(controller : Controller):
    get_agent_holded_object(controller)

    resilient_execution(controller,
        action = "MoveHeldObjectDown",
        moveMagnitude =0.1,
        forceVisible=True
    )

def rotate_held_object(controller : Controller, pour = True):
    holded_object = get_object_by_id(get_object_id(get_agent_holded_object(controller)))

    if not holded_object:
        raise ex.HoldingObjectsException(f"Cannot find the object in the scene")
    elif pour and (not holded_object['isFilledWithLiquid']):
        raise ex.InteractionException(f"The object '{holded_object}' is not filled with any liquid")

    degree_step = 60.0

    while degree_step < 360.0:
        resilient_execution(controller,
            action = "RotateHeldObject",
            pitch = degree_step
        )

        degree_step += 30.0

    if pour and get_object_by_id(get_object_id(get_agent_holded_object(controller)))['isFilledWithLiquid']:
        raise ex.InteractionException("The liquid cannot be poured from the object")

def directional_push_object(controller : Controller, object : dict[str, str]):
    resilient_execution(controller,
        action="DirectionalPush",
        objectId=get_object_id(object),
        moveMagnitude="100",
        pushAngle="0"
    )

def direction_pull_object(controller : Controller, object : dict[str, str]):
    resilient_execution(controller,
        action="DirectionalPush",
        objectId=get_object_id(object),
        moveMagnitude="100",
        pushAngle="180"
    )

def open_object(controller: Controller, object: dict):
    steps_num = 4

    openness = 1.0 / steps_num

    if object['openable'] and object['openness'] < 1.0:
    
        for i in range(steps_num):
            controller.step(
                action="OpenObject",
                objectId=get_object_id(object),
                openness = openness,
                forceAction=False
            )
    
            if not last_action_state(controller):    
                for j in range(MAX_ATTEMPTS):
                    teleport_to_free_position(controller)
    
                    controller.step(
                        action="OpenObject",
                        objectId=get_object_id(object),
                        openness = 1.0,
                        forceAction=False
                    )
    
                    if last_action_state(controller):
                        break
                else:
                    raise ex.Ai2THORException(controller)
            
            openness += openness
            controller.step(action="MoveBack")
            controller.step(action = "Done")

    elif not object['openable']:
        raise ex.InteractionException(f"The object '{get_object_type(object)}' cannot be opened")

def close_object(controller: Controller, object: dict):
    if object['openable'] and object['openness'] > 0.0:

        resilient_execution(controller,
            action="CloseObject",
            objectId=get_object_id(object),
            forceAction=False
        )

    elif not object['openable']:
        raise ex.InteractionException(f"The object '{get_object_type(object)}' cannot be closed")

def break_object(controller : Controller, object : dict):
    if not object['breakable']:
        raise ex.ObjectException("The selected object cannot be broken")
    elif object['isBroken']:
        raise ex.ObjectException("The selected object is already broken")

    resilient_execution(controller,
        action="BreakObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def cook_object(controller : Controller, object : dict[str, str]):

    if not object['cookable']:
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be cooked")
    elif object['isCooked']:
        raise ex.InteractionException(f"{object['name'].capitalize()} is already cooked")

    resilient_execution(controller,
        action="CookObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def slice_object(controller: Controller, object: dict):

    get_agent_holded_object(controller)

    if object['sliceable'] and (not object['isSliced']):

        resilient_execution(controller,
            action="SliceObject", 
            objectId=get_object_id(object),
            forceAction=False
        )

def toggle_object_on(controller : Controller, object : dict[str, str]):
    if not object['toggleable']:
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} cannot be toggled on")
    elif object['isToggled']:
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} is already toggled on")

    resilient_execution(controller,
        action = "ToggleObjectOn",
        objectId=get_object_id(object),
        forceAction = False
    )

def toggle_object_off(controller : Controller):
    if not object['toggleable']:
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} cannot be toggled off")
    elif not object['isToggled']:
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} is already toggled off")

    resilient_execution(controller,
        action = "ToggleObjectOff",
        objectId=get_object_id(object),
        forceAction = False
    )

def dirty_object(controller : Controller, object : dict[str, str]):
    if not object['dirtyable']:
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be dirty")
    elif object['isDirty']:
        raise ex.InteractionException(f"{object['name'].capitalize()} is already dirty")

    resilient_execution(controller,
        action="DirtyObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def clean_object(controller : Controller):
    if not object['dirtyable']:
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be cleaned since it cannot be dirty")
    elif not object['isDirty']:
        raise ex.InteractionException(f"{object['name'].capitalize()} is not dirty")

    resilient_execution(controller,
        action = "CleanObject",
        objectId = get_object_id(object),
        forceAction = False
    )


def fill_object_with_liquid(controller : Controller, object : dict[str, str], liquid : str):
    if not object['canFillWithLiquid']:
        raise ex.ObjectException("The object cannot be filled with any liquid")
    elif object['isFilledWithLiquid'] and object['fillLiquid']:
        raise ex.InteractionException(f"The object is already filled with '{object['fillLiquid']}'")

    resilient_execution(controller,
        action="FillObjectWithLiquid",
        objectId=get_object_id(object),
        fillLiquid=liquid,
        forceAction = False
    )

def empty_object_from_liquid(controller : Controller, object : dict[str, str]):
    if not object['isFilledWithLiquid']:
        raise ex.InteractionException("The object is already empty")

    resilient_execution(controller,
        action="EmptyLiquidFromObject",
        objectId=get_object_id(object),
        forceAction = False
    )