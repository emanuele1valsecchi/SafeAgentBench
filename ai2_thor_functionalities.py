#https://ai2thor.allenai.org/

from ai2thor.controller import Controller
import numpy as np
from scipy import spatial
import math
import networkx as nx
import custom_exceptions as ex
from utils import print_log

# === DEFAULT VALUES ===
SLEEP_BETWEEN_STEPS = 0.0001
CAMERA_HEIGHT_OFFSET = 0.675
TARGET_MAX_DISTANCE = 1.5
MAX_ATTEMPTS = 20 # NUmber of times an action is repeated before throwing an exception and closing the program

# === CREATION ===
def create_controller(agentMode = "default", 
                      visibilityDistance = TARGET_MAX_DISTANCE,
                      scene = "FloorPlan1", 
                      gridSize = 0.1, 
                      snapToGrid = False,
                      rotationStepDegrees = 1,
                      renderDepthImage = True, 
                      renderInstanceSegmentation = True, 
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

# === ENVIRONMENT MODIFICATION ===
def change_brightness(controller : Controller, min : float, max : float):
    controller.step(
        action="RandomizeLighting",
        brightness=(min, max),
        randomizeColor=True,
        hue=(0, 1),
        saturation=(0.5, 1),
        synchronized=False
    )

    if last_action_state(controller):
        controller.step(action = "Done")
        return True

    return False

def move_object_at(controller : Controller, object_id : str, receptacle_id : str):

    moving_obj = get_object_by_id(controller, object_id)

    if not is_pickupable(moving_obj) and not moving_obj['moveable']:
        raise ex.BadActionFormat(f"{get_object_type(moving_obj)} cannot be moved on the desired receptacle")

    positions = get_position_above_object(controller, receptacle_id)
    
    if not positions:
        return False

    for pos in positions:
        controller.step(
            action="PlaceObjectAtPoint",
            objectId=object_id,
            position= pos
        )

        if last_action_state(controller):
            controller.step(action = "Done")
            return True

    return False

# === UTILS ===

def last_action_state(controller : Controller):
    f"""Access the {controller} element and returns 'lastActionSuccess' property"""
    return controller.last_event.metadata['lastActionSuccess']

def print_metadata(controller : Controller):
    for k, v in controller.last_event.metadata.items():
        print_log(f"\n{k} : {v}\n")

def print_object_info(object : dict[str, str], *args : str):
    for k, v in object.items():
        if not args:
            print_log(f"{k} : {v}")
        elif k in args:
            print_log(f"{k} : {v}")
    print_log()

# === POSITION ===
def get_agent_position(controller : Controller) -> dict:
    return controller.last_event.metadata['agent']['position']

def get_agent_rotation_y(controller : Controller):
    return controller.last_event.metadata['agent']['rotation']['y']

def get_normalized_horizon(current_horizon : float):
    if current_horizon > 180:
        current_horizon -= 360
    
    return max(-30.0, min(60.0, current_horizon))

def get_agent_normalized_horizon(controller : Controller):
    return get_normalized_horizon(controller.last_event.metadata['agent']['cameraHorizon'])

def get_agent_reachable_positions(controller: Controller) -> list[dict]:
    """Get the agent's reachable position in the scene."""
    return controller.step(action="GetReachablePositions").metadata["actionReturn"]

def navigate_to(controller: Controller, target_position, steps = 60):
    """
    Interpolates the agent's position and camera to create a fluid motion.
    """

    start_pos = get_agent_position(controller)
    start_rot = get_agent_rotation_y(controller)
    start_hor = get_agent_normalized_horizon(controller)

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

        controller.step( action = "Done")

# === AGENT MOVEMENT ===
def rotate_agent_smoothly(controller: Controller, direction, total_degrees=90, step = 10):
    """Rotate the agent smoothly by stepping through smaller rotation increments.

    Args:
        direction: 'left' or 'right'
        total_degrees: how many degrees to rotate in total
        step: degrees per step"""

    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")

    start_pos = get_agent_position(controller)
    start_rot = get_agent_rotation_y(controller)
    start_hor = get_agent_normalized_horizon(controller)

    # Determine target rotation based on direction and degrees
    multiplier = 1 if direction == "right" else -1
    target_rot = start_rot + (total_degrees * multiplier)
    rot_diff = (target_rot - start_rot + 180) % 360 - 180

    steps = max(1, abs(int(rot_diff / step)))

    for i in range(1, steps + 1):
        t = i / steps
        cur_rot = start_rot + rot_diff * t
        controller.step(
            action="Teleport",
            position=start_pos,
            rotation={'x': 0, 'y': cur_rot, 'z': 0},
            horizon=start_hor,
            forceAction=True
        )

        controller.step( action = "Done")
        
def rotate_agent_left_smoothly(controller: Controller, total_degrees=90, step=10):
    rotate_agent_smoothly(controller, "left", total_degrees, step)

def rotate_agent_right_smoothly(controller: Controller, total_degrees=90, step=10):
    rotate_agent_smoothly(controller, "right", total_degrees, step)

def get_kdtree_reachable_positions(agent_reachable_positions : list[dict]) -> spatial._kdtree.KDTree:
    return spatial.KDTree(np.array([[p['x'], p['y'], p['z']] for p in agent_reachable_positions]))

def should_agent_stand(target_pos : dict):
    return target_pos['y'] > 0.6

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
    clos_pos = get_closest_reachable_position(agent_rpos, target_pos, nth)
    
    # Evaluate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
    rot_angle = math.atan2(-(target_pos['x'] - clos_pos['x']), target_pos['z'] - clos_pos['z'])
    if rot_angle > 0:
        rot_angle -= 2 * math.pi
    
    rot_angle = -(180 / math.pi) * rot_angle  # in degrees

    # DYNAMIC POSTURE: Determine if the object is low enough to crouch
    camera_offset = CAMERA_HEIGHT_OFFSET if (should_agent_stand(target_pos)) else 0.0

    # Evaluate the desired horizon angle
    camera_height = controller.last_event.metadata['agent']['position']['y'] + camera_offset
    xz_dist = math.hypot(target_pos['x'] - clos_pos['x'], target_pos['z'] - clos_pos['z'])
    hor_angle = math.atan2((target_pos['y'] - camera_height), xz_dist)
    hor_angle = (180 / math.pi) * hor_angle  # in degrees
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
    This action should be used only if the agent is stuck in a position and should free itself
    
    Raises:
        Ai2THORException: if the teleport cannot be done in MAX_ATTEMPTS times"""

    current_horizon = get_normalized_horizon(controller.last_event.metadata['agent']['cameraHorizon'])

    for j in range(1, MAX_ATTEMPTS):
        free_position = get_closest_reachable_position(get_agent_reachable_positions(controller), get_agent_position(controller), j)
                    
        controller.step(
            action = "Teleport",
            position = free_position,
            horizon = current_horizon,
            standing = True
        )

        if last_action_state(controller):
            controller.step( action = "Done")
            return
        
    raise ex.Ai2THORException(controller)

def rotate_thoward_direction(controller : Controller, target_point : dict):

    # Evaluating agent current position
    start_pos = get_agent_position(controller)
    start_rot = get_agent_rotation_y(controller)

    # Calculating new angle rotation based on the new position to reach
    move_rot_angle = math.atan2(-(target_point['x'] - start_pos['x']), target_point['z'] - start_pos['z'])
    if move_rot_angle > 0:
        move_rot_angle -= 2 * math.pi
    move_rot_angle = -(180 / math.pi) * move_rot_angle

    rot_diff = (move_rot_angle - start_rot + 180) % 360 - 180
    if rot_diff != 0:
        direction = "right" if rot_diff > 0 else "left"
        # Use your existing smooth rotation helper function
        rotate_agent_smoothly(controller, direction, total_degrees=abs(rot_diff))

    return move_rot_angle

def look_at_object(controller: Controller, target: dict[str, str]):
    """Smoothly adjusts the agent's rotation and camera horizon to look directly at the target object."""
    agent_pos = get_agent_position(controller)
    start_rot = get_agent_rotation_y(controller)
    start_hor = get_agent_normalized_horizon(controller)

    target_pos = target['position']
    standing = should_agent_stand(target_pos)

    # Calculate target rotation angle
    rot_angle = math.atan2(-(target_pos['x'] - agent_pos['x']), target_pos['z'] - agent_pos['z'])
    if rot_angle > 0:
        rot_angle -= 2 * math.pi
    rot_angle = -(180 / math.pi) * rot_angle

    camera_offset = CAMERA_HEIGHT_OFFSET if standing else 0.0

    # Calculate target horizon angle
    camera_height = agent_pos['y'] + camera_offset
    xz_dist = math.hypot(target_pos['x'] - agent_pos['x'], target_pos['z'] - agent_pos['z'])
    hor_angle = math.atan2((target_pos['y'] - camera_height), xz_dist)
    hor_angle = (180 / math.pi) * hor_angle
    hor_angle = -hor_angle

    if hor_angle < -30:
        hor_angle = -30
    elif hor_angle > 60:
        hor_angle = 60

    # Shortest path calculation for smooth rotation transition
    rot_diff = (rot_angle - start_rot + 180) % 360 - 180
    steps = 10  # Number of smoothing frames to pan the camera

    for step in range(1, steps + 1):
        t = step / steps
        interp_rot = start_rot + rot_diff * t
        interp_hor = start_hor + (hor_angle - start_hor) * t

        controller.step(
            action="Teleport",
            position=agent_pos,
            rotation={'x': 0, 'y': interp_rot, 'z': 0},
            horizon=interp_hor,
            standing= standing,
            forceAction=True
        )

        controller.step( action = "Done")

    controller.step(action="Done")

# === OBJECTS ===

def get_object_type(object : dict) -> str:
    return object['objectType']

def get_object_type_from_id(object_id : str):
    return object_id.split("|")[0].strip()

def is_object_type(object : dict, object_type : str):
    return get_object_type(object).lower() == object_type.lower()

def is_object_close(target : dict[str, str], target_max_dist = TARGET_MAX_DISTANCE) -> bool:
    return target['visible'] and target['distance'] < target_max_dist

def get_object_id(object : dict) -> str:
    return object['objectId']

def get_object_name(object : dict[str, str]):
    return object['name']

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

def get_object_by_name(controller : Controller, object_name : str) -> dict[str, str]:
    if not object_name:
        raise ex.ObjectException(f"The object id given is not valid")

    objs = get_objects_in_scene(controller)

    for obj in objs:
        if get_object_name(obj) == object_name:
            return obj

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
                    print_log(f"{objk}: {objd}")
        else:
            for objk, objd in obj.items():
                print_log(f"{objk}: {objd}")
        print_log()

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

def get_object_parent_receptacles_type(controller : Controller, object : dict):
    parent_receptacle = get_object_parent_receptacles(object)

    if parent_receptacle:
        parent_receptacle = parent_receptacle[0]

        if parent_receptacle:
            return get_object_type(get_object_by_id(controller, parent_receptacle))

    return None

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

def get_inherited_parent_object(controller : Controller, inherited_object : dict[str, str]):
    if len(get_object_id(inherited_object).split("|")) < 5:
        return None

    return get_object_by_id(controller, get_object_id(inherited_object).rsplit("|", 1)[0])

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

def is_object_in_receptacle(controller : Controller, object : dict[str, str], receptacle : dict[str, str]) -> bool:
    receptacles = get_object_parent_receptacles(get_object_by_id(controller, get_object_id(object)))

    if receptacles:
        return get_object_id(receptacle) in receptacles

    return False

def is_right_receptacle(controller : Controller, inventory_object : dict[str, str], receptacle :  dict[str, str]) -> bool:
    if is_object_in_receptacle(controller, inventory_object, receptacle):
        controller.step(action = "Done")
        return True
    else:
        return False

def right_receptacle_or_pickup(controller : Controller, inventory_object : dict[str, str], receptacle : dict[str, str]) -> bool:
    if last_action_state(controller):
        for _ in range(MAX_ATTEMPTS):
            controller.step(action="Done")

        if is_right_receptacle(controller, inventory_object, receptacle):
            controller.step(
                action = "Done"
            )
            return True
        else: #Pickup again the object if the receptacle is not right
            resilient_execution(controller,
                action = "PickupObject",
                objectId = get_object_id(inventory_object),
                forceAction = True
            )

    return False

def try_place_at_point(
        controller : Controller, 
        position_above : dict[str, str], 
        inventory_object : dict[str, str], 
        receptacle : dict[str, str]
    ):
    if get_agent_inventory(controller):
        controller.step(action="DropHandObject", forceAction=True)

    safe_position = position_above

    max_y_offset = position_above['y'] + 0.1

    for _ in range(MAX_ATTEMPTS):
        if safe_position['y'] > max_y_offset:
            break

        controller.step(
            action="PlaceObjectAtPoint",
            objectId=get_object_id(inventory_object),
            position=safe_position
        )

        if right_receptacle_or_pickup(controller, inventory_object, receptacle):
            return True

        safe_position = {
            'x' : safe_position['x'],
            'y' : safe_position['y'] + 0.02,
            'z' : safe_position['z']
        }

    return False

def get_object_position(controller : Controller, object : dict[str, str]):
    object = get_object_by_id(controller, get_object_id(object))

    return object['position']['x'], object['position']['y'], object['position']['z']

def remove_object_from_scene(controller : Controller, object_id : str):
    controller.step(
        action="DisableObject",
        objectId=object_id
    )

    if last_action_state(controller):
        controller.step(action="Done")
        return True

    return False

def get_position_above_object(controller : Controller, object_id : str):
    controller.step(
            action="GetSpawnCoordinatesAboveReceptacle",
            objectId=object_id,
            anywhere=True
        )
    
    return controller.last_event.metadata['actionReturn']

def is_toggleable(object : dict[str, str]):
    return object['toggleable']

def is_on(object : dict[str, str]) -> bool:
    return object['isToggled']

def is_breakable(object : dict[str, str]):
    return object['breakable']

def is_broken(object : dict[str, str]):
    return object['isBroken']

def can_contain_liquid(object : dict[str, str]):
    return object['canFillWithLiquid']

def contains_liquid(object : dict[str, str]):
    return object['isFilledWithLiquid']

def get_liquid_inside(object : dict[str, str]) -> str:
    return object['fillLiquid']

def is_dirtable(object : dict[str, str]):
    return object['dirtyable']

def is_dirty(object : dict[str, str]):
    return object['isDirty']

def is_cookable(object : dict[str, str]):
    return object['cookable']

def is_cooked(object : dict[str, str]):
    return object['isCooked']

def is_sliceable(object : dict[str, str]):
    return object['sliceable']

def is_sliced(object : dict[str, str]):
    return object['isSliced']

def is_openable(object : dict[str, str]):
    return object['openable']

def is_open(object : dict[str, str]):
    return object['isOpen']

def get_openness(object : dict[str, str]):
    return object['openness']

def is_completely_open(object : dict[str, str]):
    return is_open(object) and get_openness(object) == 1.0

def is_pickupable(object : dict[str, str]):
    return object['pickupable']

def is_picked_up(object : dict[str, str]):
    return object['isPickedUp']

# === TASK EXECUTION ===

def resilient_execution(controller : Controller, **kwargs):
    controller.step(**kwargs)

    if not last_action_state(controller):
        for i in range(MAX_ATTEMPTS):
            teleport_to_free_position(controller)

            controller.step(**kwargs)

            if last_action_state(controller):
                break

            if i == (MAX_ATTEMPTS - 1):

                for k in kwargs.keys():
                    if k == "forceAction":
                        kwargs[k] = True

                controller.step(**kwargs)

                if last_action_state(controller):
                    break
        else:
            raise ex.Ai2THORException(controller)
    
    controller.step(action = "Done")

def is_reached_object(controller : Controller, target : dict):
    target = get_object_by_id(controller, get_object_id(target))

    if target and is_object_close(target):
        if not get_object_parent_receptacles(target) or is_object_interactable(target):
            return True

    return False

def reach_object(controller : Controller, obj : dict[str, str]):

    nth = 1
    standing = should_agent_stand(obj['position'])

    for i in range(MAX_ATTEMPTS):

        closest_position, rotation_angle, horizon_angle = get_object_closest_position(controller, obj, nth)

        if (not closest_position):
            return

        path = get_path_to_position(controller, closest_position)

        for p in path:
            move_rot_angle = rotate_thoward_direction(controller, p)

            controller.step(
                action = "TeleportFull",
                position = p,
                rotation = {'x': 0, 'y': move_rot_angle, 'z': 0},
                horizon = 0.0,
                standing = standing
            )

            if not last_action_state(controller):
                if ( ex.Ai2THORException(controller).is_collision() ):
                    teleport_to_free_position(controller)
                
                nth += 1
                break
            else:
                controller.step(action = "Done")
        else:
            controller.step(
                action = "Teleport",
                position = closest_position,
                rotation = {'x': 0, 'y': rotation_angle, 'z': 0},
                horizon = horizon_angle,
                standing = standing,
                forceAction = True
            )

            if last_action_state(controller):
                controller.step(action = "Done")
                look_at_object(controller, obj)

                if is_reached_object(controller, obj):
                    break

                nth += 1
            else:
                if ( ex.Ai2THORException(controller).is_collision() ):
                    teleport_to_free_position(controller)
                
                nth += 1
    
def pick_up_object(controller: Controller, object : dict):

    if not is_object_close(object):
        raise ex.InteractionException("THe object is not close to the agent")
    elif get_object_parent_receptacles(object) and ( not is_object_interactable(object) ):
        raise ex.InteractionException(f"Cannot interact with the object because it is contained in {get_object_parent_receptacles(object)}")
    elif get_agent_inventory(controller):
        raise ex.HoldingObjectsException("Agent can only pick up one object at a time")

    resilient_execution(controller,
        action = "PickupObject",
        objectId = get_object_id(object),
        forceAction = False
    )

def put_object(controller: Controller, receptacle: dict[str, str], excluded_receptacle_ids : set[str] = None):

    if not excluded_receptacle_ids:
        excluded_receptacle_ids = set()

    if get_object_id(receptacle) in excluded_receptacle_ids:
        return

    inventory_object = get_object_by_id(controller, get_object_id(get_agent_holded_object(controller)))

    controller.step(
        action="PutObject", 
        objectId=get_object_id(receptacle),
        forceAction=False,
        placeStationary = False
    )

    if right_receptacle_or_pickup(controller, inventory_object, receptacle):
        return

    controller.step(
        action="PutObject", 
        objectId=get_object_id(receptacle),
        forceAction=False,
        placeStationary = True
    )

    if right_receptacle_or_pickup(controller, inventory_object, receptacle):
        return

    # Force to put the object in the target receptacle
    controller.step(
        action="PutObject", 
        objectId=get_object_id(receptacle),
        forceAction=True,
        placeStationary = False
    )
    
    if right_receptacle_or_pickup(controller, inventory_object, receptacle):
        return

    controller.step(
        action="PutObject", 
        objectId=get_object_id(receptacle),
        forceAction=True,
        placeStationary = True
    )
    
    if right_receptacle_or_pickup(controller, inventory_object, receptacle):
        return

    # Try to put the object over the receptacle centroid if the default action has not been executed successfully
    position_above = get_position_above_object(controller, get_object_id(receptacle))

    # Trying the centroid above the receptacle
    if position_above:
        centroid = {
            "x": sum([tmp['x'] for tmp in position_above])/len(position_above),
            "y": sum([tmp['y'] for tmp in position_above])/len(position_above),
            "z": sum([tmp['z'] for tmp in position_above])/len(position_above)
        }

        if try_place_at_point(controller, centroid, inventory_object, receptacle):
            return

        #Try all the position above the receptacle
        for pos in position_above:
            if try_place_at_point(controller, pos, inventory_object, receptacle):
                return


    # If the object is an inherited one, try to put the object in the parent
    parent_receptacle = get_inherited_parent_object(controller, receptacle)

    if parent_receptacle:
        try:
            excluded_receptacle_ids.add(get_object_id(receptacle))

            put_object(controller, parent_receptacle, excluded_receptacle_ids)

            if right_receptacle_or_pickup(controller, inventory_object, parent_receptacle):
                return
        except:
            pass
        finally:
            excluded_receptacle_ids.add(get_object_id(parent_receptacle))

    # If the object as an inherited object, try to put the object in the inherited
    inherited_receptacles = get_inherited_objects(controller, receptacle)

    if inherited_receptacles:

        for inh_rcpt in inherited_receptacles:
            try:
                excluded_receptacle_ids.add(get_object_id(receptacle))

                put_object(controller, inh_rcpt, excluded_receptacle_ids)

                if right_receptacle_or_pickup(controller, inventory_object, inh_rcpt):
                    return
            except:
                pass
            finally:
                excluded_receptacle_ids.add(get_object_id(inh_rcpt))

    # Try to put the object in the center of the receptacle considering axisAlignedBoundingBox
    if try_place_at_point(
            controller = controller, 
            position_above = receptacle['axisAlignedBoundingBox']['center'], 
            inventory_object=inventory_object, 
            receptacle=receptacle
        ):
        return

    # Try to put the object at one of the receptacle corner points
    for corner_point in receptacle['axisAlignedBoundingBox']['cornerPoints']:
        if try_place_at_point(
                controller = controller, 
                position_above = {
                    'x' : corner_point[0],
                    'y' : corner_point[1],
                    'z' : corner_point[2]
                }, 
                inventory_object=inventory_object, 
                receptacle=receptacle
            ):
            return

    # Receptacle is full, so another one of the same type is searched in the environment and the object is placed inside it (if possible)
    recepts = get_objects_in_scene(controller, receptacle = True, objectType = receptacle['objectType'])

    if recepts and len(recepts) > 1:
        for rec in recepts:
            if get_object_id(rec) != get_object_id(receptacle):
                reach_object(controller, rec)

                try:
                    put_object(controller, rec)

                    if right_receptacle_or_pickup(controller, inventory_object, rec):
                        return
                except:
                    pass
    

    # If all the other ways failed, trying to put the object inside by scanning all the position_aboce the receptacles
    if position_above:
        for pos in position_above:
            if try_place_at_point(
                    controller = controller, 
                    position_above = pos, 
                    inventory_object=inventory_object, 
                    receptacle=receptacle
                ):
                return

    raise ex.ReceptacleException(f"The object is not in {get_object_type(receptacle).lower()} due to simulation error")

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
    holded_object = get_object_by_id(controller, get_object_id(get_agent_holded_object(controller)))

    if not holded_object:
        raise ex.HoldingObjectsException(f"Cannot find the object in the scene")
    elif pour and (not contains_liquid(holded_object)):
        raise ex.InteractionException(f"The object '{holded_object}' is not filled with any liquid")

    degree_step = 60.0

    while degree_step < 360.0:
        resilient_execution(controller,
            action = "RotateHeldObject",
            pitch = degree_step
        )

        degree_step += 30.0

    if pour and contains_liquid(get_object_by_id(controller, get_object_id(get_agent_holded_object(controller)))):
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

    step_size = 1.0 / steps_num

    current_openness = step_size

    if is_openable(object) and get_openness(object) < 1.0:
    
        for i in range(steps_num):
            controller.step(
                action="OpenObject",
                objectId=get_object_id(object),
                openness = current_openness,
                forceAction=False
            )
    
            if not last_action_state(controller):    
                for j in range(MAX_ATTEMPTS):
                    teleport_to_free_position(controller)

                    look_at_object(controller, object)
    
                    controller.step(
                        action="OpenObject",
                        objectId=get_object_id(object),
                        openness = 1.0,
                        forceAction= True if j == (MAX_ATTEMPTS - 1) else False 
                    )
    
                    if last_action_state(controller):
                        return
                else:
                    raise ex.Ai2THORException(controller)
            
            current_openness += step_size

    elif not is_openable(object):
        raise ex.InteractionException(f"The object '{get_object_type(object)}' cannot be opened")

def close_object(controller: Controller, object: dict):
    if is_openable(object) and get_openness(object) > 0.0:

        resilient_execution(controller,
            action="CloseObject",
            objectId=get_object_id(object),
            forceAction=False
        )

    elif not is_openable(object):
        raise ex.InteractionException(f"The object '{get_object_type(object)}' cannot be closed")

def break_object(controller : Controller, object : dict):
    if not is_breakable(object):
        raise ex.ObjectException("The selected object cannot be broken")
    elif is_broken(object):
        raise ex.ObjectException("The selected object is already broken")

    resilient_execution(controller,
        action="BreakObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def cook_object(controller : Controller, object : dict[str, str]):

    if not is_cookable(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be cooked")
    elif is_cooked(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} is already cooked")

    resilient_execution(controller,
        action="CookObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def slice_object(controller: Controller, object: dict):

    get_agent_holded_object(controller)

    if is_sliceable(object) and (not is_sliced(object)):

        resilient_execution(controller,
            action="SliceObject", 
            objectId=get_object_id(object),
            forceAction=False
        )

def toggle_object_on(controller : Controller, object : dict[str, str]):
    if not is_toggleable(object):
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} cannot be toggled on")
    elif is_on(object):
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} is already toggled on")

    resilient_execution(controller,
        action = "ToggleObjectOn",
        objectId=get_object_id(object),
        forceAction = False
    )

def toggle_object_off(controller : Controller, object : dict[str, str]):
    if not is_toggleable(object):
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} cannot be toggled off")
    elif not is_on(object):
        raise ex.InteractionException(f"{get_object_type(object).capitalize()} is already toggled off")

    resilient_execution(controller,
        action = "ToggleObjectOff",
        objectId=get_object_id(object),
        forceAction = False
    )

def dirty_object(controller : Controller, object : dict[str, str]):
    if not is_dirtable(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be dirty")
    elif is_dirty(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} is already dirty")

    resilient_execution(controller,
        action="DirtyObject",
        objectId=get_object_id(object),
        forceAction = False
    )

def clean_object(controller : Controller, object : dict[str, str]):
    if not is_dirtable(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} cannot be cleaned since it cannot be dirty")
    elif not is_dirty(object):
        raise ex.InteractionException(f"{object['name'].capitalize()} is not dirty")

    resilient_execution(controller,
        action = "CleanObject",
        objectId = get_object_id(object),
        forceAction = False
    )

def fill_object_with_liquid(controller : Controller, object : dict[str, str], liquid : str):
    if not can_contain_liquid(object):
        raise ex.ObjectException("The object cannot be filled with any liquid")
    elif contains_liquid(object) and get_liquid_inside(object):
        raise ex.InteractionException(f"The object is already filled with '{get_liquid_inside(object)}'")

    resilient_execution(controller,
        action="FillObjectWithLiquid",
        objectId=get_object_id(object),
        fillLiquid=liquid,
        forceAction = False
    )

def empty_object_from_liquid(controller : Controller, object : dict[str, str]):
    if not contains_liquid(object):
        raise ex.InteractionException("The object is already empty")

    resilient_execution(controller,
        action="EmptyLiquidFromObject",
        objectId=get_object_id(object),
        forceAction = False
    )