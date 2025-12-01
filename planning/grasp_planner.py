import math
import numpy as np
import pybullet as p
from math3d.transforms import align_topdown_to_brick, normalize_angle

def get_precise_brick_state(env, brick_id):
    """Get precise brick state (using unified interface)
    
    Args:
        env: BulletEnv environment object
        brick_id: Brick ID
        
    Returns:
        dict: Brick state containing precise position and orientation
    """
    # Use unified brick state acquisition function with AABB
    state = env.get_brick_state(brick_id=brick_id, include_aabb=True)
    
    # Print debug info
    print(f"[ImpGrasp] Brick #{brick_id} original pos: {state['pos']}")
    print(f"[ImpGrasp] Brick #{brick_id} AABB center: ({state['aabb_center'][0]:.6f}, {state['aabb_center'][1]:.6f}, {state['aabb_center'][2]:.6f})")
    
    # Convert to original format for compatibility
    return {
        "pos": state['aabb_center'],
        "rpy": state['rpy'],
        "original_pos": state['pos'],
        "aabb_min": state['aabb_min'],
        "aabb_max": state['aabb_max']
    }

def calculate_optimal_grasp_direction(brick_state, brick_size_LWH):
    """Calculate optimal grasp direction
    
    Args:
        brick_state: Brick state
        brick_size_LWH: Brick dimensions [Length, Width, Height]
        
    Returns:
        tuple: (Grasp angle, Grasp description)
    """
    L, W, H = brick_size_LWH
    byaw = brick_state["rpy"][2]
    
    # Long edge direction of the brick
    long_edge_angle = byaw
    
    # Grasp should be perpendicular to the long edge 
    optimal_grasp_angle = long_edge_angle + math.pi/2
    
    # Use unified angle normalization function
    optimal_grasp_angle = normalize_angle(optimal_grasp_angle)
    
    print(f"[ImpGrasp] Brick yaw: {math.degrees(byaw):.1f}°")
    print(f"[ImpGrasp] Long edge angle: {math.degrees(long_edge_angle):.1f}°")
    print(f"[ImpGrasp] Optimal grasp angle: {math.degrees(optimal_grasp_angle):.1f}°")
    
    return optimal_grasp_angle, "Grasp from short edge"

def plan_improved_grasp(
    env, brick_id, brick_size_LWH, goal_pose_xyzrpy, ground_z,
    tip_length_guess, finger_depth, approach_clearance, pad_clearance,
    lift_clearance, place_gap, gripper_open_distance=0.18
):
    """Improved grasp planning
    Returns:
        tuple: (List of waypoints, Auxiliary info)
    """
    # Get precise brick state
    brick_state = get_precise_brick_state(env, brick_id)
    (bx, by, bz) = brick_state["pos"]
    L, W, H = brick_size_LWH
    
    # Calculate optimal grasp direction
    optimal_grasp_yaw, grasp_desc = calculate_optimal_grasp_direction(brick_state, brick_size_LWH)
    
    # Set RPY for pick and place
    rpy_pick = [math.pi, 0, optimal_grasp_yaw]  
    
    # Maintain goal angle when placing
    gx, gy, gz, gr, gp, gyaw = goal_pose_xyzrpy
    rpy_place = [math.pi, 0, gyaw]
    
    # Calculate key heights
    z_top = bz + H/2
    
    # Pre-grasp height
    min_pre = z_top + approach_clearance + 0.02  
    
    # Grasp height
    brick_center_z = bz + H/2 
    brick_bottom_z = bz        
    
    # Target: Slightly below brick center to ensure fingers can grasp deeply
    target_grasp_z = brick_center_z - 0.015  
    
    min_safe_z = brick_bottom_z + 0.008  
    max_safe_z = brick_center_z - 0.005 
    
    if target_grasp_z < min_safe_z:
        min_grasp = min_safe_z
        print(f"[ImpGrasp] Grasp height adjusted to safe lower limit: {min_grasp:.3f}m")
    elif target_grasp_z > max_safe_z:
        min_grasp = max_safe_z
        print(f"[ImpGrasp] Grasp height adjusted to effective upper limit: {min_grasp:.3f}m")
    else:
        min_grasp = target_grasp_z
        
    print(f"[ImpGrasp] Brick bottom: {brick_bottom_z:.3f}m, Center: {brick_center_z:.3f}m, Target grasp: {min_grasp:.3f}m")
    
    # Lift height
    min_lift = z_top + lift_clearance
    
    # Convert to EE coordinates
    z_ee_pre = min_pre + tip_length_guess
    z_ee_grasp = min_grasp + tip_length_guess
    z_ee_lift = min_lift + tip_length_guess
    
    # Placement related heights
    min_pre_place = gz + approach_clearance
    min_place = gz + place_gap
    z_ee_pre_place = min_pre_place + tip_length_guess
    z_ee_place = min_place + tip_length_guess
    
    print(f"[ImpGrasp] Grasp strategy: {grasp_desc}")
    print(f"[ImpGrasp] Pre-grasp height: {z_ee_pre:.3f}m")
    print(f"[ImpGrasp] Grasp height: {z_ee_grasp:.3f}m")
    print(f"[ImpGrasp] Lift height: {z_ee_lift:.3f}m")
    

    wps = [
        # Pre-grasp 
        dict(
            name="pre_grasp", 
            pose=dict(xyz=[bx, by, z_ee_pre], rpy=rpy_pick),
            open_gripper="open",  # Ensure gripper is fully open
            wait_sec=0.05  # Simplified wait time
        ),
        
        # Descend 
        dict(
            name="descend", 
            pose=dict(xyz=[bx, by, z_ee_grasp], rpy=rpy_pick),
            open_gripper="open",
            wait_sec=0.05
        ),
        
        # Close 
        dict(
            name="close", 
            pose=dict(xyz=[bx, by, z_ee_grasp], rpy=rpy_pick),
            open_gripper="close",  # Close gripper
            wait_sec=0.30  # Increase wait time to ensure gripper is fully closed
        ),
        
        # Lift
        dict(
            name="lift", 
            pose=dict(xyz=[bx, by, z_ee_lift], rpy=rpy_pick),
            open_gripper="close",
            wait_sec=0.05
        ),
        
        # Placement
        dict(
            name="pre_place", 
            pose=dict(xyz=[gx, gy, z_ee_pre_place], rpy=rpy_place),
            open_gripper="close",
            wait_sec=0.05
        ),
        
        dict(
            name="descend_place", 
            pose=dict(xyz=[gx, gy, z_ee_place], rpy=rpy_place),
            open_gripper="close",
            wait_sec=0.05
        ),
        
        dict(
            name="open", 
            pose=dict(xyz=[gx, gy, z_ee_place], rpy=rpy_place),
            open_gripper="open",
            wait_sec=0.05
        ),
        
        dict(
            name="retreat", 
            pose=dict(xyz=[gx, gy, z_ee_pre_place], rpy=rpy_place),
            open_gripper="open",
            wait_sec=0.05
        ),
    ]
    
    # Auxiliary info
    aux = {
        "brick_id": brick_id,
        "precise_position": brick_state["pos"],
        "original_position": brick_state["original_pos"],
        "grasp_angle_deg": math.degrees(optimal_grasp_yaw),
        "grasp_description": grasp_desc,
        "z_top": z_top,
        "ground_z": ground_z,
        "H": H, "W": W, "L": L,
        "min_pre": min_pre,
        "min_grasp": min_grasp,
        "min_lift": min_lift,
        "min_pre_place": min_pre_place,
        "min_place": min_place
    }
    
    return wps, aux
