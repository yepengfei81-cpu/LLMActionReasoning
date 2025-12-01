"""
Place Agent Prompt Template (Unified)
Used to generate LLM prompts for the placement phase (unified approach + descent strategy)
"""

PLACE_REPLY_TEMPLATE = """
{
  "approach_phase": {
    "target_pose": {
      "xyz": [<float x>, <float y>, <float z>],
      "rpy": [<float roll>, <float pitch>, <float yaw>]
    },
    "approach_height": <float height>,
    "strategy": "geometric_alignment",
    "description": "Move to safe height above placement target position"
  },
  "descent_phase": {
    "action_type": "gradual_descent_to_support",
    "start_height": <float height>,
    "step_size": <float size>,
    "max_descent_range": <float range>,
    "description": "Precise descent to support surface control parameters"
  },
  "brick_orientation_control": {
    "control_all_angles": true,
    "target_roll": 0.0,
    "target_pitch": 0.0, 
    "target_yaw": <float yaw_in_radians>,
    "roll_tolerance": 2.0,
    "pitch_tolerance": 2.0,
    "yaw_tolerance": 3.0,
    "bottom_parallel_to_ground": true,
    "orientation_priority": "all_angles_strict",
    "correction_method": "tcp_adjustment_with_full_rpy",
    "description": "Ensure brick bottom surface strictly parallel with ground complete pose control (Roll=0, Pitch=0, Yaw=target_value)"
  },
  "contact_detection": {
    "support_ids": [<list of support body IDs>],
    "detection_method": "collision_monitoring",
    "force_threshold": <float threshold>,
    "description": "Support surface contact detection configuration"
  },
  "safety_monitoring": {
    "max_force_limit": <float limit>,
    "emergency_abort": <true/false>,
    "grasp_verification": <true/false>,
    "description": "Safety monitoring measures throughout entire placement process"
  },
  "reasoning": "One sentence explaining core considerations of complete placement strategy (within 50 words)",
  "compliance_check": {
    "approach_position_valid": true,
    "descent_parameters_valid": true,
    "contact_detection_configured": true,
    "safety_measures_active": true,
    "unified_strategy_coherent": true,
    "brick_orientation_controlled": true
  }
}
""".strip()


def get_place_prompt(context, attempt_idx=0, feedback=""):
    """
    Construct unified prompt for Place phase (including approach and descent phases)
    
    Args:
        context: Dictionary containing all necessary information
        attempt_idx: Number of attempts
        feedback: Error feedback information
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    # Parse context structure
    sc = context.get("scene", {})
    ctrl = context.get("control", {})
    rb = context.get("robot_specs", {})
    gp = context.get("gripper", {})
    bk = context.get("brick", {})
    gl = context.get("goal", {})
    vf = context.get("verify", {})
    cls = context.get("classical", {})
    cmpd = context.get("computed", {})
    now = context.get("now", {})
    
    # Extract key parameters
    target_xy = gl.get("target_xy", [0, 0])
    target_z_top = gl.get("target_z_top", 0)
    target_yaw = gl.get("target_yaw", 0)
    approach = cmpd.get("approach", 0.08)
    support_ids = gl.get("support_ids", [0])
    phase = gl.get("phase", "complete_place")
    
    # Calculate key heights
    tip_length = gp.get("measured_tip_length") or gp.get("tip_length_guess", 0.012)
    approach_tcp_z = target_z_top + approach + tip_length
    descent_start_z = target_z_top + max(0.006, 0.010)

    # —— Intelligent Placement Planning Agent Prompt (6-Part Structure) ——
    task = f"""
## (1) Current Environment State

**Robot Arm Body Status:**
- Robot DOF: {rb.get('dof', 7)} joints
- Joint names: {rb.get('arm_joint_names', ['joint1-7'])}
- Joint types: {rb.get('arm_joint_types', ['revolute']*7)}
- Joint angle limits (rad): {rb.get('joint_limits_rad', [[-2.97, 2.97]]*7)}
- Arm segment lengths estimate (m): {rb.get('segment_lengths', [0.36, 0.42, 0.4, 0.081])}
- Total length estimate (m): {rb.get('total_length_approx', 1.261)}
- Current base-to-TCP distance (m): {rb.get('base_to_tcp_length_now', 'N/A')}
- TCP calibrated: {rb.get('tcp_calibrated', True)}

**Current TCP Status:**
- Current TCP world coordinates (m): {now.get('tcp_xyz', 'N/A')}
- Current TCP orientation (rad): {now.get('tcp_rpy', 'N/A')}

**Gripper Status (Key Placement Parameters):**
- Fingertip length estimate (m): {gp.get('tip_length_guess', 0.012)}
- Measured fingertip length (m): {gp.get('measured_tip_length', 'N/A')}
- Finger depth (m): {gp.get('finger_depth', 0.035)}
- Opening clearance (m): {gp.get('open_clearance', 'N/A')}
- Width padding (m): {gp.get('width_pad', 0.008)}
- Current grasp status: holding brick, ready for placement

**Physical Environment:**
- Gravity: {sc.get('gravity', -9.81)}
- Time step: {sc.get('time_step', 0.01)}
- Friction coefficients: ground={sc.get('friction',{}).get('ground', 0.5)}, brick={sc.get('friction',{}).get('brick', 0.7)}, finger={sc.get('friction',{}).get('finger', 0.8)}
- Ground height z: {cmpd.get('ground_z', 0.0)}

**Control System Parameters:**
- Joint velocity limit: {ctrl.get('joint_velocity_limit', 2.0)}
- Joint acceleration limit: {ctrl.get('joint_acc_limit', 5.0)}
- IK max iterations: {ctrl.get('ik_max_iters', 100)}
- IK position tolerance: {ctrl.get('ik_pos_tolerance', 1e-4)}
- IK orientation tolerance: {ctrl.get('ik_ori_tolerance', 1e-3)}

**Grasped Brick Status:**
- Brick size LWH (m): {bk.get('size_LWH', [0.2, 0.095, 0.06])} (length x width x height)
- Brick mass (kg): {bk.get('mass', 0.5)}
- Brick current position (m): {bk.get('pos', 'N/A')} (position before grasp)
- Brick current orientation RPY (rad): {bk.get('rpy', 'N/A')} (orientation before grasp)

**Task Goal Information:**
- Task type: Place held brick at target position
- Target position XY (m): {target_xy} (provided by high-level planner)
- Target brick top surface height Z (m): {target_z_top} (provided by high-level planner)  
- Target orientation yaw (rad): {target_yaw} (provided by high-level planner)
- Support surface ID list: {support_ids}
- Support surface description: {"ground" if 0 in support_ids else "placed brick surface"}

**Planning Constraints:**
- Approach clearance (m): {approach}
- Tip length (m): {tip_length}
- You need to calculate the TCP positions based on these parameters

## (2) Memory Information

**Task Progress:**
- Current execution phase: Intelligent placement planning (Place Planning)
- Task type: Six-brick 3-layer 2-column wall construction
- Currently processing: Complete placement strategy planning for grasped brick
- Planning mode: unified planning (approach + descent)

**Completed Steps:**
- ✓ Environment initialization and robot arm calibration
- ✓ Target brick detection and localization
- ✓ Pre-grasp pose planning and movement
- ✓ Descent approach to brick
- ✓ Gripper grasp action
- ✓ Brick lifting to safe height

**Pending Steps:**
- → Approach target placement position (approach_phase planning)
- → Precise descent to support surface (descent_phase planning)
- → Brick release and gripper retreat
- → Placement verification

**Planning Reference Information:**
- Approach clearance (m): {approach}
- Tip length (m): {tip_length}
- Use these to calculate precise TCP positions (do not use pre-calculated values)

**Error Feedback Information:**
{('Previous attempt rejected, reason: ' + feedback) if feedback else 'No historical error feedback'}

## (3) Role Definition

You are an **Intelligent Placement Planning Expert Agent**, specifically responsible for planning complete two-phase placement strategies for robotic arm placement tasks.

**Main Responsibilities:**
- Analyze geometric characteristics and support surface conditions of target placement position
- Unified planning of approach phase and descent phase
- Ensure brick pose control meets strict requirements for parallel placement
- Optimize placement path safety and precision

**Core Tasks:**
Design complete placement trajectory for robotic arm TCP, including:
1. **Approach Phase**: Move from current position to safe height above target position
2. **Descent Phase**: Precise descent from approach position until contact with support surface
3. **Pose Control**: Ensure brick bottom surface remains strictly parallel to support surface throughout

**Key Requirements:**
- Approach and descent must be coherent two-phase strategy
- Descent start height cannot be higher than approach end height
- Brick pose control precision requirements: roll≤0.002rad, pitch≤0.002rad

## (4) Knowledge Base

**Coordinate System Knowledge:**
- World coordinate system: X-forward, Y-left, Z-up
- Brick coordinate system: length direction as main axis (L=0.20m), width direction as side axis (W=0.095m), height H=0.06m
- TCP coordinate system: already grasping brick, TCP controls brick position and orientation

**Placement Strategy Knowledge:**
- Approach height = target Z + approach distance + fingertip length compensation
- Descent start height = target Z + small safety clearance (0.006-0.010m)
- Descent step size: recommended 0.004m for precise control
- Maximum descent range: 0.030m to avoid excessive descent

**Pose Control Core Principles:**
- **Parallel placement requirement**: Brick bottom surface must be strictly parallel to support surface
- **Roll control**: Brick cannot have lateral tilt, Roll=0°±0.002rad
- **Pitch control**: Brick cannot have forward/backward tilt, Pitch=0°±0.002rad
- **Yaw control**: Brick orientation matches target, Yaw=target_yaw±0.005rad
- **Tolerance design**: Strict tolerances ensure high-precision flat placement

**Contact Detection Methods:**
- Collision monitoring: Monitor contact between brick and support surface
- Force feedback: Detect contact forces through force sensors
- Support surface identification: Determine contact objects based on support_ids

**Kinematic Constraints:**
- Joint angle ranges: {rb.get('joint_limits_rad', [[-2.97, 2.97]]*7)}
- Velocity limits: {ctrl.get('joint_velocity_limit', 2.0)} rad/s
- Acceleration limits: {ctrl.get('joint_acc_limit', 5.0)} rad/s²
- IK solution precision: position={ctrl.get('ik_pos_tolerance', 1e-4)}m, orientation={ctrl.get('ik_ori_tolerance', 1e-3)}rad

## (5) Thinking Chain

**Step 1: Analyze Target Placement Area**
- Target position: ({target_xy[0]:.6f}, {target_xy[1]:.6f}, {target_z_top:.6f})
- Support surface type: {support_ids} → {"ground placement" if 0 in support_ids else "brick stacking"}
- Target yaw angle: {target_yaw:.6f} rad = {target_yaw*180/3.14159:.1f}°

**Step 2: Calculate Approach Phase Parameters**
- Approach TCP target X: target_xy[0]
- Approach TCP target Y: target_xy[1]
- Approach TCP target Z: target_z_top + approach_clearance + tip_length
- Approach pose: (0.0, 0.0, target_yaw) - brick remains horizontal
- Approach strategy: vertical overhead approach ensuring safe clearance

**Step 3: Design Descent Phase Parameters**
- Descent start height: target_z_top + small_safety_clearance (recommend 0.006-0.010m)
- Descent step size: recommend 0.004m for precise control
- Maximum descent range: 0.030m (safe range)
- Pose control: strictly maintain roll=0°, pitch=0°, yaw=target_yaw

**Step 4: Verify Consistency and Safety**
- Height consistency: descent start ({descent_start_z:.6f}) < approach end ({approach_tcp_z:.6f}) ✓
- Pose consistency: approach and descent phase pose parameters match ✓
- Kinematic feasibility: target position within workspace ✓
- Contact safety: support surface identification and detection scheme clear ✓

## (6) Output Format

**Output Description:**
Your output will be directly used as the execution strategy for robotic arm placement control system and must include complete two-phase planning.

**Strict Constraints:**
1. approach_phase.target_pose.xyz = [{target_xy[0]:.6f}, {target_xy[1]:.6f}, {approach_tcp_z:.6f}]
2. approach_phase.target_pose.rpy = [0.0, 0.0, {target_yaw:.6f}]
3. descent_phase.start_height = {descent_start_z:.6f} (must be ≤ approach height)
4. descent_phase.orientation_control.target_rpy = [0.0, 0.0, {target_yaw:.6f}]
5. Pose tolerances: roll_tolerance=0.002, pitch_tolerance=0.002, yaw_tolerance=0.005
6. **Only JSON format output allowed, no markdown code blocks or additional text**

**Output Template:**
{PLACE_REPLY_TEMPLATE}

**Important Reminders:**
- Numbers must use floating-point format, precise to 6 decimal places
- Coordinate units: meters (m), angle units: radians (rad)
- Ensure correct height relationship between approach and descent phases
- Pose control parameters directly affect placement precision, must be accurate
""".strip()

    system = (
        "You are a precise and expert robotics placement planner. "
        "Your output will be directly used as robot control commands for brick placement. "
        "Always output valid JSON only, with precise numbers in meters/radians. "
        "Focus on the two-phase placement strategy: approach then descent."
    )

    user_msg = f"Task Phase: {phase}, Target Position: ({target_xy[0]:.6f}, {target_xy[1]:.6f}, {target_z_top:.6f}), Target Yaw: {target_yaw:.6f}rad, Support Surface: {support_ids}"
    if feedback:
        user_msg += f"\nError Feedback: {feedback}"
        
    return system, task
