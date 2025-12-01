"""
Descend Agent Prompt Template
Used to generate LLM prompts for the descent grasp phase
"""

DESCEND_REPLY_TEMPLATE = """
{
  "target_tcp_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "rpy": [<float roll>, <float pitch>, <float yaw>]
  },
  "gripper_command": {
    "action_type": "open_to_gap",
    "target_gap": <float gap_width>,
    "description": "Gripper opening to specified gap to accommodate brick width"
  },
  "motion_command": {
    "action_type": "descend_to_grasp",
    "description": "TCP descending to grasp position, fingertips approaching brick bottom"
  },
  "reasoning": "One sentence explaining descent strategy and key safety considerations (within 50 words)",
  "compliance_check": {
    "gripper_gap_sufficient": true,
    "tcp_z_at_grasp_height": true,
    "finger_bottom_above_ground": true,
    "xy_centered_on_brick": true,
    "orientation_maintained": true
  }
}
""".strip()


def get_descend_prompt(context, attempt_idx=0, feedback=None):
    """
    Construct prompt for Descend phase
    
    Args:
        context: Dictionary containing all necessary information
        attempt_idx: Number of attempts
        feedback: Error feedback information
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
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

    # —— Descend Phase 6-Part Structured Prompt ——
    task = f"""
## (1) Current Environment State

**Robot Arm Body Status:**
- Robot DOF: {rb.get('dof')} joints
- Joint angle limits (rad): {rb.get('joint_limits_rad')}
- Total arm length estimate (m): {rb.get('total_length_approx')}
- Current base-to-TCP distance (m): {rb.get('base_to_tcp_length_now')}

**Current TCP Status:**
- Current TCP world coordinates (m): {now.get('tcp_xyz')}
- Current TCP orientation (rad): {now.get('tcp_rpy')}

**Gripper Status:**
- Measured fingertip length (m): {gp.get('measured_tip_length')}
- Finger depth (m): {gp.get('finger_depth')}
- Current opening clearance (m): {gp.get('open_clearance')}
- Max symmetric opening angle (rad): {gp.get('max_sym_open_rad')}
- Current measured gripper gap width (m): {gp.get('measured_gap_width_now')}

**Physical Environment:**
- Gravity: {sc.get('gravity')}
- Friction coefficients: ground={sc.get('friction',{}).get('ground')}, brick={sc.get('friction',{}).get('brick')}, finger={sc.get('friction',{}).get('finger')}
- Ground height z: {cmpd.get('ground_z')}
- Ground safety margin (m): {gp.get('ground_margin', 0.003)}

**Target Brick Status:**
- Brick size LWH (m): {bk.get('size_LWH')} (length x width x height)
- Brick center coordinates (m): {bk.get('pos')}
- Brick orientation RPY (rad): {bk.get('rpy')}
- Brick yaw angle (rad): {bk.get('rpy')[2] if bk.get('rpy') else None}
- Brick top surface height z_top (m): {cmpd.get('z_top')}
- Brick bottom surface height z_bottom (m): {bk.get('pos')[2] - bk.get('size_LWH')[2]/2 if bk.get('pos') and bk.get('size_LWH') else None}

## (2) Memory Information

**Task Progress:**
- Current execution phase: Descend grasp planning
- Previous phase: ✓ Pre-grasp pose planning completed, TCP positioned above brick
- Currently processing: Gripper opening and descent to grasp position

**Completed Steps:**
- ✓ Environment initialization and robot arm calibration
- ✓ Brick detection and localization
- ✓ Pre-grasp pose planning and movement
- ✓ TCP positioned above brick at safe distance

**Pending Steps:**
- → Gripper opening to appropriate gap (current task part 1)
- → TCP descent to grasp position (current task part 2)
- → Gripper closing for grasp
- → Brick lifting
- → Movement to placement position

**Planning Constraints:**
- Brick width W (m): {bk.get('size_LWH')[1] if bk.get('size_LWH') else None}
- Ground height (m): {context.get('constraints', {}).get('ground_z')}
- Ground safety margin (m): {gp.get('ground_margin', 0.003)}
- Opening clearance (m): {gp.get('open_clearance')}
- Finger depth (m): {gp.get('finger_depth')}
- Tip length (m): {gp.get('measured_tip_length')}

**Your Task:**
Calculate the required gripper gap and TCP descent height based on the above parameters.
- Required gap formula: brick_width + open_clearance
- TCP descent height formula: max(ground_z + ground_margin, brick_bottom - finger_depth/2) + tip_length

**Error Feedback Information:**
{('Previous attempt rejected, reason: ' + feedback) if feedback else 'No historical error feedback'}

## (3) Role Definition

You are a **Descent Grasp Planning Expert Agent**, specifically responsible for planning safe descent from pre-grasp position to grasp position.

**Main Responsibilities:**
- Calculate precise gripper gap opening to accommodate brick width
- Plan TCP descent to appropriate grasp height with fingertips near but not touching ground
- Ensure safety during descent process, avoiding collisions and excessive contact

**Specific Tasks:**
1. **Gripper Opening Control**: Calculate target_gap = brick width + opening clearance
2. **TCP Descent Planning**: Calculate TCP descent z-coordinate with fingertip bottom at safe height
3. **Position Precision Maintenance**: Maintain TCP x,y coordinates aligned with brick center
4. **Pose Maintenance**: Maintain same top-down pose as pre-grasp phase

## (4) Knowledge Base

**Geometric Calculation Knowledge:**
- Gripper gap calculation: required_gap = brick_width + open_clearance
- TCP height calculation: tcp_z = desired_bottom + finger_tip_length
- Fingertip bottom safe height: desired_bottom = max(ground_z + ground_margin, brick_bottom - finger_depth/2)
- Contact avoidance principle: fingertip bottom should be above ground but near brick bottom

**Motion Constraint Knowledge:**
- Gripper opening limit: cannot exceed max_sym_open_rad
- TCP descent limit: z-coordinate cannot be lower than safe bottom height
- Collision avoidance: monitor contact forces during descent

**Grasp Preparation Knowledge:**
- Optimal grasp position: fingertips surrounding brick middle section for stable grasp
- Safety margin: maintain appropriate clearance to avoid accidental contact
- Pose stability: maintain gripper direction unchanged during descent

**Control System Parameters:**
- Gripper control precision: {vf.get('gripper_tolerance', 0.001)}m
- TCP position precision: {vf.get('tcp_pos_tolerance', 0.003)}m
- Joint velocity limit: {ctrl.get('joint_velocity_limit')}

## (5) Thinking Chain

**Step 1: Gripper Gap Calculation**
- Identify brick width: W = {bk.get('size_LWH')[1] if bk.get('size_LWH') else 'N/A'} m
- Calculate required gap: target_gap = W + open_clearance
- Verify gap reachability: check if within gripper opening range

**Step 2: Descent Height Calculation**
- Determine ground safe height: ground_safe = ground_z + ground_margin
- Calculate brick bottom: brick_bottom = brick_center_z - height/2
- Determine fingertip bottom target: desired_bottom = max(ground_safe, brick_bottom - finger_depth/2)
- Calculate TCP target height: tcp_z = desired_bottom + tip_length

**Step 3: Position and Pose Maintenance**
- X,Y coordinates: maintain alignment with brick center
- RPY pose: maintain current top-down direction
- Verify kinematic reachability

**Step 4: Safety Check**
- Confirm TCP descent path has no collisions
- Verify gripper opening causes no accidental contact
- Check all parameters are within reasonable ranges

## (6) Output Format

**Output Description:**
Your output contains two control commands:
1. Gripper opening control - target_gap will be used for gripper.open_to_gap()
2. TCP descent control - target_tcp_pose will be used for robotic arm motion control

**Strict Constraints:**
1. target_gap = brick width + opening clearance (precise calculation)
2. tcp_z = desired_bottom + tip_length (safe descent)
3. tcp_x = brick center x, tcp_y = brick center y (maintain alignment)
4. rpy maintains top-down pose (consistent with pre-grasp phase)
5. **Only JSON format output allowed, no markdown code blocks or additional text**

**Output Template:**
{DESCEND_REPLY_TEMPLATE}

**Important Reminders:**
- target_gap units: meters (m), must be positive and within gripper capability range
- Coordinate units: meters (m), angle units: radians (rad)
- All items in compliance_check must be true
- reasoning field should explain core considerations of descent strategy
""".strip()

    system = (
        "You are a robotic grasp descent planning expert. "
        "Your output controls gripper opening and TCP descent for safe grasping. "
        "Always output valid JSON only, with precise numbers in meters/radians."
    )
    
    return system, task
