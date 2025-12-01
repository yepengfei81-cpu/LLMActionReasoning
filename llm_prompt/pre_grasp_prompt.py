"""
Pre-Grasp Agent Prompt Template
Used to generate LLM prompts for the pre-grasp phase
"""

PRE_GRASP_REPLY_TEMPLATE = """
{
  "target_tcp_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "rpy": [<float roll>, <float pitch>, <float yaw>]
  },
  "control_command": {
    "action_type": "move_to_pose",
    "description": "Robot arm TCP movement to pre-grasp pose control command"
  },
  "reasoning": "One sentence explaining your decision key points and critical considerations (within 50 words)",
  "compliance_check": {
    "xy_aligned_with_brick_center": true,
    "z_at_safe_approach_height": true,
    "topdown_orientation_correct": true,
    "x_axis_aligned_with_brick_width": true,
    "within_joint_limits": true
  }
}
""".strip()


def get_pre_grasp_prompt(context, attempt_idx=0, feedback=None):
    """
    Construct prompt for Pre-Grasp phase
    
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

    # —— Pre-Grasp Planning Agent Prompt (6-Part Structure) ——
    task = f"""
## (1) Current Environment State

**Robot Arm Body Status:**
- Robot DOF: {rb.get('dof')} joints
- Joint names: {rb.get('arm_joint_names')}
- Joint types: {rb.get('arm_joint_types')}
- Joint angle limits (rad): {rb.get('joint_limits_rad')}
- Arm segment lengths estimate (m): {rb.get('segment_lengths')}
- Total length estimate (m): {rb.get('total_length_approx')}
- Current base-to-TCP distance (m): {rb.get('base_to_tcp_length_now')}
- TCP calibrated: {rb.get('tcp_calibrated')}

**Current TCP Status:**
- Current TCP world coordinates (m): {now.get('tcp_xyz')}
- Current TCP orientation (rad): {now.get('tcp_rpy')}

**Gripper Status:**
- Fingertip length estimate (m): {gp.get('tip_length_guess')}
- Measured fingertip length (m): {gp.get('measured_tip_length')}
- Finger depth (m): {gp.get('finger_depth')}
- Opening clearance (m): {gp.get('open_clearance')}
- Width padding (m): {gp.get('width_pad')}
- Max symmetric opening angle (rad): {gp.get('max_sym_open_rad')}
- Current measured gripper gap width (m): {gp.get('measured_gap_width_now')}

**Physical Environment:**
- Gravity: {sc.get('gravity')}
- Time step: {sc.get('time_step')}
- Friction coefficients: ground={sc.get('friction',{}).get('ground')}, brick={sc.get('friction',{}).get('brick')}, finger={sc.get('friction',{}).get('finger')}
- Ground height z: {cmpd.get('ground_z')}

**Control System Parameters:**
- Joint velocity limit: {ctrl.get('joint_velocity_limit')}
- Joint acceleration limit: {ctrl.get('joint_acc_limit')}
- IK max iterations: {ctrl.get('ik_max_iters')}
- IK position tolerance: {ctrl.get('ik_pos_tolerance')}
- IK orientation tolerance: {ctrl.get('ik_ori_tolerance')}

**Target Brick Status:**
- Brick size LWH (m): {bk.get('size_LWH')} (length x width x height)
- Brick mass (kg): {bk.get('mass')}
- Brick center coordinates (m): {bk.get('pos')}
- Brick orientation RPY (rad): {bk.get('rpy')}
- Brick yaw angle (rad): {bk.get('rpy')[2] if bk.get('rpy') else None}
- Brick top surface height z_top (m): {cmpd.get('z_top')}

## (2) Memory Information

**Task Progress:**
- Current execution phase: Pre-Grasp planning
- Task type: Three-brick intelligent wall construction
- Currently processing: Single brick pre-grasp pose planning

**Completed Steps:**
- ✓ Environment initialization and robot arm calibration
- ✓ Brick detection and localization
- ✓ Geometric planning reference pose calculation

**Pending Steps:**
- → Pre-grasp pose planning (current task)
- → Grasp planning
- → Motion execution
- → Placement planning
- → Task verification

**Planning Constraints Information:**
- Safety clearance requirement approach_clearance (m): {context.get('constraints', {}).get('approach_clearance')}
- Ground reference height (m): {context.get('constraints', {}).get('ground_z')}
- Description: {context.get('constraints', {}).get('description', 'Use these constraints to compute target positions')}

**Error Feedback Information:**
{('Previous attempt rejected, reason: ' + feedback) if feedback else 'No historical error feedback'}

## (3) Role Definition

You are a **Pre-Grasp Pose Planning Expert Agent**, specifically responsible for planning safe pre-grasp poses for robotic arm grasping tasks.

**Main Responsibilities:**
- Analyze current brick position, orientation and geometric characteristics
- Consider robotic arm kinematic constraints and gripper geometric features
- Plan a safe pre-grasp pose located directly above the brick
- Ensure feasibility and safety of subsequent grasping actions

**Specific Task:**
Calculate the pre-grasp pose that the robotic arm TCP should move to, ensuring:
1. TCP is located directly above the brick center, maintaining safe clearance
2. Gripper orientation is top-down for convenient subsequent descending grasp
3. Gripper x-axis aligns with brick width direction, ensuring grasping stability

## (4) Knowledge Base

**Coordinate System Knowledge:**
- World coordinate system: X-forward, Y-left, Z-up
- Brick coordinate system: length direction as main axis (L=0.20m), width direction as side axis (W=0.095m), height H=0.06m
- TCP coordinate system: x-axis points towards gripper opening direction, z-axis along gripper centerline

**Geometric Relationship Knowledge:**
- Brick yaw angle: rotation angle of brick length direction relative to world X-axis
- Brick width direction: perpendicular to length direction, i.e., yaw angle + π/2 direction
- Pre-grasp height: brick top surface + safety clearance + fingertip length compensation

**Grasping Pose Knowledge:**
- Top-down pose: TCP's ez axis points towards world -Z direction
- Standard top-down RPY: roll=π, pitch=0, yaw=adjustable
- Gripper alignment principle: TCP's x-axis should align with brick width direction for improved grasping stability

**Kinematic Constraints:**
- Joint angles must be within limit ranges: {rb.get('joint_limits_rad')}
- Velocity and acceleration limits: velocity={ctrl.get('joint_velocity_limit')}, acceleration={ctrl.get('joint_acc_limit')}
- IK solution tolerance: position={ctrl.get('ik_pos_tolerance')}, orientation={ctrl.get('ik_ori_tolerance')}

## (5) Thinking Chain

**Step 1: Analyze Brick Geometry**
- Identify brick center position: (x,y,z) = {bk.get('pos')}
- Determine brick main axis direction: yaw = {bk.get('rpy')[2] if bk.get('rpy') else None} rad
- Calculate brick width direction: width direction angle = yaw + π/2

**Step 2: Calculate Pre-Grasp Position**
- X coordinate: same as brick center X
- Y coordinate: same as brick center Y
- Z coordinate formula (CRITICAL - must match GraspModule calculation):
  - Step 1: Calculate brick top surface: z_top = brick_z + (brick_height / 2)
  - Step 2: Add safety clearances: min_pre = z_top + approach_clearance
  - Step 3: Compensate for fingertip: TCP_z = min_pre + tip_length
  - Final formula: TCP_z = brick_z + (brick_height/2) + approach_clearance + tip_length
  - This ensures fingertip has: approach_clearance above brick top

**Step 3: Calculate Pre-Grasp Orientation**
- Use standard top-down pose: roll=π, pitch=0
- Yaw angle aligns with brick width direction: yaw = brick_yaw + π/2
- Verify pose satisfies kinematic constraints

**Step 4: Output Verification**
- Check if xyz satisfies position constraints
- Check if rpy satisfies orientation constraints
- Confirm compliance with all planning requirements

## (6) Output Format

**Output Description:**
Your output will be directly used as the target pose for the robotic arm control system and must be precise and accurate.

**Strict Constraints:**
1. xyz.x = brick center x, xyz.y = brick center y (error ≤ 1e-6)
2. xyz.z = brick_z + (brick_height/2) + approach_clearance + tip_length (calculate precisely, must match GraspModule)
3. rpy uses standard top-down: roll=π, pitch=0
4. yaw = brick_yaw + π/2 (TCP x-axis aligns with brick width direction)
5. **Only JSON format output allowed, no markdown code blocks or additional text**

**Calculation Example:**
Given:
- Brick position: (0.5, 0.3, 0.03) m
- Brick size LWH: (0.2, 0.095, 0.06) m  
- Brick yaw: 0.785 rad (45°)
- Approach clearance: 0.08 m
- Tip length: 0.012 m

Calculate (GraspModule method):
- Brick top z = brick_z + height/2 = 0.03 + 0.06/2 = 0.06 m
- Add safety: min_pre = z_top + approach = 0.06 + 0.08 = 0.14 m
- Compensate tip: TCP_z = min_pre + tip = 0.14 + 0.012 = 0.152 m
- Fingertip will be at: 0.152 - 0.012 = 0.14 m (above brick top 0.06m by 0.08m ✓)
- TCP target xyz: (0.5, 0.3, 0.152)
- TCP target rpy: (3.14159, 0.0, 2.356) where 2.356 = 0.785 + π/2

**Output Template:**
{PRE_GRASP_REPLY_TEMPLATE}

**Important Reminders:**
- Numbers must use floating-point format
- Coordinate units: meters (m), angle units: radians (rad)
- All items in compliance_check must be true
- reasoning field should explain your core decision basis in one sentence
""".strip()

    system = (
        "You are a precise and careful robotics planner and motion control expert. "
        "Your output will be directly used as robot control commands. "
        "Always output valid JSON only, with precise numbers in meters/radians."
    )
    
    return system, task
