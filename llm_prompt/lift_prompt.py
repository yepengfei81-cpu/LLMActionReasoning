"""
Lift Agent Prompt Template
Used to generate LLM prompts for the lift phase
"""

LIFT_REPLY_TEMPLATE = """
{
  "target_tcp_pose": {
    "xyz": [<float x>, <float y>, <float z>],
    "rpy": [<float roll>, <float pitch>, <float yaw>]
  },
  "lift_strategy": {
    "action_type": "lift_with_grasp",
    "lift_height": <float height>,
    "lift_clearance": <float clearance>,
    "description": "Safely lifting brick to specified height"
  },
  "stability_control": {
    "verify_attachment": <true/false>,
    "check_stability": <true/false>,
    "force_monitoring": <true/false>,
    "description": "Stability control during lifting process"
  },
  "fallback_strategy": {
    "re_attach_on_failure": <true/false>,
    "retry_lift": <true/false>,
    "max_retries": <int count>,
    "description": "Fallback strategy when lifting fails"
  },
  "reasoning": "One sentence explaining lifting strategy and key stability considerations (within 50 words)",
  "compliance_check": {
    "lift_height_appropriate": true,
    "brick_remains_attached": true,
    "tcp_centered_on_brick": true,
    "no_collision_risk": true,
    "stability_maintained": true
  }
}
""".strip()


def get_lift_prompt(context, attempt_idx=0, feedback=None):
    """
    Construct prompt for Lift phase
    
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

    # —— Lift Phase 6-Part Structured Prompt ——
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
- Current gripper state: closed and grasping brick
- Grasp connection status: {gp.get('attachment_status', 'connected')}

**Physical Environment:**
- Gravity: {sc.get('gravity')}
- Friction coefficients: ground={sc.get('friction',{}).get('ground')}, brick={sc.get('friction',{}).get('brick')}, finger={sc.get('friction',{}).get('finger')}
- Ground height z: {cmpd.get('ground_z')}

**Target Brick Status:**
- Brick size LWH (m): {bk.get('size_LWH')} (length x width x height)
- Brick mass (kg): {bk.get('mass')}
- Brick center coordinates (m): {bk.get('pos')}
- Brick orientation RPY (rad): {bk.get('rpy')}
- Brick yaw angle (rad): {bk.get('rpy')[2] if bk.get('rpy') else None}
- Brick top surface height z_top (m): {cmpd.get('z_top')}

## (2) Memory Information

**Task Progress:**
- Current execution phase: Brick lifting (Lift) planning
- Completed phases: ✓ Pre-grasp pose planning → ✓ Descent to grasp position → ✓ Gripper closing and grasp
- Currently processing: Safe lifting of grasped brick to specified height

**Completed Steps:**
- ✓ Environment initialization and robot arm calibration
- ✓ Brick detection and localization
- ✓ Pre-grasp pose planning and movement
- ✓ Gripper opening and descent to grasp position
- ✓ Gripper closing and successful brick grasp

**Pending Steps:**
- → Safe lifting of brick to specified height (current task)
- → Movement to placement position
- → Descent to placement position
- → Brick release and retreat

**Grasp Status Record:**
- TCP position before gripper closing: {cmpd.get('pre_lift_tcp_pos', 'recorded')}
- Grasp verification result: {cmpd.get('grasp_verified', True)}
- Contact assist status: {cmpd.get('contact_assist_used', False)}

**Planning Constraints:**
- Lift clearance requirement (m): {context.get('constraints', {}).get('lift_clearance', 0.15)}
- Safety clearance (m): {context.get('constraints', {}).get('safety_clearance', 0.05)}
- Ground height (m): {context.get('constraints', {}).get('ground_z')}
- Tip length (m): {gp.get('measured_tip_length')}
- Brick height (m): {bk.get('size_LWH')[2] if bk.get('size_LWH') else None}

**Your Task:**
Calculate the target TCP lift height based on the above parameters.
- Formula: TCP_z = brick_z + (brick_height/2) + lift_clearance + tip_length
- Ensure brick bottom has sufficient clearance from ground

**Error Feedback Information:**
{('Previous attempt rejected, reason: ' + feedback) if feedback else 'No historical error feedback'}

## (3) Role Definition

You are a **Brick Lifting Control Expert Agent**, specifically responsible for safely lifting bricks to specified heights while ensuring grasp stability.

**Main Responsibilities:**
- Calculate appropriate lift height ensuring brick clears ground with sufficient safety clearance
- Plan TCP lift path maintaining brick stability without swaying
- Monitor grasp connection status ensuring brick doesn't slip during lifting
- Design failure recovery and re-grasp retry strategies

**Specific Tasks:**
1. **Height Calculation**: Calculate TCP target height = brick top + lift clearance + fingertip length
2. **Path Planning**: Maintain TCP directly above brick center for vertical lifting
3. **Stability Control**: Ensure smooth lifting process avoiding inertial impacts
4. **Connection Monitoring**: Real-time check of brick-gripper connection status

## (4) Knowledge Base

**Lifting Dynamics Knowledge:**
- Lifting acceleration should be controlled within reasonable range to avoid inertial forces causing brick slippage
- Vertical lifting is most stable, avoiding lateral components that cause swaying
- Lifting speed should be moderate: too slow affects efficiency, too fast may lose control

**Geometric Calculation Knowledge:**
- TCP target height = brick_top_z + lift_clearance + tip_length
- Maintain x,y coordinates aligned with brick center, only change z coordinate
- Lift clearance recommended 0.10-0.20m ensuring brick completely clears ground

**Stability Control Knowledge:**
- Check gripper connection status ensuring physical constraints are effective
- Monitor grip force avoiding too tight (damaging brick) or too loose (causing slippage)
- Maintain pose unchanged during lifting avoiding rotational disturbances

**Failure Recovery Knowledge:**
- Lift failure possible causes: connection break, insufficient grip force, path collision
- Reconnection strategy: return to grasp position, re-establish physical connection
- Retry mechanism: maximum 2-3 retries avoiding infinite loops

**Control System Parameters:**
- TCP position precision: {vf.get('tcp_pos_tolerance', 0.003)}m
- Lift velocity limit: {ctrl.get('lift_velocity_limit', 0.1)}m/s
- Joint velocity limit: {ctrl.get('joint_velocity_limit')}

## (5) Thinking Chain

**Step 1: Assess Current Grasp Status**
- Check gripper connection status stability
- Confirm brick position and current TCP position
- Evaluate grasp quality and stability

**Step 2: Calculate Lift Parameters**
- Determine target lift height: z_target = z_top + lift_clearance + tip_length
- Calculate lift distance: delta_z = z_target - current_tcp_z
- Verify lift height safety and reasonableness

**Step 3: Plan Lift Strategy**
- Keep x,y coordinates unchanged, only vertical lifting
- Set appropriate lift velocity and acceleration
- Configure stability monitoring and connection checking

**Step 4: Design Fallback Mechanism**
- Define lift failure judgment criteria
- Plan reconnection operation steps
- Set maximum retry count and termination conditions

## (6) Output Format

**Output Description:**
Your output will be directly used for robotic arm TCP motion control and stability monitoring:
1. target_tcp_pose - TCP target pose for lifting
2. lift_strategy - Lifting strategy and parameter configuration
3. stability_control - Stability monitoring configuration
4. fallback_strategy - Fallback strategy for failures

**Strict Constraints:**
1. tcp_z = brick_top_z + lift_clearance + tip_length (precise calculation)
2. tcp_x = brick center x, tcp_y = brick center y (maintain alignment)
3. rpy maintains same pose as during grasp (avoid rotational disturbance)
4. lift_height must be greater than 0 and less than workspace limits
5. **Only JSON format output allowed, no markdown code blocks or additional text**

**Output Template:**
{LIFT_REPLY_TEMPLATE}

**Important Reminders:**
- lift_height and lift_clearance units: meters (m), must be positive numbers
- Coordinate units: meters (m), angle units: radians (rad)
- All items in compliance_check must be true
- reasoning field should explain core considerations of lifting strategy
""".strip()

    system = (
        "You are a robotic lift control expert. "
        "Your output controls safe brick lifting with stability monitoring. "
        "Always output valid JSON only, with precise numbers in meters/radians."
    )
    
    return system, task
