"""
Single Agent Prompt Template
Used to generate complete task planning prompts for a single LLM call
This is the single-agent mode for comparison experiments, planning all 6 phases at once
"""

SINGLE_AGENT_REPLY_TEMPLATE = """
{
  "pre_grasp": {
    "target_tcp_pose": {
      "xyz": [<float x>, <float y>, <float z>],
      "rpy": [<float roll>, <float pitch>, <float yaw>]
    },
    "control_strategy": {
      "action_type": "move_to_pose",
      "approach_clearance": <float>,
      "orientation_control": "topdown_aligned"
    },
    "reasoning": "Pre-grasp positioning strategy and key considerations"
  },
  "descend": {
    "target_tcp_pose": {
      "xyz": [<float x>, <float y>, <float z>],
      "rpy": [<float roll>, <float pitch>, <float yaw>]
    },
    "gripper_control": {
      "action_type": "open_to_gap",
      "target_gap": <float>,
      "timing": "before_descent"
    },
    "motion_strategy": {
      "descent_type": "controlled_approach",
      "safety_margins": <float>
    },
    "reasoning": "Descent and gripper opening coordination strategy"
  },
  "close": {
    "gripper_command": {
      "action_type": "close_grasp",
      "force_control": true
    },
    "tcp_adjustment": {
      "enabled": <true/false>,
      "position_offset": [<float dx>, <float dy>, <float dz>]
    },
    "attachment_strategy": {
      "use_contact_assist": <true/false>,
      "contact_threshold": <float>,
      "verification_method": "force_feedback"
    },
    "reasoning": "Gripper closing and attachment strategy"
  },
  "lift": {
    "target_tcp_pose": {
      "xyz": [<float x>, <float y>, <float z>],
      "rpy": [<float roll>, <float pitch>, <float yaw>]
    },
    "lift_strategy": {
      "action_type": "vertical_lift",
      "lift_height": <float>,
      "stability_control": true
    },
    "safety_monitoring": {
      "attachment_verification": true,
      "force_limits": <float>,
      "emergency_procedures": ["re_grasp", "abort"]
    },
    "reasoning": "Safe lifting strategy with stability control"
  },
  "place": {
    "approach_phase": {
      "target_pose": {
        "xyz": [<float x>, <float y>, <float z>],
        "rpy": [<float roll>, <float pitch>, <float yaw>]
      },
      "approach_height": <float>,
      "strategy": "overhead_alignment"
    },
    "descent_phase": {
      "action_type": "gradual_descent_to_support",
      "start_height": <float>,
      "step_size": <float>,
      "max_descent_range": <float>,
      "orientation_control": {
        "target_rpy": [<float r>, <float p>, <float y>],
        "roll_tolerance": <float>,
        "pitch_tolerance": <float>,
        "yaw_tolerance": <float>,
        "strict_parallel": true
      }
    },
    "contact_detection": {
      "support_ids": [<list of support IDs>],
      "detection_method": "collision_monitoring",
      "force_threshold": <float>
    },
    "reasoning": "Complete placement strategy with precise orientation control"
  },
  "release": {
    "release_control": {
      "action_type": "gradual_release",
      "target_gap": <float>,
      "increment_step": <float>,
      "release_sequence": "progressive_opening"
    },
    "contact_monitoring": {
      "detection_method": "force_and_contact",
      "separation_verification": true,
      "anti_stick_measures": true
    },
    "retreat_strategy": {
      "lift_distance": <float>,
      "withdrawal_path": "vertical_then_horizontal",
      "safe_position": [<float x>, <float y>, <float z>]
    },
    "reasoning": "Safe release with anti-stick and verification"
  },
  "global_strategy": {
    "execution_sequence": ["pre_grasp", "descend", "close", "lift", "place", "release"],
    "coordination_principles": "Maintain pose consistency across phases",
    "safety_priorities": ["collision_avoidance", "stable_grasp", "precise_placement"],
    "fallback_mechanisms": ["geometric_planning", "classical_control", "emergency_stop"]
  },
  "compliance_verification": {
    "all_phases_planned": true,
    "pose_consistency": true,
    "parameter_validity": true,
    "safety_considerations": true,
    "execution_feasibility": true
  }
}
""".strip()


def get_single_agent_prompt(context, attempt_idx=0, feedback=None):
    """
    Construct complete task planning prompt for single LLM call
    
    Args:
        context: Dictionary containing all necessary information
        attempt_idx: Number of attempts
        feedback: Error feedback information
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    # Parse context information
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
    brick_pos = bk.get("pos", [0, 0, 0])
    brick_size = bk.get("size_LWH", [0.2, 0.095, 0.06])
    brick_yaw = bk.get("rpy", [0, 0, 0])[2] if bk.get("rpy") else 0
    
    goal_xy = gl.get("target_xy", [0, 0])
    goal_z_top = gl.get("target_z_top", 0)
    goal_yaw = gl.get("target_yaw", 0)
    support_ids = gl.get("support_ids", [0])
    
    approach = cmpd.get("approach", 0.08)
    tip_length = gp.get("measured_tip_length") or gp.get("tip_length_guess", 0.012)
    
    # Calculate key heights and positions (using GraspModule formulas)
    z_top = brick_pos[2] + brick_size[2]/2
    # GraspModule formula: min_pre = z_top + approach + 0.02, then TCP = min_pre + tip
    pre_grasp_z = z_top + approach + 0.02 + tip_length
    approach_z = goal_z_top + approach + tip_length
    descent_start_z = goal_z_top + 0.010
    
    # —— Single Agent Unified Planning Prompt ——
    system_prompt = """
You are a professional robotic task planning expert, responsible for unified planning of complete brick manipulation tasks.
You need to plan all phases of the task at once, ensuring coordination and consistency between phases.
Output must be valid JSON format, do not use markdown code blocks.
""".strip()

    user_prompt = f"""
## Task Description
Plan a complete robotic brick grasping and placement task, containing 6 main phases:
1. Pre-grasp positioning (pre_grasp)
2. Descent for grasp (descend) 
3. Gripper closing (close)
4. Brick lifting (lift)
5. Brick placement (place)
6. Gripper release (release)

## Environment State Information

**Robot Specifications:**
- DOF: {rb.get('dof', 7)} joints
- Joint limits: {rb.get('joint_limits_rad', [[-2.97, 2.97]]*7)}
- Total arm length: {rb.get('total_length_approx', 1.261)}m
- Current TCP position: {now.get('tcp_xyz', 'N/A')}
- Current TCP pose: {now.get('tcp_rpy', 'N/A')}

**Gripper Parameters:**
- Fingertip length: {tip_length:.6f}m
- Fingertip depth: {gp.get('finger_depth', 0.035):.6f}m
- Opening clearance: {gp.get('open_clearance', 0.010):.6f}m
- Max opening: {gp.get('max_sym_open_rad', 0.523):.6f}rad

**Target Brick:**
- Size LWH: {brick_size[0]:.3f} x {brick_size[1]:.3f} x {brick_size[2]:.3f}m
- Current position: ({brick_pos[0]:.6f}, {brick_pos[1]:.6f}, {brick_pos[2]:.6f})m
- Current yaw: {brick_yaw:.6f}rad
- Brick top height: {z_top:.6f}m

**Target Placement Position:**
- Target XY: ({goal_xy[0]:.6f}, {goal_xy[1]:.6f})m
- Target top height: {goal_z_top:.6f}m
- Target yaw: {goal_yaw:.6f}rad
- Support surface IDs: {support_ids}

**Calculation References:**
- Pre-grasp TCP height: {pre_grasp_z:.6f}m
- Placement approach height: {approach_z:.6f}m
- Descent start height: {descent_start_z:.6f}m
- Approach clearance: {approach:.6f}m

**Physical Environment:**
- Gravity: {sc.get('gravity', -9.81)}
- Friction coefficients: ground={sc.get('friction',{}).get('ground', 0.5)}, brick={sc.get('friction',{}).get('brick', 0.7)}
- Ground height: {cmpd.get('ground_z', 0.0)}m

## Key Constraints

**Position Precision Requirements:**
- XY positioning error ≤ {vf.get('tol_xy', 0.006)}m
- Z positioning error ≤ {vf.get('tol_z', 0.003)}m  
- Angle error ≤ {vf.get('tol_angle', 0.05)}rad

**Pose Control Requirements:**
- Brick bottom must be strictly parallel to support surface
- Roll control precision: ±0.002rad
- Pitch control precision: ±0.002rad
- Yaw control precision: ±0.005rad

**Safety Constraints:**
- Joint angles within limits
- Avoid collisions
- Stable grasp
- Smooth placement

## Planning Requirements

**Phase Coordination:**
- Maintain TCP pose consistency
- Ensure reasonable gripper state transitions
- Smooth and continuous height changes
- Stable orientation angles

**Specific Calculation Requirements:**
1. **pre_grasp**: TCP located directly above brick, height={pre_grasp_z:.6f}m
2. **descend**: TCP descends, gripper opening={brick_size[1] + gp.get('open_clearance', 0.010):.6f}m
3. **close**: Gripper closes, including contact assist strategy
4. **lift**: TCP lifts to safe height, verify grasp
5. **place**: 
   - approach phase: TCP to ({goal_xy[0]:.6f}, {goal_xy[1]:.6f}, {approach_z:.6f})
   - descent phase: descend from {descent_start_z:.6f}m to contact support surface
6. **release**: Progressive opening release, anti-stick strategy

**Error Feedback:**
{('Reason for last failure: ' + feedback) if feedback else 'No historical error feedback'}

## Output Requirements

**Output Format:** Strict JSON format, containing complete planning for all 6 phases
**Numerical Precision:** Coordinate units in meters (m), angle units in radians (rad), precise to 6 decimal places
**Consistency Check:** Parameters across phases must be coordinated
**Feasibility:** All parameters must be within physical constraints

Please output the complete unified planning result following this template:

{SINGLE_AGENT_REPLY_TEMPLATE}
""".strip()

    return system_prompt, user_prompt
