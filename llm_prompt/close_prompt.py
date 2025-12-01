"""
Close Agent Prompt Template
Used to generate LLM prompts for the gripper closing phase
"""

CLOSE_REPLY_TEMPLATE = '''
{
  "gripper_command": {
    "action_type": "close_grasp",
    "description": "Gripper closing to grasp brick"
  },
  "tcp_adjustment": {
    "enabled": true,
    "position_offset": [0.001, 0.0, -0.002],
    "description": "TCP position fine-tuning to optimize grasping effect"
  },
  "attachment_strategy": {
    "use_contact_assist": true,
    "contact_threshold": 5.0,
    "description": "Contact-assisted grasping strategy"
  },
  "reasoning": "Intelligent gripper closing planning: ensure stable grasping",
  "compliance_check": {
    "gripper_closed_successfully": true,
    "brick_securely_grasped": true,
    "tcp_centered_on_brick": true,
    "attachment_confirmed": true,
    "no_excessive_force": true
  }
}
'''


def get_close_prompt(context, attempt_idx=0, feedback=""):
    """
    Construct prompt for Close phase
    
    Args:
        context: Dictionary containing all necessary information
        attempt_idx: Number of attempts
        feedback: Error feedback information
        
    Returns:
        (system_prompt, user_prompt) tuple
    """
    tcp_xyz = context.get("now", {}).get("tcp_xyz", [0, 0, 0])
    brick_pos = context.get("brick", {}).get("pos", [0, 0, 0])
    brick_size = context.get("brick", {}).get("size_LWH", [0.2, 0.095, 0.06])
    
    current_gap = context.get("current_gap", 0.1)
    finger_contacts = context.get("finger_contacts", 0)
    
    gravity = context.get("scene", {}).get("gravity", -9.81)
    friction = context.get("scene", {}).get("friction", {})
    
    task = f"""
You are a robotic gripper closing control expert. After positioning the gripper, you need to intelligently close it to securely grasp the brick.

## Current Status:
- TCP Position: {tcp_xyz}
- Brick Position: {brick_pos}
- Brick Size (LWH): {brick_size}
- Current Gripper Gap: {current_gap:.6f}m
- Finger Contact Count: {finger_contacts}
- Gravity: {gravity}
- Friction Coefficients: {friction}

## Your Task:
Plan the gripper closing strategy to ensure:
1. Gripper closes symmetrically to center-grasp the brick
2. Sufficient contact force without damaging the brick
3. Optimal TCP position adjustment if needed
4. Use contact-assisted attachment strategy when appropriate

## Error Feedback:
{feedback if feedback else 'No previous errors'}

## Output Requirements:
Please output your gripper closing plan in JSON format following the template.
Ensure all compliance checks are true.

**Output Template:**
{CLOSE_REPLY_TEMPLATE}

**Important:**
- Output valid JSON only, no markdown code blocks
- Include your reasoning for the closing strategy
- All numeric values should be precise floats
""".strip()

    system = (
        "You are a robotic gripper control expert. "
        "Your output controls gripper closing for secure brick grasping. "
        "Always output valid JSON only."
    )
    
    return system, task
