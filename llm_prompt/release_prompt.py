"""
Release Agent Prompt Template
Used to generate LLM prompts for the release phase
"""

RELEASE_REPLY_TEMPLATE = '''
{
  "extra_clearance": 0.015,
  "tolerance": 0.001, 
  "increment_step": 0.005,
  "max_retries": 8,
  "reasoning": "Intelligent release planning: select optimal release strategy based on contact status"
}
'''


def get_release_prompt(current_gap, required_gap, finger_contacts, max_opening):
    """
    Construct prompt for Release phase
    
    Args:
        current_gap: Current gripper gap
        required_gap: Required gap
        finger_contacts: Number of finger contacts
        max_opening: Maximum opening
        
    Returns:
        Complete prompt string
    """
    prompt = f'''
You are a robotic gripper release control expert. After completing precise brick placement, you need to intelligently release the brick.

Please plan the release strategy based on the following information:
- Current gripper gap: {current_gap:.6f}m
- Required brick clearance: {required_gap:.6f}m  
- Contact status: {finger_contacts}
- Maximum opening: {max_opening:.6f}m

Please output release planning results in JSON format:
{{
  "extra_clearance": 0.015,
  "tolerance": 0.001, 
  "increment_step": 0.005,
  "max_retries": 8,
  "reasoning": "Intelligent release planning: select optimal release strategy based on contact status"
}}
'''
    return prompt
