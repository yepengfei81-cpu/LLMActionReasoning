PUSH_TOPPLE_REPLY_TEMPLATE = """{
  "pose_analysis": {
    "detected_pose": "flat|side|upright",
    "measured_centroid_z": <float>,
    "confidence": <float 0-1>,
    "reasoning": "<explanation of pose classification>"
  },
  "action_required": <true if not flat, false if already flat>,
  "push_plan": {
    "approach_pose": {
      "xyz": [<approach_x>, <approach_y>, <approach_z>],
      "rpy": [0.0, 0.0, <yaw_facing_brick>]
    },
    "push_start_pose": {
      "xyz": [<start_x>, <start_y>, <start_z>],
      "rpy": [0.0, 0.0, <yaw>]
    },
    "push_direction": [<dx>, <dy>, 0.0],
    "push_distance": <float in meters>,
    "push_contact_height": <float, height above ground to contact brick>,
    "push_speed": "slow|medium",
    "description": "<explanation of push strategy>"
  },
  "retreat_pose": {
    "xyz": [<retreat_x>, <retreat_y>, <retreat_z>],
    "rpy": [0.0, 0.0, <yaw>]
  },
  "expected_result": {
    "final_pose": "flat",
    "expected_height_after": <float>
  }
}"""


def get_push_topple_prompt(context: dict, attempt_idx: int = 0, feedback: str = "") -> tuple:
    """
    Generate push topple phase prompt.
    
    Args:
        context: Dictionary containing brick, gripper, constraints, and robot state
        attempt_idx: Current attempt number for retry logic
        feedback: Error feedback from previous attempts
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Extract brick information
    brick_pos = context["brick"]["pos"]
    brick_size = context["brick"]["size_LWH"]  # [L, W, H]
    L, W, H = brick_size
    measured_height = context["brick"].get("measured_height", brick_pos[2])
    mask_area = context["brick"].get("mask_area", 0)
    
    # Robot state
    tcp_xyz = context["now"]["tcp_xyz"]
    tcp_rpy = context["now"]["tcp_rpy"]
    
    # Ground and safety
    ground_z = context.get("constraints", {}).get("ground_z", 0.0)
    safety_clearance = context.get("constraints", {}).get("safety_clearance", 0.05)
    
    # Gripper info
    tip_length = context["gripper"].get("measured_tip_length") or context["gripper"].get("tip_length_guess", 0.05)
    
    task = f"""
## (1) Current Environment State

**Brick Geometry (IMPORTANT - memorize these values):**
- Brick dimensions: L={L:.4f}m (longest), W={W:.4f}m (medium), H={H:.4f}m (shortest)
- When lying flat (normal): height ≈ H = {H:.4f}m, top face area = L×W (largest)
- When on side: height ≈ W = {W:.4f}m, top face area = L×H (medium)  
- When upright: height ≈ L = {L:.4f}m, top face area = W×H (smallest)

**Detected Brick State:**
- Brick centroid position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f})m
- Measured top surface height: {measured_height:.4f}m
- Detected mask area: {mask_area} pixels

**Expected Heights for Pose Classification:**
- Flat (normal): centroid_z ≈ {H/2:.4f}m (half of H)
- On side: centroid_z ≈ {W/2:.4f}m (half of W)
- Upright: centroid_z ≈ {L/2:.4f}m (half of L)

**Robot State:**
- Current TCP position: ({tcp_xyz[0]:.4f}, {tcp_xyz[1]:.4f}, {tcp_xyz[2]:.4f})m
- Current TCP orientation (rpy): ({tcp_rpy[0]:.4f}, {tcp_rpy[1]:.4f}, {tcp_rpy[2]:.4f})rad
- Gripper tip length: {tip_length:.4f}m

**Environment:**
- Ground Z: {ground_z:.4f}m
- Safety clearance: {safety_clearance:.4f}m

## (2) Memory Information

**Task Progress:**
- Current phase: Push topple planning
- Objective: Push the standing/tilted brick to make it lie flat (L×W face on ground)
- Attempt: {attempt_idx + 1}

**Error Feedback:**
{('Previous attempt failed: ' + feedback) if feedback else 'No previous feedback'}

## (3) Role Definition

You are a **Brick Pose Correction Expert Agent**, responsible for analyzing brick orientation and planning push actions to topple standing bricks.

**Your Tasks:**
1. **Pose Analysis**: Determine if brick is "flat", "side", or "upright" based on measured height
2. **Push Direction**: Calculate which direction to push to make the brick fall onto its L×W face
3. **Approach Position**: Determine where the gripper should position before pushing
4. **Push Parameters**: Calculate push distance, height, and speed

## (4) Knowledge Base

**Pose Classification Logic:**
- Compare measured centroid height with expected values:
  - If centroid_z ≈ H/2 ({H/2:.4f}m) ± 0.02m → brick is FLAT (no action needed)
  - If centroid_z ≈ W/2 ({W/2:.4f}m) ± 0.02m → brick is on SIDE
  - If centroid_z ≈ L/2 ({L/2:.4f}m) ± 0.02m → brick is UPRIGHT

**Push Physics:**
- To topple a brick, push at approximately 2/3 of its current height
- Push perpendicular to the face you want to become the base
- For SIDE brick: push along ±Y direction (perpendicular to L axis) to rotate around L axis
- For UPRIGHT brick: push along ±X direction (perpendicular to W axis) to rotate around W axis

**Approach Strategy:**
- Approach from the side opposite to push direction
- Keep gripper at push_height = 2/3 × current_brick_height
- Maintain safe distance before initiating push

**Push Distance Calculation:**
- For SIDE: push_distance ≈ W/2 to W (to overcome center of gravity)
- For UPRIGHT: push_distance ≈ L/3 to L/2 (taller objects need less push)

## (5) Thinking Chain

**Step 1: Classify Current Pose**
- Measured centroid height: {brick_pos[2]:.4f}m
- Compare with: H/2={H/2:.4f}m, W/2={W/2:.4f}m, L/2={L/2:.4f}m
- Determine: is brick "flat", "side", or "upright"?

**Step 2: If Not Flat, Determine Push Strategy**
- For SIDE: 
  - Current ground contact face: L×H
  - Target ground contact face: L×W  
  - Rotation axis: L (along X if brick aligned with X)
  - Push direction: ±Y
  
- For UPRIGHT:
  - Current ground contact face: W×H
  - Target ground contact face: L×W
  - Rotation axis: W (along Y if brick aligned with Y)  
  - Push direction: ±X

**Step 3: Calculate Approach Position**
- approach_x = brick_x + (offset based on push direction)
- approach_y = brick_y + (offset based on push direction)
- approach_z = safe height for approach (≥ 0.15m)

**Step 4: Calculate Push Execution**
- push_start_z = ground_z + push_contact_height
- push_direction = unit vector of push
- push_distance = calculated distance to topple

## (6) Output Format

**Output JSON with the following structure:**
{PUSH_TOPPLE_REPLY_TEMPLATE}

**Constraints:**
1. All coordinates in meters
2. approach_z should be safe height (≥ 0.15m)
3. push_contact_height should be about 60-70% of current brick height
4. push_direction must be a unit vector (or close to it)
5. If brick is already flat, set action_required=false and push_plan can be null
6. **Output valid JSON only, no markdown code blocks**
""".strip()

    system = (
        "You are a robotic manipulation expert specialized in correcting brick poses. "
        "Analyze the brick orientation and plan a push action to topple standing bricks. "
        "Output valid JSON only with carefully calculated positions in meters."
    )
    
    return system, task