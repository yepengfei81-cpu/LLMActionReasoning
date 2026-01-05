import numpy as np

# ========== Push Distance Configuration ==========
PUSH_DISTANCE_MIN = 0.12          # Minimum push distance (meters)
PUSH_DISTANCE_RECOMMENDED = 0.15  # Recommended push distance (meters) 
PUSH_DISTANCE_MAX = 0.20          # Maximum push distance (meters)

# ========== Approach Distance Configuration - Prevent Knocking Over Brick ==========
APPROACH_OFFSET = 0.12            # Approach position offset from brick center (meters)
PUSH_START_OFFSET = 0.06          # Push start position offset from brick center (meters)
# Note: Larger PUSH_START_OFFSET means starting push further from the brick
# ================================================

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
      "rpy": [3.14159, 0.0, 0.0]
    },
    "push_start_pose": {
      "xyz": [<start_x>, <start_y>, <start_z>],
      "rpy": [3.14159, 0.0, 0.0]
    },
    "push_direction": [<dx>, <dy>, 0.0],
    "push_distance": <float in meters>,
    "push_contact_height": <float, height above ground to contact brick>,
    "push_speed": "slow|medium",
    "description": "<explanation of push strategy>"
  },
  "retreat_pose": {
    "xyz": [<retreat_x>, <retreat_y>, <retreat_z>],
    "rpy": [3.14159, 0.0, 0.0]
  },
  "expected_result": {
    "final_pose": "flat",
    "expected_height_after": <float>
  }
}"""


def get_push_topple_prompt(context: dict, attempt_idx: int = 0, feedback: str = "") -> tuple:
    """
    Generate push topple phase prompt.
    """
    # Extract brick information
    brick_pos = context["brick"]["pos"]
    brick_size = context["brick"]["size_LWH"]  # [L, W, H]
    L, W, H = brick_size
    measured_height = context["brick"].get("measured_height", brick_pos[2])
    mask_area = context["brick"].get("mask_area", 0)
    
    # Get estimated yaw angle from SAM3
    estimated_yaw = context["brick"].get("estimated_yaw", 0.0)
    estimated_yaw_deg = np.degrees(estimated_yaw)
    
    # Long edge direction detected by SAM3
    long_axis_x = np.cos(estimated_yaw)
    long_axis_y = np.sin(estimated_yaw)
    
    # CRITICAL: Push direction should be PERPENDICULAR to the detected long edge
    # Perpendicular direction = rotate 90 degrees = (-sin(yaw), cos(yaw), 0)
    # This ensures proper toppling instead of sliding along the long edge
    push_dir_x = -np.sin(estimated_yaw)  # Perpendicular to long edge
    push_dir_y = np.cos(estimated_yaw)   # Perpendicular to long edge
    
    # Robot state
    tcp_xyz = context["now"]["tcp_xyz"]
    tcp_rpy = context["now"]["tcp_rpy"]
    
    # Ground and safety
    ground_z = context.get("constraints", {}).get("ground_z", 0.0)
    safety_clearance = context.get("constraints", {}).get("safety_clearance", 0.05)
    
    # Gripper info
    tip_length = context["gripper"].get("measured_tip_length") or context["gripper"].get("tip_length_guess", 0.05)
    
    # Calculate preset positions (for LLM reference)
    # Approach position: on the opposite side of push direction, at APPROACH_OFFSET distance
    approach_x = brick_pos[0] - push_dir_x * APPROACH_OFFSET
    approach_y = brick_pos[1] - push_dir_y * APPROACH_OFFSET
    
    # Push start position: on the opposite side of push direction, at PUSH_START_OFFSET distance
    # This position should be just outside the brick edge
    push_start_x = brick_pos[0] - push_dir_x * PUSH_START_OFFSET
    push_start_y = brick_pos[1] - push_dir_y * PUSH_START_OFFSET
    
    task = f"""
## (1) Current Environment State

### Brick Geometry (IMPORTANT - memorize these values)
- Brick dimensions: L={L:.4f}m (longest), W={W:.4f}m (medium), H={H:.4f}m (shortest)
- When lying flat (normal): centroid height ≈ H/2 = {H/2:.4f}m
- When on side: centroid height ≈ W/2 = {W/2:.4f}m
- When upright: centroid height ≈ L/2 = {L/2:.4f}m

### Detected Brick State
- Brick centroid position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f})m
- Measured centroid height: {brick_pos[2]:.4f}m
- Detected mask area: {mask_area} pixels

### CRITICAL: Brick Orientation from SAM3
- Estimated yaw angle: {estimated_yaw:.4f} rad ({estimated_yaw_deg:.1f}°)
- Detected long edge direction: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
- **Push direction (perpendicular to long edge):** ({push_dir_x:.4f}, {push_dir_y:.4f}, 0)

### Expected Heights for Pose Classification
- Flat (normal): centroid_z ≈ {H/2:.4f}m (half of H={H:.4f}m)
- On side: centroid_z ≈ {W/2:.4f}m (half of W={W/2:.4f}m)
- Upright: centroid_z ≈ {L/2:.4f}m (half of L={L:.4f}m)

### Robot State
- Current TCP position: ({tcp_xyz[0]:.4f}, {tcp_xyz[1]:.4f}, {tcp_xyz[2]:.4f})m
- Gripper tip length: {tip_length:.4f}m

### Environment
- Ground Z: {ground_z:.4f}m
- Safety clearance: {safety_clearance:.4f}m

## (2) Memory Information

### Task Progress
- Current phase: Push topple planning
- Objective: Push the standing/tilted brick to make it lie flat (L×W face on ground)
- Attempt: {attempt_idx + 1}

### Error Feedback
{('Previous attempt failed: ' + feedback) if feedback else 'No previous feedback'}

## (3) Role Definition

You are a **Brick Pose Correction Expert Agent**. Your task is to:
1. Determine if the brick needs correction based on its height
2. Plan a push action **perpendicular to the detected long edge** to topple the brick

## (4) Knowledge Base - Push Direction Calculation

### KEY INSIGHT: Push Perpendicular to the Long Edge

To topple a standing brick, you must push it **perpendicular to its long edge**.
- Long edge direction from SAM3: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
- Push direction = rotate 90°: ({push_dir_x:.4f}, {push_dir_y:.4f}, 0)

**Why perpendicular?**
- Pushing along the long edge just "scrapes" the side face - the brick won't fall
- Pushing perpendicular applies torque around the bottom edge, causing rotation and toppling

```
    Long edge direction: →
    
    ┌──────────────┐      Push ↓ (perpendicular)
    │              │            ↓
    │    BRICK     │      ──────────────
    │              │      Brick falls flat!
    └──────────────┘
```

### IMPORTANT: Safe Distance from Brick

To avoid accidentally knocking the brick before pushing:
- Approach offset: {APPROACH_OFFSET}m (distance from brick center when approaching)
- Push start offset: {PUSH_START_OFFSET}m (distance from brick center when starting push)

For UPRIGHT brick (height ≈ L/2 = {L/2:.4f}m), the brick is very unstable!
- Keep larger distance to avoid accidental contact
- Push start should be at least {PUSH_START_OFFSET}m away from brick center

### Approach Position Calculation
To push in direction ({push_dir_x:.4f}, {push_dir_y:.4f}, 0), approach from the opposite side:
- approach_x = brick_x - push_dir_x × {APPROACH_OFFSET} = {approach_x:.4f}
- approach_y = brick_y - push_dir_y × {APPROACH_OFFSET} = {approach_y:.4f}
- approach_z = 0.15m (safe height)

### Push Start Position (Keep Safe Distance)
- push_start_x = brick_x - push_dir_x × {PUSH_START_OFFSET} = {push_start_x:.4f}
- push_start_y = brick_y - push_dir_y × {PUSH_START_OFFSET} = {push_start_y:.4f}
- push_start_z = ground_z + push_contact_height

### Push Contact Height
- push_contact_height ≈ 60-70% of current brick height above ground
- For SIDE: contact at ~{0.65 * W:.4f}m
- For UPRIGHT: contact at ~{0.65 * L:.4f}m

### Push Distance (Must Be Sufficient to Topple)
- For SIDE brick: push_distance = {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm, ensure complete topple)
- For UPRIGHT brick: push_distance = {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm, ensure complete topple)
- Minimum push distance: {PUSH_DISTANCE_MIN}m
- Maximum push distance: {PUSH_DISTANCE_MAX}m

## (5) Thinking Chain

### Step 1: Classify Current Pose
- Measured centroid height: {brick_pos[2]:.4f}m
- Compare with: H/2={H/2:.4f}m (flat), W/2={W/2:.4f}m (side), L/2={L/2:.4f}m (upright)
- Tolerance: ±0.015m

### Step 2: If Not Flat, Calculate Push Strategy
- Brick long edge direction from SAM3: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
- **Push direction (perpendicular):** ({push_dir_x:.4f}, {push_dir_y:.4f}, 0) or ({-push_dir_x:.4f}, {-push_dir_y:.4f}, 0)
- Choose direction that pushes toward open space

### Step 3: Calculate Positions (Use Provided Values)
Using push_direction = ({push_dir_x:.4f}, {push_dir_y:.4f}, 0):
- approach_xyz = ({approach_x:.4f}, {approach_y:.4f}, 0.15)
- push_start_xyz = ({push_start_x:.4f}, {push_start_y:.4f}, ground_z + contact_height)
- push_distance = {PUSH_DISTANCE_RECOMMENDED} ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm for reliable topple)

## (6) Output Format

### Output JSON
{PUSH_TOPPLE_REPLY_TEMPLATE}

### Constraints
1. All coordinates in meters
2. approach_z should be safe height (≥ 0.15m)
3. push_contact_height ≈ 60-70% of current brick height above ground
4. push_direction MUST be perpendicular to long edge: ({push_dir_x:.4f}, {push_dir_y:.4f}, 0) or ({-push_dir_x:.4f}, {-push_dir_y:.4f}, 0)
5. push_distance MUST be at least {PUSH_DISTANCE_MIN}m, recommended {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm)
6. approach position offset MUST be at least {APPROACH_OFFSET}m from brick center
7. push_start position offset MUST be at least {PUSH_START_OFFSET}m from brick center
8. All rpy values should be [3.14159, 0.0, 0.0] (gripper pointing down)
9. If brick is already flat, set action_required=false
10. Output valid JSON only, no markdown code blocks
""".strip()

    system = (
        "You are a robotic manipulation expert specialized in correcting brick poses. "
        "To topple a brick, push perpendicular to its detected long edge. "
        "The push direction must be (-sin(yaw), cos(yaw), 0) or (sin(yaw), -cos(yaw), 0). "
        f"Push distance should be at least {PUSH_DISTANCE_MIN}m, recommended {PUSH_DISTANCE_RECOMMENDED}m for reliable toppling. "
        f"Keep safe distance from brick - approach offset {APPROACH_OFFSET}m, push start offset {PUSH_START_OFFSET}m. "
        "Output valid JSON only with precise positions in meters."
    )
    
    return system, task