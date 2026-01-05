# import numpy as np

# # ========== Push Distance Configuration ==========
# PUSH_DISTANCE_MIN = 0.12          # Minimum push distance (meters)
# PUSH_DISTANCE_RECOMMENDED = 0.15  # Recommended push distance (meters) 
# PUSH_DISTANCE_MAX = 0.20          # Maximum push distance (meters)

# # ========== Approach Distance Configuration - Prevent Knocking Over Brick ==========
# APPROACH_OFFSET = 0.12            # Approach position offset from brick center (meters)
# PUSH_START_OFFSET = 0.06          # Push start position offset from brick center (meters)
# # Note: Larger PUSH_START_OFFSET means starting push further from the brick
# # ================================================

# PUSH_TOPPLE_REPLY_TEMPLATE = """{
#   "pose_analysis": {
#     "detected_pose": "flat|side|upright",
#     "measured_centroid_z": <float>,
#     "confidence": <float 0-1>,
#     "reasoning": "<explanation of pose classification>"
#   },
#   "action_required": <true if not flat, false if already flat>,
#   "push_plan": {
#     "approach_pose": {
#       "xyz": [<approach_x>, <approach_y>, <approach_z>],
#       "rpy": [3.14159, 0.0, 0.0]
#     },
#     "push_start_pose": {
#       "xyz": [<start_x>, <start_y>, <start_z>],
#       "rpy": [3.14159, 0.0, 0.0]
#     },
#     "push_direction": [<dx>, <dy>, 0.0],
#     "push_distance": <float in meters>,
#     "push_contact_height": <float, height above ground to contact brick>,
#     "push_speed": "slow|medium",
#     "description": "<explanation of push strategy>"
#   },
#   "retreat_pose": {
#     "xyz": [<retreat_x>, <retreat_y>, <retreat_z>],
#     "rpy": [3.14159, 0.0, 0.0]
#   },
#   "expected_result": {
#     "final_pose": "flat",
#     "expected_height_after": <float>
#   }
# }"""


# def get_push_topple_prompt(context: dict, attempt_idx: int = 0, feedback: str = "") -> tuple:
#     """
#     Generate push topple phase prompt.
#     """
#     # Extract brick information
#     brick_pos = context["brick"]["pos"]
#     brick_size = context["brick"]["size_LWH"]  # [L, W, H]
#     L, W, H = brick_size
#     measured_height = context["brick"].get("measured_height", brick_pos[2])
#     mask_area = context["brick"].get("mask_area", 0)
    
#     # Get estimated yaw angle from SAM3
#     estimated_yaw = context["brick"].get("estimated_yaw", 0.0)
#     estimated_yaw_deg = np.degrees(estimated_yaw)
    
#     # Long edge direction detected by SAM3
#     long_axis_x = np.cos(estimated_yaw)
#     long_axis_y = np.sin(estimated_yaw)
    
#     # CRITICAL: Push direction should be PERPENDICULAR to the detected long edge
#     # Perpendicular direction = rotate 90 degrees = (-sin(yaw), cos(yaw), 0)
#     # This ensures proper toppling instead of sliding along the long edge
#     push_dir_x = -np.sin(estimated_yaw)  # Perpendicular to long edge
#     push_dir_y = np.cos(estimated_yaw)   # Perpendicular to long edge
    
#     # Robot state
#     tcp_xyz = context["now"]["tcp_xyz"]
#     tcp_rpy = context["now"]["tcp_rpy"]
    
#     # Ground and safety
#     ground_z = context.get("constraints", {}).get("ground_z", 0.0)
#     safety_clearance = context.get("constraints", {}).get("safety_clearance", 0.05)
    
#     # Gripper info
#     tip_length = context["gripper"].get("measured_tip_length") or context["gripper"].get("tip_length_guess", 0.05)
    
#     # Calculate preset positions (for LLM reference)
#     # Approach position: on the opposite side of push direction, at APPROACH_OFFSET distance
#     approach_x = brick_pos[0] - push_dir_x * APPROACH_OFFSET
#     approach_y = brick_pos[1] - push_dir_y * APPROACH_OFFSET
    
#     # Push start position: on the opposite side of push direction, at PUSH_START_OFFSET distance
#     # This position should be just outside the brick edge
#     push_start_x = brick_pos[0] - push_dir_x * PUSH_START_OFFSET
#     push_start_y = brick_pos[1] - push_dir_y * PUSH_START_OFFSET
    
#     task = f"""
# ## (1) Current Environment State

# ### Brick Geometry (IMPORTANT - memorize these values)
# - Brick dimensions: L={L:.4f}m (longest), W={W:.4f}m (medium), H={H:.4f}m (shortest)
# - When lying flat (normal): centroid height ≈ H/2 = {H/2:.4f}m
# - When on side: centroid height ≈ W/2 = {W/2:.4f}m
# - When upright: centroid height ≈ L/2 = {L/2:.4f}m

# ### Detected Brick State
# - Brick centroid position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f})m
# - Measured centroid height: {brick_pos[2]:.4f}m
# - Detected mask area: {mask_area} pixels

# ### CRITICAL: Brick Orientation from SAM3
# - Estimated yaw angle: {estimated_yaw:.4f} rad ({estimated_yaw_deg:.1f}°)
# - Detected long edge direction: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
# - **Push direction (perpendicular to long edge):** ({push_dir_x:.4f}, {push_dir_y:.4f}, 0)

# ### Expected Heights for Pose Classification
# - Flat (normal): centroid_z ≈ {H/2:.4f}m (half of H={H:.4f}m)
# - On side: centroid_z ≈ {W/2:.4f}m (half of W={W/2:.4f}m)
# - Upright: centroid_z ≈ {L/2:.4f}m (half of L={L:.4f}m)

# ### Robot State
# - Current TCP position: ({tcp_xyz[0]:.4f}, {tcp_xyz[1]:.4f}, {tcp_xyz[2]:.4f})m
# - Gripper tip length: {tip_length:.4f}m

# ### Environment
# - Ground Z: {ground_z:.4f}m
# - Safety clearance: {safety_clearance:.4f}m

# ## (2) Memory Information

# ### Task Progress
# - Current phase: Push topple planning
# - Objective: Push the standing/tilted brick to make it lie flat (L×W face on ground)
# - Attempt: {attempt_idx + 1}

# ### Error Feedback
# {('Previous attempt failed: ' + feedback) if feedback else 'No previous feedback'}

# ## (3) Role Definition

# You are a **Brick Pose Correction Expert Agent**. Your task is to:
# 1. Determine if the brick needs correction based on its height
# 2. Plan a push action **perpendicular to the detected long edge** to topple the brick

# ## (4) Knowledge Base - Push Direction Calculation

# ### KEY INSIGHT: Push Perpendicular to the Long Edge

# To topple a standing brick, you must push it **perpendicular to its long edge**.
# - Long edge direction from SAM3: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
# - Push direction = rotate 90°: ({push_dir_x:.4f}, {push_dir_y:.4f}, 0)

# **Why perpendicular?**
# - Pushing along the long edge just "scrapes" the side face - the brick won't fall
# - Pushing perpendicular applies torque around the bottom edge, causing rotation and toppling

# ```
#     Long edge direction: →
    
#     ┌──────────────┐      Push ↓ (perpendicular)
#     │              │            ↓
#     │    BRICK     │      ──────────────
#     │              │      Brick falls flat!
#     └──────────────┘
# ```

# ### IMPORTANT: Safe Distance from Brick

# To avoid accidentally knocking the brick before pushing:
# - Approach offset: {APPROACH_OFFSET}m (distance from brick center when approaching)
# - Push start offset: {PUSH_START_OFFSET}m (distance from brick center when starting push)

# For UPRIGHT brick (height ≈ L/2 = {L/2:.4f}m), the brick is very unstable!
# - Keep larger distance to avoid accidental contact
# - Push start should be at least {PUSH_START_OFFSET}m away from brick center

# ### Approach Position Calculation
# To push in direction ({push_dir_x:.4f}, {push_dir_y:.4f}, 0), approach from the opposite side:
# - approach_x = brick_x - push_dir_x × {APPROACH_OFFSET} = {approach_x:.4f}
# - approach_y = brick_y - push_dir_y × {APPROACH_OFFSET} = {approach_y:.4f}
# - approach_z = 0.15m (safe height)

# ### Push Start Position (Keep Safe Distance)
# - push_start_x = brick_x - push_dir_x × {PUSH_START_OFFSET} = {push_start_x:.4f}
# - push_start_y = brick_y - push_dir_y × {PUSH_START_OFFSET} = {push_start_y:.4f}
# - push_start_z = ground_z + push_contact_height

# ### Push Contact Height
# - push_contact_height ≈ 60-70% of current brick height above ground
# - For SIDE: contact at ~{0.65 * W:.4f}m
# - For UPRIGHT: contact at ~{0.65 * L:.4f}m

# ### Push Distance (Must Be Sufficient to Topple)
# - For SIDE brick: push_distance = {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm, ensure complete topple)
# - For UPRIGHT brick: push_distance = {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm, ensure complete topple)
# - Minimum push distance: {PUSH_DISTANCE_MIN}m
# - Maximum push distance: {PUSH_DISTANCE_MAX}m

# ## (5) Thinking Chain

# ### Step 1: Classify Current Pose
# - Measured centroid height: {brick_pos[2]:.4f}m
# - Compare with: H/2={H/2:.4f}m (flat), W/2={W/2:.4f}m (side), L/2={L/2:.4f}m (upright)
# - Tolerance: ±0.015m

# ### Step 2: If Not Flat, Calculate Push Strategy
# - Brick long edge direction from SAM3: ({long_axis_x:.4f}, {long_axis_y:.4f}, 0)
# - **Push direction (perpendicular):** ({push_dir_x:.4f}, {push_dir_y:.4f}, 0) or ({-push_dir_x:.4f}, {-push_dir_y:.4f}, 0)
# - Choose direction that pushes toward open space

# ### Step 3: Calculate Positions (Use Provided Values)
# Using push_direction = ({push_dir_x:.4f}, {push_dir_y:.4f}, 0):
# - approach_xyz = ({approach_x:.4f}, {approach_y:.4f}, 0.15)
# - push_start_xyz = ({push_start_x:.4f}, {push_start_y:.4f}, ground_z + contact_height)
# - push_distance = {PUSH_DISTANCE_RECOMMENDED} ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm for reliable topple)

# ## (6) Output Format

# ### Output JSON
# {PUSH_TOPPLE_REPLY_TEMPLATE}

# ### Constraints
# 1. All coordinates in meters
# 2. approach_z should be safe height (≥ 0.15m)
# 3. push_contact_height ≈ 60-70% of current brick height above ground
# 4. push_direction MUST be perpendicular to long edge: ({push_dir_x:.4f}, {push_dir_y:.4f}, 0) or ({-push_dir_x:.4f}, {-push_dir_y:.4f}, 0)
# 5. push_distance MUST be at least {PUSH_DISTANCE_MIN}m, recommended {PUSH_DISTANCE_RECOMMENDED}m ({PUSH_DISTANCE_RECOMMENDED*100:.0f}cm)
# 6. approach position offset MUST be at least {APPROACH_OFFSET}m from brick center
# 7. push_start position offset MUST be at least {PUSH_START_OFFSET}m from brick center
# 8. All rpy values should be [3.14159, 0.0, 0.0] (gripper pointing down)
# 9. If brick is already flat, set action_required=false
# 10. Output valid JSON only, no markdown code blocks
# """.strip()

#     system = (
#         "You are a robotic manipulation expert specialized in correcting brick poses. "
#         "To topple a brick, push perpendicular to its detected long edge. "
#         "The push direction must be (-sin(yaw), cos(yaw), 0) or (sin(yaw), -cos(yaw), 0). "
#         f"Push distance should be at least {PUSH_DISTANCE_MIN}m, recommended {PUSH_DISTANCE_RECOMMENDED}m for reliable toppling. "
#         f"Keep safe distance from brick - approach offset {APPROACH_OFFSET}m, push start offset {PUSH_START_OFFSET}m. "
#         "Output valid JSON only with precise positions in meters."
#     )
    
#     return system, task

import numpy as np

# ========== Push Distance Configuration ==========
PUSH_DISTANCE_MIN = 0.12
PUSH_DISTANCE_RECOMMENDED = 0.15
PUSH_DISTANCE_MAX = 0.20

# ========== Approach Distance Configuration ==========
APPROACH_OFFSET = 0.12
PUSH_START_OFFSET = 0.06

# ========== 面积判断阈值（简化为单一阈值）==========
# 大于此值 = 平躺或堆叠（面积大）
# 小于此值 = 竖立或侧躺（面积小）
AREA_THRESHOLD = 500  # 像素，根据实际相机调整
# ================================================

# ========== 批量姿态检测 ==========
BATCH_POSE_CHECK_REPLY_TEMPLATE = """{
  "brick_analyses": [
    {
      "brick_id": <int>,
      "detected_pose": "flat|side|upright|stacked",
      "action_required": <bool>,
      "height_category": "low|medium|high|stacked_high",
      "area_category": "small|large",
      "confidence": <float 0-1>,
      "reasoning": "<brief explanation>"
    }
  ],
  "summary": {
    "total_bricks": <int>,
    "flat_count": <int>,
    "stacked_count": <int>,
    "needs_correction_count": <int>,
    "first_to_fix_id": <int or null if none needs fixing>
  }
}"""

# ========== 单砖推倒规划（仅用于需要修复的砖块）==========
PUSH_PLAN_REPLY_TEMPLATE = """{
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
    "push_contact_height": <float>,
    "description": "<explanation of push strategy>"
  },
  "retreat_pose": {
    "xyz": [<retreat_x>, <retreat_y>, <retreat_z>],
    "rpy": [3.14159, 0.0, 0.0]
  }
}"""


def get_batch_pose_check_prompt(brick_list: list, brick_size_LWH: list, ground_z: float = 0.0) -> tuple:
    """
    生成批量砖块姿态检测的提示词（一次判断所有砖块）
    
    Args:
        brick_list: List of brick info dicts, each containing:
            - brick_id: int
            - position: [x, y, z]
            - measured_height: float
            - mask_area: int
            - estimated_yaw: float
        brick_size_LWH: [L, W, H] brick dimensions
        ground_z: ground surface height
    
    Returns:
        (system_prompt, user_prompt)
    """
    L, W, H = brick_size_LWH
    
    # 构建砖块列表表格
    brick_table = "| ID | Height (m) | Area (px) | Yaw (°) |\n"
    brick_table += "|-----|------------|-----------|----------|\n"
    
    for brick in brick_list:
        bid = brick.get('brick_id', -1)
        height = brick.get('measured_height', brick.get('position', [0,0,0])[2])
        area = brick.get('mask_area', 0)
        yaw = brick.get('estimated_yaw', 0.0)
        yaw_deg = np.degrees(yaw)
        brick_table += f"| {bid} | {height:.4f} | {area} | {yaw_deg:.1f} |\n"
    
    task = f"""
## Task: Batch Brick Pose Classification

You need to classify ALL {len(brick_list)} bricks in ONE response.

### Brick Geometry Reference
- Dimensions: L={L:.4f}m, W={W:.4f}m, H={H:.4f}m
- Height thresholds:
  - FLAT: H/2 = {H/2:.4f}m (±0.015m tolerance)
  - SIDE: W/2 = {W/2:.4f}m (±0.015m tolerance)
  - UPRIGHT: L/2 = {L/2:.4f}m
  - STACKED: ≥ H + H/2 = {H + H/2:.4f}m

### Area Threshold: {AREA_THRESHOLD} pixels
- Area ≥ {AREA_THRESHOLD} → LARGE (flat or stacked)
- Area < {AREA_THRESHOLD} → SMALL (side or upright)

### Classification Rules
| Height | Area | Pose | action_required |
|--------|------|------|-----------------|
| LOW (~{H/2:.3f}m) | LARGE | FLAT | false |
| MEDIUM (~{W/2:.3f}m) | SMALL | SIDE | true |
| HIGH (~{L/2:.3f}m) | SMALL | UPRIGHT | true |
| HIGH (≥{H+H/2:.3f}m) | LARGE | STACKED | false |

### Bricks to Classify

{brick_table}

### Output Format

{BATCH_POSE_CHECK_REPLY_TEMPLATE}

### Requirements
1. Classify ALL {len(brick_list)} bricks
2. Set action_required=false for FLAT and STACKED
3. Set action_required=true for SIDE and UPRIGHT
4. In summary.first_to_fix_id, provide the brick_id of the FIRST brick that needs correction (or null if none)
5. Output valid JSON only, no markdown
""".strip()

    system = (
        f"You are a brick pose classifier. Classify all {len(brick_list)} bricks using height + area. "
        f"Area threshold = {AREA_THRESHOLD} pixels. "
        "FLAT/STACKED = large area, SIDE/UPRIGHT = small area. "
        "Output JSON with all brick analyses in one response."
    )
    
    return system, task


def get_push_plan_prompt(brick_info: dict, brick_size_LWH: list, ground_z: float = 0.0) -> tuple:
    """
    为单个需要推倒的砖块生成推倒规划提示词
    
    Args:
        brick_info: 需要推倒的砖块信息
        brick_size_LWH: [L, W, H] brick dimensions
        ground_z: ground surface height
    
    Returns:
        (system_prompt, user_prompt)
    """
    L, W, H = brick_size_LWH
    
    brick_pos = brick_info.get('position', [0, 0, 0])
    measured_height = brick_info.get('measured_height', brick_pos[2])
    estimated_yaw = brick_info.get('estimated_yaw', 0.0)
    detected_pose = brick_info.get('detected_pose', 'unknown')
    
    # 计算推动方向（垂直于长边）
    push_dir_x = -np.sin(estimated_yaw)
    push_dir_y = np.cos(estimated_yaw)
    
    # 计算接近位置
    approach_x = brick_pos[0] - push_dir_x * APPROACH_OFFSET
    approach_y = brick_pos[1] - push_dir_y * APPROACH_OFFSET
    push_start_x = brick_pos[0] - push_dir_x * PUSH_START_OFFSET
    push_start_y = brick_pos[1] - push_dir_y * PUSH_START_OFFSET
    
    # 根据姿态计算接触高度
    if detected_pose == 'upright':
        contact_height = ground_z + L * 0.65  # 竖立砖块，接触高度约65%
    elif detected_pose == 'side':
        contact_height = ground_z + W * 0.65  # 侧躺砖块
    else:
        contact_height = ground_z + measured_height * 0.65
    
    task = f"""
## Task: Plan Push Action for Brick {brick_info.get('brick_id', -1)}

### Brick State
- Position: ({brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f})m
- Detected pose: {detected_pose}
- Measured height: {measured_height:.4f}m
- Estimated yaw: {np.degrees(estimated_yaw):.1f}°

### Brick Geometry
- Dimensions: L={L:.4f}m, W={W:.4f}m, H={H:.4f}m
- Ground Z: {ground_z:.4f}m

### Pre-calculated Values (use these)
- Push direction (perpendicular to long edge): ({push_dir_x:.4f}, {push_dir_y:.4f}, 0)
- Approach position: ({approach_x:.4f}, {approach_y:.4f}, 0.15)
- Push start position: ({push_start_x:.4f}, {push_start_y:.4f}, {contact_height:.4f})
- Recommended push distance: {PUSH_DISTANCE_RECOMMENDED}m
- Contact height: {contact_height:.4f}m

### Output Format

{PUSH_PLAN_REPLY_TEMPLATE}

### Requirements
1. Use the pre-calculated push direction
2. push_distance must be at least {PUSH_DISTANCE_MIN}m
3. All rpy values should be [3.14159, 0.0, 0.0] (gripper pointing down)
4. Output valid JSON only, no markdown
""".strip()

    system = (
        "You are a robotic manipulation expert. Plan push action to topple a standing brick. "
        "Use the pre-calculated values provided. Output valid JSON only."
    )
    
    return system, task