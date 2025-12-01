import pybullet as p
import yaml
import time
from .utils import set_search_path
from .robot_model import RobotModel

class BulletEnv:
    def __init__(self, cfg_path="configs/kuka.yaml", use_gui=True):
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.dt = self.cfg["scene"]["time_step"]
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, self.cfg["scene"]["gravity"])
        set_search_path()

        self._load_scene()
        self._load_robot()

        self.brick_ids = self._spawn_bricks()
        self.brick_id = self.brick_ids[0]

        # Parse layout configuration
        self._parse_layout()

    def _parse_layout(self):
        """Parse goal_layout configuration, generate flattened target list"""
        self.layout_targets = []
        goal_layout = self.cfg.get("goal_layout", {})
        
        # Extract all level keys and sort 
        sorted_levels = []
        for key in goal_layout.keys():
            if key.startswith("level_"):
                try:
                    lvl_idx = int(key.split("_")[1])
                    sorted_levels.append((lvl_idx, key))
                except ValueError:
                    continue
        sorted_levels.sort(key=lambda x: x[0])
        
        for lvl_idx, key in sorted_levels:
            positions = goal_layout[key].get("positions", [])
            for pos_idx, pos in enumerate(positions):
                self.layout_targets.append({
                    "level": lvl_idx,
                    "pos_index": pos_idx,
                    "xy": pos,
                    "level_key": key
                })
        
        print(f"[Env] Parsed layout with {len(self.layout_targets)} targets across {len(sorted_levels)} levels")

    def _load_scene(self):
        self.ground_id = None
        if self.cfg["scene"].get("plane", True):
            self.ground_id = p.loadURDF("plane.urdf")
            p.changeDynamics(self.ground_id, -1, lateralFriction=self.cfg["scene"]["friction"]["ground"])
        self.table_id = None

    def _load_robot(self):
        sdf_path = self.cfg["robot"]["sdf_path"]
        ids = p.loadSDF(sdf_path)
        self.robot_id = ids[0]
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        self.robot_model = RobotModel(self.robot_id, self.cfg)
        for j in self.robot_model.finger_joint_indices:
            p.changeDynamics(self.robot_id, j, lateralFriction=self.cfg["scene"]["friction"]["finger"])

    def _ground_top_z(self):
        return 0.0

    def _spawn_bricks(self):
        L, W, H = self.cfg["brick"]["size_LWH"]
        color = self.cfg["brick"]["color_rgba"]
        mass = self.cfg["brick"]["mass"]

        sp_list = self.cfg["brick"].get("spawn_xy_list", None)
        if not sp_list:
            sp_list = [self.cfg["brick"]["spawn_xy"]]
        brick_ids = []
        for (sx, sy) in sp_list:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, H/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L/2, W/2, H/2], rgbaColor=color)
            z = self._ground_top_z() + H/2 + 0.005
            bid = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[sx, sy, z],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            p.changeDynamics(bid, -1, lateralFriction=self.cfg["scene"]["friction"]["brick"])
            brick_ids.append(bid)
        return brick_ids

    def compute_goal_pose_for_level(self, level:int):
        L, W, H = self.cfg["brick"]["size_LWH"]
        
        goal_layout = self.cfg.get("goal_layout", None)
        if goal_layout:
            return self.compute_goal_pose_from_layout(level)
        
        ox, oy = self.cfg["goal"]["offset_xy"]
        gz0 = self._ground_top_z() + H/2
        gz = gz0 + level * H
        yaw = self.cfg["goal"]["yaw"]
        return (ox, oy, gz, 0.0, 0.0, yaw)
    
    def compute_goal_pose_from_layout(self, brick_index:int):
        """
        Calculate target position for specific brick based on goal_layout configuration
        Dynamically supports arbitrary level structure
        """
        L, W, H = self.cfg["brick"]["size_LWH"]
        yaw = self.cfg["goal"]["yaw"]

        if not hasattr(self, "layout_targets"):
            self._parse_layout()
            
        if brick_index < len(self.layout_targets):
            target = self.layout_targets[brick_index]
            level = target["level"]
            ox, oy = target["xy"]
        else:
            # If exceeding layout definition, use default strategy
            print(f"[Env] Warning: Brick index {brick_index} exceeds layout definition. Using default offset.")
            ox, oy = self.cfg["goal"]["offset_xy"]
            ox += (brick_index - len(self.layout_targets)) * 0.1
            level = 0 # Default to base level
            
        # Calculate Z coordinate (level height)
        gz0 = self._ground_top_z() + H/2
        gz = gz0 + level * H
        
        return (ox, oy, gz, 0.0, 0.0, yaw)
    
    def get_brick_placement_sequence(self):
        """
        Return brick placement sequence
        Based on parsed layout target count and actual brick count
        """
        if not hasattr(self, "layout_targets"):
            self._parse_layout()
            
        num_tasks = min(len(self.brick_ids), len(self.layout_targets))
        return list(range(num_tasks))

    def get_related_support_ids(self, brick_index):
        """Get list of support IDs for current brick"""
        if not hasattr(self, "layout_targets"):
            self._parse_layout()
            
        if brick_index >= len(self.layout_targets):
            return [self.ground_id]
            
        current_level = self.layout_targets[brick_index]["level"]
        
        if current_level == 0:
            return [self.ground_id]
            
        support_brick_indices = [
            i for i, t in enumerate(self.layout_targets) 
            if t["level"] < current_level and i < len(self.brick_ids)
        ]
        
        support_ids = [self.ground_id] + [self.brick_ids[i] for i in support_brick_indices]
        return support_ids

    def get_level_name(self, brick_index):
        """Get brick level name"""
        if not hasattr(self, "layout_targets"):
            self._parse_layout()
        
        if brick_index < len(self.layout_targets):
            level = self.layout_targets[brick_index]["level"]
            return f"Level {level}"
        return "Unknown Level"

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            time.sleep(self.dt)

    def get_brick_state(self, brick_id=None, include_aabb=False):
        """Get brick state
        Returns:
            dict: Brick state dictionary
                Basic mode: {pos, rpy}
                Detailed mode: {pos, rpy, aabb_center, aabb_min, aabb_max}
        """
        if brick_id is None:
            brick_id = self.brick_id
            
        pos, orn = p.getBasePositionAndOrientation(brick_id)
        rpy = p.getEulerFromQuaternion(orn)
        
        state = dict(pos=pos, rpy=rpy)
        
        if include_aabb:
            aabb_min, aabb_max = p.getAABB(brick_id)
            aabb_center = tuple((aabb_min[i] + aabb_max[i]) / 2 for i in range(3))
            state.update({
                'aabb_center': aabb_center,
                'aabb_min': aabb_min,
                'aabb_max': aabb_max
            })
            
        return state

    def get_ground_top(self):
        return self._ground_top_z()

    def disconnect(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
