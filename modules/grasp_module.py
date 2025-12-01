from planning.grasp_planner import plan_improved_grasp

class GraspModule:
    def __init__(self, env):

        self.env = env
        self.cfg = env.cfg
        print(f"[GraspModule] Using improved grasp planner")

    def plan(self, brick_state, goal_pose, ground_z, brick_id=None):
        L, W, H = self.cfg["brick"]["size_LWH"]
        g = self.cfg["gripper_geom"]
        c = self.cfg["clearance"]
        
        if brick_id is None:
            raise ValueError("[GraspModule] brick_id cannot be None, improved planner requires brick_id")
        
        print(f"[GraspModule] Planning grasp path for brick #{brick_id}")
        wps, aux = plan_improved_grasp(
            self.env, brick_id, [L, W, H], goal_pose, ground_z,
            tip_length_guess=g["tip_length_guess"],
            finger_depth=g["finger_depth"],
            approach_clearance=c["approach"],
            pad_clearance=c["pad"],
            lift_clearance=c["lift"],
            place_gap=c["place_gap"],
            gripper_open_distance=g.get("max_open", 0.11)
        )
        return wps, aux
