import pybullet as p
import numpy as np
import math

class MotionVisualizer:
    def __init__(self, env, robot_model, gripper_helper, verifier):
        self.env = env
        self.rm = robot_model
        self.gripper = gripper_helper
        self.vf = verifier
        self.debug = env.cfg.get("debug", {})
        
        self.draw_axes = bool(self.debug.get("draw_tcp_axes", True))
        self.axes_len = float(self.debug.get("axes_len", 0.08))

    def banner(self, text):
        if not self.debug.get("show_phase_text", True): return
        p.addUserDebugText(text, [0.0, -0.6, 1.0], [1.0, 1.0, 0.2], 1.2)

    def draw_tcp_axes(self):
        if not self.draw_axes: return
        pos, orn = self.rm.tcp_world_pose()
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        L = self.axes_len
        p.addUserDebugLine(pos, (np.array(pos)+R[:,0]*L).tolist(), [0,1,0], 2, lifeTime=1.0)
        p.addUserDebugLine(pos, (np.array(pos)+R[:,1]*L).tolist(), [1,0,1], 2, lifeTime=1.0)
        p.addUserDebugLine(pos, (np.array(pos)+R[:,2]*L).tolist(), [0,0,1], 2, lifeTime=1.0)

    def draw_down_at_tcp(self):
        if not self.draw_axes: return
        pos, _ = self.rm.tcp_world_pose()
        L = self.axes_len
        a = np.array(pos); b = (a + np.array([0.0, 0.0, -L])).tolist()
        p.addUserDebugLine(a.tolist(), b, [1,1,0], 2, lifeTime=1.0)

    def draw_brick_width_at_center(self, bx, by, bz, yaw, length=0.25):
        cy, sy = math.cos(yaw), math.sin(yaw)
        yb = np.array([-sy, cy, 0.0])
        c  = np.array([bx, by, bz])
        a  = (c - yb*length/2).tolist(); b = (c + yb*length/2).tolist()
        p.addUserDebugLine(a, b, [1.0, 0.6, 0.0], 3, lifeTime=1.0)

    def snap(self, label):
        s = self.vf.debug_snapshot()
        import math as _m
        left_deg  = [_m.degrees(a) for a in s["left_rad"]]
        right_deg = [_m.degrees(a) for a in s["right_rad"]]
        try:
            theta_last = self.gripper.last_sym_theta()
        except Exception:
            theta_last = 0.0
        # print(
        #     "[SNAP] {:<20s} tcp_z={:.3f}  bottom_z={:.3f}  tip={:.3f}  gap={:.3f}  "
        #     "θ_last={:.3f}rad ({:.1f}°)  L(deg)=[{}]  R(deg)=[{}]  attached={}  contacts={}".format(
        #         label, s["tcp_z"], s["finger_bottom_z"], s["finger_tip_len"], s["gap"],
        #         theta_last, _m.degrees(theta_last),
        #         ",".join(f"{d:+.1f}" for d in left_deg),
        #         ",".join(f"{d:+.1f}" for d in right_deg),
        #         s["attached"], s["contacts"]
        #     )
        # )
