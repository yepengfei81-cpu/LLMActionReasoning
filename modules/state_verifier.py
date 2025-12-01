import pybullet as p
import math
import numpy as np

class StateVerifier:
    def __init__(self, env, robot_model, gripper_helper, brick_id):
        self.env = env
        self.rm = robot_model
        self.gripper = gripper_helper
        self.brick_id = brick_id
        self.cfg_v = env.cfg.get("verify", {})
        self.last_brick_z = None


    def brick_state(self):
        state = self.env.get_brick_state(brick_id=self.brick_id, include_aabb=False)
        return state['pos'], state['rpy']

    def tcp_pos_orn(self):
        return self.rm.tcp_world_pose()

    def tcp_pos(self):
        pos, _ = self.tcp_pos_orn()
        return pos

    def tip_bottom_z(self, tip_length_tcp):
        return self.tcp_pos()[2] - tip_length_tcp


    def debug_snapshot(self):
        tcp_z = float(self.tcp_pos()[2])
        tip = self.rm.estimate_tip_length_from_tcp(fallback=self.env.cfg["gripper_geom"].get("tip_length_guess", 0.16))
        bottom = tcp_z - tip
        gap = self.rm.estimate_gap_width()
        left_rad  = [p.getJointState(self.rm.id, j)[0] for j in getattr(self.rm, "left_act_joints", [])]
        right_rad = [p.getJointState(self.rm.id, j)[0] for j in getattr(self.rm, "right_act_joints", [])]
        contacts  = self.finger_contacts_with(self.brick_id)
        attached  = self.gripper.is_attached()
        return dict(
            tcp_z=tcp_z, finger_tip_len=tip, finger_bottom_z=bottom,
            gap=gap, left_rad=left_rad, right_rad=right_rad,
            contacts=contacts, attached=attached
        )


    def finger_contacts_with(self, body_id):
        total = 0
        links = self.rm.finger_link_indices if self.rm.finger_link_indices else [self.rm.ee_link]
        for link in links:
            cps = p.getContactPoints(self.rm.id, body_id, linkIndexA=link, linkIndexB=-1)
            total += len(cps)
        return total

    def finger_touching_ground(self, ground_id):
        links = self.rm.finger_link_indices if self.rm.finger_link_indices else [self.rm.ee_link]
        for link in links:
            cps = p.getContactPoints(self.rm.id, ground_id, linkIndexA=link, linkIndexB=-1)
            if cps: return True
        return False


    def brick_touching_body(self, other_body_id):
        if other_body_id is None: return False
        cps = p.getContactPoints(bodyA=self.brick_id, bodyB=other_body_id, linkIndexA=-1, linkIndexB=-1)
        return len(cps) > 0

    def brick_touching_support(self, support_ids):
        if not support_ids: return False
        for sid in support_ids:
            if self.brick_touching_body(sid):
                return True
        return False


    def cos_down_metrics(self):
        (tx, ty, tz), orn = self.tcp_pos_orn()
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        ez = R[:,2]
        cos_ori = float(np.dot(ez, np.array([0,0,-1.0])))
        ee_pos, _ = self.rm.ee_world_pose()
        v = np.array([tx, ty, tz]) - np.array(ee_pos)
        n = np.linalg.norm(v)
        cos_geo = float(abs(np.dot(v / n, np.array([0,0,-1.0])))) if n > 1e-9 else 0.0
        return cos_ori, cos_geo

    def tcp_z_down_ok(self, cos_thresh=None):
        if cos_thresh is None:
            cos_thresh = float(self.cfg_v.get("tcp_z_down_cos_thresh", 0.95))
        allow_geo = bool(self.cfg_v.get("allow_geo_as_fallback", True))
        cos_ori, cos_geo = self.cos_down_metrics()
        return (cos_ori >= cos_thresh) or (allow_geo and (cos_geo >= cos_thresh))

    def _width_dir(self, yaw):
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array([-sy, cy, 0.0])

    def tcp_ex_align_width_ok(self, ref_yaw, cos_thresh=None):
        if ref_yaw is None:
            return True
        if cos_thresh is None:
            cos_thresh = float(self.cfg_v.get("tcp_ex_align_cos_thresh", 0.92))
        _, orn = self.tcp_pos_orn()
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        ex = R[:,0]
        yb = self._width_dir(ref_yaw)
        cosv = abs(float(np.dot(ex, yb)))
        return cosv >= cos_thresh

    def xy_ok(self, target_xy):
        tol_xy = float(self.cfg_v.get("tol_xy", 0.006))
        tx, ty, _ = self.tcp_pos()
        dx, dy = tx - target_xy[0], ty - target_xy[1]
        return (dx*dx + dy*dy) <= (tol_xy*tol_xy)


    def check_pre_grasp(self, z_top, tip_length_tcp, clearance, brick_xy, brick_yaw=None):
        if brick_yaw is None: _, r = self.brick_state(); brick_yaw = r[2]
        tol_z = float(self.cfg_v.get("tol_z", 0.004))
        thr   = float(self.cfg_v.get("tcp_z_down_cos_thresh", 0.95))
        allow_geo = bool(self.cfg_v.get("allow_geo_as_fallback", True))

        z_min = self.tip_bottom_z(tip_length_tcp)
        want  = z_top + clearance - tol_z
        z_ok  = (z_min >= want)
        xy_ok = self.xy_ok(brick_xy)
        cos_ori, cos_geo = self.cos_down_metrics()
        down_ok = (cos_ori >= thr) or (allow_geo and (cos_geo >= thr))
        ex_ok  = self.tcp_ex_align_width_ok(brick_yaw)

        ok = z_ok and xy_ok and down_ok and ex_ok
        if self.cfg_v.get("verbose", True):
            print(f"[PRE_GRASP] z_ok={z_ok} (z_min={z_min:.3f} want>={want:.3f}) "
                  f"xy_ok={xy_ok} down_ok={down_ok} (ori={cos_ori:.3f} geo={cos_geo:.3f}) ex_ok={ex_ok}")
        return ok

    def check_descend_grasp(self, desired_bottom_z, tip_length_tcp, ground_id, brick_xy, brick_yaw=None):
        if brick_yaw is None: _, r = self.brick_state(); brick_yaw = r[2]
        tol_z = float(self.cfg_v.get("tol_z", 0.004))
        z_min = self.tip_bottom_z(tip_length_tcp)
        ok_h = abs(z_min - desired_bottom_z) <= tol_z
        xyok = self.xy_ok(brick_xy)
        downok = self.tcp_z_down_ok()
        exok = self.tcp_ex_align_width_ok(brick_yaw)
        gtouch = self.finger_touching_ground(ground_id)
        ok = ok_h and (not gtouch) and xyok and downok and exok
        if self.cfg_v.get("verbose", True):
            print(f"[DESCEND] z_ok={ok_h} (z_min={z_min:.3f} want={desired_bottom_z:.3f}) "
                  f"xy_ok={xyok} down_ok={downok} ex_ok={exok} finger_touch_ground={gtouch}")
        return ok

    def check_close(self, assist_cfg):
        if assist_cfg.get("enabled", False) and self.gripper.is_attached():
            return True
        return self.finger_contacts_with(self.brick_id) >= int(assist_cfg.get("min_contacts_total", 2))

    def mark_before_lift(self):
        self.last_brick_z = p.getBasePositionAndOrientation(self.brick_id)[0][2]

    def check_lift(self):
        if self.last_brick_z is None: return False
        now = p.getBasePositionAndOrientation(self.brick_id)[0][2]
        return (now - self.last_brick_z) >= float(self.cfg_v.get("lift_success_dz", 0.015))

    def check_pre_place(self, gz_top, tip_length_tcp, clearance, goal_xy, goal_yaw=None):
        tol_z = float(self.cfg_v.get("tol_z", 0.004))
        z_min = self.tip_bottom_z(tip_length_tcp)
        return (z_min >= gz_top + clearance - tol_z) and self.xy_ok(goal_xy) and \
               self.tcp_z_down_ok() and self.tcp_ex_align_width_ok(goal_yaw)

    def check_descend_place(self, desired_bottom_z, tip_length_tcp, ground_id, goal_xy, goal_yaw=None):
        tol_z = float(self.cfg_v.get("tol_z", 0.004))
        z_min = self.tip_bottom_z(tip_length_tcp)
        ok_h = abs(z_min - desired_bottom_z) <= tol_z
        return ok_h and (not self.finger_touching_ground(ground_id)) and self.xy_ok(goal_xy) and \
               self.tcp_z_down_ok() and self.tcp_ex_align_width_ok(goal_yaw)

    def check_open(self, assist_cfg):
        if assist_cfg.get("enabled", False):
            return not self.gripper.is_attached()
        return self.finger_contacts_with(self.brick_id) == 0
