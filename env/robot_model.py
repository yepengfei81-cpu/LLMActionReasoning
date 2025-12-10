import pybullet as p
from .utils import get_all_joint_info
import math
import numpy as np
from math3d.transforms import mat_to_rpy

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9: return v
    return v / n

REV = p.JOINT_REVOLUTE
PRI = p.JOINT_PRISMATIC

class RobotModel:
    def __init__(self, body_id, cfg):
        self.id = body_id
        self.cfg = cfg
        self.joint_infos = get_all_joint_info(body_id)

        self.ee_link = self._infer_ee_link()
        self._classify_fingers()

        self.chain_indices = self._chain_to_link(self.ee_link)
        self.arm_joint_indices = [j for j in self.chain_indices if j not in self.finger_joint_indices]

        self.gripper_open  = cfg["robot"]["default_open"]
        self.gripper_close = cfg["robot"]["default_close"]

        self.tcp_off_pos  = [0.0, 0.0, 0.0]
        self.tcp_off_quat = [0.0, 0.0, 0.0, 1.0]
        self.tcp_calibrated = False
        
        # Initialize TCP offset based on tip_length_guess
        self._init_tcp_offset_from_config()

        (self.ik_chain_indices,
         self.ik_chain_ll, self.ik_chain_ul, self.ik_chain_jr,
         self.ik_chain_rp, self.ik_chain_jd) = self._build_ik_params(self.chain_indices)

        all_dof_indices = [ji["index"] for ji in self.joint_infos if ji["type"] in (REV, PRI)]
        (self.ik_full_indices,
         self.ik_full_ll, self.ik_full_ul, self.ik_full_jr,
         self.ik_full_rp, self.ik_full_jd) = self._build_ik_params(all_dof_indices)

        self.ik_all_indices = self.ik_chain_indices
        self.ik_ll, self.ik_ul = self.ik_chain_ll, self.ik_chain_ul
        self.ik_jr, self.ik_rp, self.ik_jd = self.ik_chain_jr, self.ik_chain_rp, self.ik_chain_jd

    def _infer_ee_link(self):
        hint = (self.cfg["robot"].get("ee_link_name_hint") or "").lower()
        if hint:
            for ji in self.joint_infos:
                if hint in ji["link_name"].lower():
                    return ji["index"]
        return self.joint_infos[-1]["index"] if self.joint_infos else -1

    def _classify_fingers(self):
        self.finger_joint_indices = []
        self.finger_link_indices  = []

        self.left_act_joints  = []
        self.right_act_joints = []
        self.left_tip_link  = None
        self.right_tip_link = None

        keys = [k.lower() for k in self.cfg["robot"]["gripper_finger_name_keywords"]]
        for ji in self.joint_infos:
            if ji["type"] not in (REV, PRI):
                continue
            s = (ji["name"] + " " + ji["link_name"]).lower()
            if any(k in s for k in keys):
                j = ji["index"]
                self.finger_joint_indices.append(j)
                self.finger_link_indices.append(j)
                is_left  = ("left"  in s)
                is_right = ("right" in s)
                is_tip   = ("tip"   in s)
                if (not is_tip) and ("finger" in s):
                    if is_left:  self.left_act_joints.append(j)
                    if is_right: self.right_act_joints.append(j)
                if is_tip and self.left_tip_link is None and is_left:
                    self.left_tip_link = j
                if is_tip and self.right_tip_link is None and is_right:
                    self.right_tip_link = j

        if self.left_tip_link is None:
            for j in self.finger_link_indices:
                name = (p.getJointInfo(self.id, j)[12]).decode("utf-8").lower()
                if "left" in name: self.left_tip_link = j; break
        if self.right_tip_link is None:
            for j in self.finger_link_indices:
                name = (p.getJointInfo(self.id, j)[12]).decode("utf-8").lower()
                if "right" in name: self.right_tip_link = j; break

        if not self.left_act_joints:
            self.left_act_joints = [j for j in self.finger_joint_indices
                                    if "left" in (p.getJointInfo(self.id, j)[1]).decode("utf-8").lower()]
        if not self.right_act_joints:
            self.right_act_joints = [j for j in self.finger_joint_indices
                                     if "right" in (p.getJointInfo(self.id, j)[1]).decode("utf-8").lower()]

    def _chain_to_link(self, link_index):
        chain = []
        cur = link_index
        while cur != -1:
            ji = p.getJointInfo(self.id, cur)
            if ji[2] in (REV, PRI):
                chain.append(cur)
            cur = ji[16]
        chain.reverse()
        return chain

    def _build_ik_params(self, indices):
        ll, ul, jr, rp, jd = [], [], [], [], []
        valid_indices = []
        for j in indices:
            info = p.getJointInfo(self.id, j)
            jtype = info[2]
            if jtype not in (REV, PRI): continue
            lo, hi = info[8], info[9]
            if (not math.isfinite(lo)) or (not math.isfinite(hi)) or (hi <= lo):
                lo, hi = -math.pi, math.pi
            valid_indices.append(j)
            ll.append(lo); ul.append(hi); jr.append(hi - lo)
            rp.append(p.getJointState(self.id, j)[0])
            jd.append(0.08)
        return valid_indices, ll, ul, jr, rp, jd

    def ee_world_pose(self):
        ls = p.getLinkState(self.id, self.ee_link)
        return ls[4], ls[5]

    def set_arm_positions(self, q, kp=0.5, kd=0.3, max_force=200.):
        p.setJointMotorControlArray(
            self.id, self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=q,
            positionGains=[kp]*len(self.arm_joint_indices),
            velocityGains=[kd]*len(self.arm_joint_indices),
            forces=[max_force]*len(self.arm_joint_indices)
        )

    def set_finger_pos(self, target, max_force=50.):
        if not self.finger_joint_indices: return
        p.setJointMotorControlArray(
            self.id, self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[self.gripper_open if target is None else target]*len(self.finger_joint_indices),
            forces=[max_force]*len(self.finger_joint_indices)
        )

    def set_finger_sym_angles(self, theta, max_force=80.):
        if not (self.left_act_joints or self.right_act_joints):
            self.set_finger_pos(self.gripper_open, max_force=max_force)
            return
        ids, tpos, forces = [], [], []
        for j in self.left_act_joints:
            ids.append(j);  tpos.append(+theta); forces.append(max_force)
        for j in self.right_act_joints:
            ids.append(j);  tpos.append(-theta); forces.append(max_force)
        p.setJointMotorControlArray(self.id, ids, p.POSITION_CONTROL,
                                    targetPositions=tpos, forces=forces)

    def calibrate_tcp(self):
        ee_pos, ee_orn = self.ee_world_pose()

        if (self.left_tip_link is not None) and (self.right_tip_link is not None):
            lpos = np.array(p.getLinkState(self.id, self.left_tip_link)[4])
            rpos = np.array(p.getLinkState(self.id, self.right_tip_link)[4])
            tcp_pos_world = ((lpos + rpos) * 0.5).tolist()
            ez = _normalize(np.array(tcp_pos_world) - np.array(ee_pos))
            ex_meas = _normalize(rpos - lpos)
            ex_proj = ex_meas - ez * float(ex_meas @ ez)
            if np.linalg.norm(ex_proj) < 1e-6:
                ref = np.array([0.0, 0.0, 1.0])
                if abs(float(ref @ ez)) > 0.95: ref = np.array([1.0, 0.0, 0.0])
                ex_proj = _normalize(np.cross(ez, ref))
            ey = _normalize(np.cross(ez, ex_proj))
            ex = _normalize(np.cross(ey, ez))
            R = np.column_stack([ex, ey, ez])
            tcp_orn_world = p.getQuaternionFromEuler(mat_to_rpy(R))
        else:
            tcp_pos_world, tcp_orn_world = ee_pos, ee_orn

        inv_ee = p.invertTransform(ee_pos, ee_orn)
        off_pos, off_orn = p.multiplyTransforms(inv_ee[0], inv_ee[1], tcp_pos_world, tcp_orn_world)
        self.tcp_off_pos = list(off_pos)
        self.tcp_off_quat = list(off_orn)
        self.tcp_calibrated = True

        tcp_pos, tcp_orn = self.tcp_world_pose()
        Rw = np.array(p.getMatrixFromQuaternion(tcp_orn)).reshape(3,3)
        ez_w = Rw[:,2]
        print(f"[TCP] pos= {tcp_pos} ez= {ez_w.tolist()} cos_to_-Z(ori)= {float(ez_w @ np.array([0,0,-1.0]))}")
        return True

    def tcp_world_pose(self):
        ee_pos, ee_orn = self.ee_world_pose()
        return p.multiplyTransforms(ee_pos, ee_orn, self.tcp_off_pos, self.tcp_off_quat)

    def ee_pose_from_tcp(self, tcp_xyz, tcp_rpy):
        tcp_quat = p.getQuaternionFromEuler(tcp_rpy)
        inv_off = p.invertTransform(self.tcp_off_pos, self.tcp_off_quat)
        ee_pos, ee_orn = p.multiplyTransforms(tcp_xyz, tcp_quat, inv_off[0], inv_off[1])
        return dict(xyz=list(ee_pos), rpy=list(p.getEulerFromQuaternion(ee_orn)))

    def estimate_tip_length_from_tcp(self, fallback=None):
        if fallback is None:
            fallback = float(self.cfg["gripper_geom"].get("tip_length_guess", 0.16))
        tcp_pos, _ = self.tcp_world_pose()
        tcp_z = tcp_pos[2]
        if not self.finger_link_indices: return fallback
        min_z = None
        for lk in self.finger_link_indices:
            aabb_min, aabb_max = p.getAABB(self.id, lk)
            z = aabb_min[2]
            min_z = z if (min_z is None or z < min_z) else min_z
        if min_z is None: return fallback
        tip = tcp_z - float(min_z)
        if not math.isfinite(tip) or tip <= 0 or tip > 1.0: return fallback
        return float(tip)

    def _init_tcp_offset_from_config(self):
        """Initialize TCP offset to zero, to be set by dynamic calibration later"""
        self.tcp_off_pos = [0.0, 0.0, 0.0]
        self.tcp_off_quat = [0.0, 0.0, 0.0, 1.0]
        # print(f"[TCP] TCP offset initialized to zero - will be calibrated dynamically")

    def estimate_finger_depth(self, fallback=None):
        if fallback is None:
            fallback = float(self.cfg["gripper_geom"].get("finger_depth", 0.065))
        if not self.finger_link_indices: return fallback
        vals = []
        for lk in self.finger_link_indices:
            aabb_min, aabb_max = p.getAABB(self.id, lk)
            vals.append(aabb_max[2] - aabb_min[2])
        if not vals: return fallback
        v = sum(vals)/len(vals)
        if v <= 0 or v > 0.3: return fallback
        return float(v)

    def estimate_extra_ee_drop_below_fingers(self, fallback=0.0):
        try:
            if not self.tcp_calibrated or not self.finger_link_indices:
                return float(fallback)
            tcp_pos, _ = self.tcp_world_pose()
            tz = float(tcp_pos[2])
            min_finger_z = None
            for lk in self.finger_link_indices:
                aabb_min, aabb_max = p.getAABB(self.id, lk)
                z = aabb_min[2]
                min_finger_z = z if (min_finger_z is None or z < min_finger_z) else min_finger_z
            if min_finger_z is None: return float(fallback)
            aabb_min_ee, _ = p.getAABB(self.id, self.ee_link)
            min_ee_z = float(aabb_min_ee[2])
            tip_finger = tz - float(min_finger_z)
            tip_ee     = tz - float(min_ee_z)
            delta = tip_ee - tip_finger
            if not math.isfinite(delta): return float(fallback)
            return float(delta) if delta > 0.0 else 0.0
        except Exception:
            return float(fallback)

    def finger_axis_world(self):
        if (self.left_tip_link is None) or (self.right_tip_link is None): return None
        lpos = np.array(p.getLinkState(self.id, self.left_tip_link)[4])
        rpos = np.array(p.getLinkState(self.id, self.right_tip_link)[4])
        v = rpos - lpos
        n = np.linalg.norm(v)
        if n < 1e-6: return None
        return (v / n)

    def estimate_gap_width(self):
        if (self.left_tip_link is None) or (self.right_tip_link is None): return 0.0
        lpos = np.array(p.getLinkState(self.id, self.left_tip_link)[4])
        rpos = np.array(p.getLinkState(self.id, self.right_tip_link)[4])
        _, orn = self.tcp_world_pose()
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        ex = R[:,0]
        return float(abs((rpos - lpos) @ ex))

    def get_ee_link_endpoints(self):
        """Get two endpoints of the end-effector link"""
        try:

            aabb_min, aabb_max = p.getAABB(self.id, self.ee_link)
            
            # Get world position and orientation of the end-effector link
            link_state = p.getLinkState(self.id, self.ee_link)
            link_pos = link_state[4] 
            link_orn = link_state[5] 
            
            # Convert quaternion to rotation matrix
            R = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
            
            local_z_extent = aabb_max[2] - aabb_min[2]
            local_endpoint1 = np.array([0, 0, -local_z_extent/2])  # Bottom end
            local_endpoint2 = np.array([0, 0, +local_z_extent/2])  # Top end
            
            # Convert to world frame
            world_endpoint1 = np.array(link_pos) + R @ local_endpoint1
            world_endpoint2 = np.array(link_pos) + R @ local_endpoint2
            
            return world_endpoint1.tolist(), world_endpoint2.tolist()
        except Exception as e:
            print(f"[DEBUG] Error getting EE link endpoints: {e}")
            return None, None
