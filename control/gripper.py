import pybullet as p
import math
from .force_feedback import ForceController

class GripperHelper:
    def __init__(self, robot_model):
        self.rm = robot_model
        self._active_constraint = None
        self._is_closed = False
        self._last_open_angle = 0.0
        self._dt = float(self.rm.cfg.get("scene", {}).get("time_step", 1.0/240.0))
        self.force_controller = ForceController(p, self.rm.finger_joint_indices)

    def last_sym_theta(self):
        return float(self._last_open_angle)

    def _settle(self, sec):
        steps = max(1, int(sec / max(self._dt, 1e-5)))
        for _ in range(steps):
            p.stepSimulation()

    def _joint_pos(self, j):
        return float(p.getJointState(self.rm.id, j)[0])

    def _joint_limits(self, j):
        ji = p.getJointInfo(self.rm.id, j)
        lo, hi = float(ji[8]), float(ji[9])
        if not math.isfinite(lo): lo = -math.pi
        if not math.isfinite(hi): hi = +math.pi
        if hi <= lo:
            lo, hi = -math.pi, +math.pi
        return lo, hi

    def _apply_pair(self, jL, jR, baseL, baseR, theta, sL=+1.0, sR=-1.0, force=80.):
        tgtL = baseL + sL*theta
        tgtR = baseR + sR*theta
        # Send control commands simultaneously to reduce time difference
        p.setJointMotorControlArray(
            self.rm.id, [jL, jR],
            p.POSITION_CONTROL,
            targetPositions=[tgtL, tgtR],
            forces=[force, force]
        )

    def open(self, target=None):
        self._is_closed = False
        if self.rm.finger_joint_indices:
            p.setJointMotorControlArray(
                self.rm.id, self.rm.finger_joint_indices, 
                p.POSITION_CONTROL, 
                targetPositions=[self.rm.gripper_open if target is None else target]*len(self.rm.finger_joint_indices), # Target position
                forces=[100.]*len(self.rm.finger_joint_indices)
            )
            self._settle(0.15) 

    def close(self, target=None):
        self._is_closed = True
        if self.rm.finger_joint_indices:
            p.setJointMotorControlArray(
                self.rm.id, self.rm.finger_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=[0.0]*len(self.rm.finger_joint_indices), 
                forces=[100.]*len(self.rm.finger_joint_indices) 
            )
            self._settle(0.20)

    def open_to_gap(self, required_gap, tol=0.001, iters=18, settle_sec=0.15, verbose=True):
        candL = []; candR = []
        if getattr(self.rm, "left_tip_link", None)  is not None:  candL.append(self.rm.left_tip_link)
        candL += list(getattr(self.rm, "left_act_joints", []))
        if getattr(self.rm, "right_tip_link", None) is not None:  candR.append(self.rm.right_tip_link)
        candR += list(getattr(self.rm, "right_act_joints", []))
        if not candL or not candR:
            if not candL and self.rm.finger_joint_indices:
                candL = [self.rm.finger_joint_indices[0]]
            if not candR and self.rm.finger_joint_indices:
                candR = [self.rm.finger_joint_indices[-1]]

        # Find most effective joint pair (jL,jR) and direction (sL,sR)
        def _slope(jL, jR, sL, sR, delta=0.02):
            baseL, baseR = self._joint_pos(jL), self._joint_pos(jR)
            g0 = self.rm.estimate_gap_width()
            self._apply_pair(jL, jR, baseL, baseR, +delta, sL, sR)
            self._settle(settle_sec)
            g1 = self.rm.estimate_gap_width()
            self._apply_pair(jL, jR, baseL, baseR, 0.0, sL, sR)
            self._settle(self._dt*3)
            return (g1 - g0) / max(delta, 1e-6)

        best = None
        for jL in candL:
            for jR in candR:
                for sL, sR in [(+1,-1),(+1,+1),(-1,-1),(-1,+1)]:
                    s = _slope(jL, jR, sL, sR, delta=0.02)
                    # if verbose:
                    #     print(f"[GAP] probe L={jL} R={jR} sign=({sL:+d},{sR:+d}) slope={s:+.4f} m/rad")
                    if (best is None) or (s > best[0]):
                        best = (s, jL, jR, sL, sR)

        slope, jL, jR, sL, sR = best if best else (0.0, candL[0], candR[0], +1, -1)
        g0 = self.rm.estimate_gap_width()
        if slope <= 1e-5:
            # if verbose:
            #     print(f"[GAP] no effective pair (slope={slope:.3e}), keep current gap={g0:.3f}.")
            return (g0 >= required_gap - tol), 0.0, g0, dict(pair=(jL,jR), slope=slope, signs=(sL,sR))

        baseL, baseR = self._joint_pos(jL), self._joint_pos(jR)
        loL, hiL = self._joint_limits(jL); loR, hiR = self._joint_limits(jR)
        thetaL = (hiL - baseL) if sL > 0 else (baseL - loL)
        thetaR = (baseR - loR) if sR < 0 else (hiR - baseR)
        theta_lim = max(0.0, min(thetaL, thetaR))
        theta_conf = float(self.rm.cfg.get("gripper_geom", {}).get("max_sym_open_rad", 0.8))
        theta_hi = min(theta_lim, max(0.0, theta_conf))
        if theta_hi <= 1e-4:
            # if verbose:
            #     print(f"[GAP] theta upper bound too small (lim={theta_lim:.4f}, conf={theta_conf:.4f}).")
            return (g0 >= required_gap - tol), 0.0, g0, dict(pair=(jL,jR), slope=slope, signs=(sL,sR), theta_hi=theta_hi)

        # if verbose:
        #     print(f"[GAP] pick L={jL} R={jR} signs=({sL:+d},{sR:+d}) slope={slope:.4f} "
        #           f"theta_hi={theta_hi:.3f} rad  g0={g0:.3f} need>={required_gap:.3f}")

        lo, hi = 0.0, theta_hi
        ok, theta_best, gap_best = False, 0.0, g0
        for it in range(iters):
            mid = 0.5*(lo + hi)
            self._apply_pair(jL, jR, baseL, baseR, mid, sL, sR)
            self._settle(settle_sec)
            g = self.rm.estimate_gap_width()
            # if verbose:
            #     print(f"[GAP] bisect it={it:02d} theta={mid:.4f}  gap={g:.3f}  need>={required_gap:.3f}")
            theta_best, gap_best = mid, g
            if g >= required_gap - tol:
                ok = True; hi = mid
            else:
                lo = mid
            if (hi - lo) < 1e-4: break

        if not ok:
            self._apply_pair(jL, jR, baseL, baseR, theta_hi, sL, sR)
            self._settle(settle_sec)
            gap_best = self.rm.estimate_gap_width()
            theta_best = theta_hi
            # if verbose:
            #     print(f"[GAP] cap by limits: theta={theta_hi:.4f}  gap_max={gap_best:.3f}  need>={required_gap:.3f}")

        self._last_open_angle = float(theta_best)
        self._is_closed = False
        return ok, theta_best, gap_best, dict(pair=(jL,jR), slope=slope, signs=(sL,sR), theta_hi=theta_hi)


    def _finger_contacts(self, body_id):
        contact_map = {}
        candidate_links = self.rm.finger_link_indices if self.rm.finger_link_indices else [self.rm.ee_link]
        for link in candidate_links:
            cps = p.getContactPoints(self.rm.id, body_id, linkIndexA=link, linkIndexB=-1)
            if cps and len(cps) > 0:
                contact_map[link] = len(cps)
        return contact_map

    def try_attach_with_contact(self, body_id, gate_cfg):
        if self._active_constraint is not None:
            return True
        if not gate_cfg.get("contact_gate", True):
            if not self._is_closed:
                return False
            return self._attach_fixed(body_id)
        if not self._is_closed:
            return False
        cmap = self._finger_contacts(body_id)
        total = sum(cmap.values())
        if gate_cfg.get("require_both_fingers", True):
            if len(cmap.keys()) < 2:
                return False
        if total < gate_cfg.get("min_contacts_total", 2):
            return False
        return self._attach_fixed(body_id)

    def _attach_fixed(self, body_id, parent_link=None):
        if parent_link is None:
            parent_link = self.rm.ee_link
        if self._active_constraint is not None:
            return True
        ee = p.getLinkState(self.rm.id, parent_link)
        ee_pos, ee_orn = ee[4], ee[5]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)
        inv_ee = p.invertTransform(ee_pos, ee_orn)
        parentFramePos, parentFrameOrn = p.multiplyTransforms(inv_ee[0], inv_ee[1], obj_pos, obj_orn)
        cid = p.createConstraint(
            parentBodyUniqueId=self.rm.id, parentLinkIndex=parent_link,
            childBodyUniqueId=body_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0,0,0],
            parentFramePosition=parentFramePos, parentFrameOrientation=parentFrameOrn,
            childFramePosition=[0,0,0], childFrameOrientation=[0,0,0,1]
        )
        self._active_constraint = cid
        return True

    def detach(self):
        if self._active_constraint is not None:
            p.removeConstraint(self._active_constraint)
            self._active_constraint = None

    def is_attached(self):
        return self._active_constraint is not None
    
    def force_feedback_release(self, brick_id, vf_checker, max_attempts=3):
        """Intelligent release method based on force feedback"""
        success = False
        
        for attempt in range(max_attempts):
            print(f"[FORCE_RELEASE] Attempt {attempt + 1}/{max_attempts}")
            
            brick_pos_before, _ = vf_checker.brick_state()
            initial_contacts = vf_checker.finger_contacts_with(brick_id)

            try:
                release_success, final_gap, force_data = self.force_controller.adaptive_release_strategy(
                    self.rm.id, brick_id, self
                )
                
                brick_pos_after, _ = vf_checker.brick_state()
                displacement = ((brick_pos_after[0] - brick_pos_before[0])**2 + 
                              (brick_pos_after[1] - brick_pos_before[1])**2)**0.5
                
                print(f"[FORCE_RELEASE] Displacement: {displacement*1000:.1f}mm")
                
                if self._active_constraint is not None:
                    p.removeConstraint(self._active_constraint)
                    self._active_constraint = None
                
                self._settle(0.05)
                
                final_contacts = vf_checker.finger_contacts_with(brick_id)
                
                if final_contacts == 0 and not self.is_attached():
                    print(f"[FORCE_RELEASE] Success! Displacement: {displacement*1000:.1f}mm")
                    success = True
                    break
                elif displacement > 0.008:  # 8mm displacement limit
                    print(f"[FORCE_RELEASE] Excessive displacement, stopping")
                    break
                else:
                    print(f"[FORCE_RELEASE] Still {final_contacts} contacts, retrying...")
                    
            except Exception as e:
                print(f"[FORCE_RELEASE] Error in attempt {attempt}: {e}")
                continue
        
        return success, final_gap if 'final_gap' in locals() else 0.0
    
    def get_current_width(self):
        """Get current gripper opening"""
        # Get positions of left and right fingers
        left_positions = []
        right_positions = []
        
        for joint_id in self.rm.finger_joint_indices:
            pos = self._joint_pos(joint_id)
            joint_info = p.getJointInfo(self.rm.id, joint_id)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            if 'left' in joint_name:
                left_positions.append(pos)
            elif 'right' in joint_name:
                right_positions.append(pos)
        
        # Estimate current opening 
        if left_positions and right_positions:
            avg_left = sum(left_positions) / len(left_positions)
            avg_right = sum(right_positions) / len(right_positions)
            # Approximate opening calculation
            current_theta = (abs(avg_left) + abs(avg_right)) / 2
            # Convert to actual opening based on gripper geometry (this is an approximation)
            return current_theta * 0.2  # Estimated scaling factor
        
        return 0.04 
    
    def set_width(self, target_width):
        """Set gripper opening"""
        target_theta = target_width / 0.2 
        
        # Apply symmetric control to all finger joints
        for joint_id in self.rm.finger_joint_indices:
            joint_info = p.getJointInfo(self.rm.id, joint_id)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            if 'left' in joint_name:
                target_pos = -target_theta
            elif 'right' in joint_name:
                target_pos = target_theta
            else:
                continue
                
            p.setJointMotorControl2(
                self.rm.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=40.0
            )
