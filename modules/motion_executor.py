import pybullet as p
import numpy as np
import time
from control.controllers import move_tcp_cartesian
from math3d.transforms import align_topdown_to_brick
import math
from typing import Any, Dict, Optional
from modules.motion_visualizer import MotionVisualizer
from modules.motion_llm import MotionLLMHandler

try:
    from control.angle_optimizer import AngleOptimizer
    ANGLE_OPT_AVAILABLE = True
except ImportError:
    ANGLE_OPT_AVAILABLE = False
    print("[ANGLE_OPT] Angle optimizer not available")

class MotionExecutor:
    def __init__(self, env, robot_model, gripper_helper, verifier):
        self.env = env
        self.rm = robot_model
        self.gripper = gripper_helper
        self.vf = verifier
        self.timing = env.cfg.get("timing", {})
        self.debug = env.cfg.get("debug", {})
        self.gcfg = env.cfg.get("gripper_geom", {})
        self.ccfg = env.cfg.get("clearance", {})
        self.vcfg = env.cfg.get("verify", {})
        self.ocfg = env.cfg.get("orientation", {})
        self.open_clearance = float(self.gcfg.get("open_clearance", 0.010))

        self.move_sec = float(self.timing.get("move_per_phase_sec", 1.3))
        self.settle_sec = float(self.timing.get("settle_sec", 0.3))

        self.retries = int(self.vcfg.get("max_retries_per_phase", 2))
        self.max_restarts = int(self.vcfg.get("max_global_restarts", 3))
        self.recentre_step = float(self.vcfg.get("recentre_step_xy", 0.005))

        self.rpy_bias = [0.0, 0.0]
        self.yaw_quadrant = 0

        # Brick release strategy
        self.release_above = float(self.ccfg.get("release_above", 0.010))
        self.release_open_extra = float(self.ccfg.get("release_open_extra", 0.020))
        self.release_retry_extra = float(self.ccfg.get("release_retry_extra", 0.010))
        self.release_max_extra = float(self.ccfg.get("release_max_extra", 0.050))
        
        # Quick release switches
        self.use_quick_release = bool(self.ccfg.get("use_quick_sync_release", True))
        self.use_force_feedback = bool(self.ccfg.get("use_force_feedback_release", True))

        # Baseline mode configuration
        self.baseline_cfg = env.cfg.get("baseline", {})
        self.baseline_enabled = bool(self.baseline_cfg.get("enabled", False))
        self.skip_collision_detection = bool(self.baseline_cfg.get("skip_collision_detection", False))
        self.skip_force_feedback = bool(self.baseline_cfg.get("skip_force_feedback", False))
        self.skip_angle_correction = bool(self.baseline_cfg.get("skip_angle_correction", False))
        self.direct_release_gap = float(self.baseline_cfg.get("direct_release_gap", 0.05))
        
        if self.baseline_enabled:
            print("="*60)
            print("[BASELINE MODE] Enabled - Simplified placement strategy")
            print(f"  - Skip collision detection: {self.skip_collision_detection}")
            print(f"  - Skip force feedback: {self.skip_force_feedback}")
            print(f"  - Skip angle correction: {self.skip_angle_correction}")
            print(f"  - Direct release gap: {self.direct_release_gap:.3f}m")
            print("="*60)

        # Initialize submodules
        self.viz = MotionVisualizer(env, robot_model, gripper_helper, verifier)
        self.llm_handler = MotionLLMHandler(env, robot_model, gripper_helper, verifier)
        
        # Compatibility attribute for external access
        self.llm_agent = self.llm_handler.llm_agent
        
        # Angle optimization system
        self.angle_optimizer = None
        if ANGLE_OPT_AVAILABLE and not (self.baseline_enabled and self.skip_angle_correction):
            import pybullet as p
            self.angle_optimizer = AngleOptimizer(robot_model, p)
            print("[ANGLE_OPT] Angle optimizer initialized")
        elif self.baseline_enabled and self.skip_angle_correction:
            print("[ANGLE_OPT] Disabled in baseline mode")

    def _wait(self, sec):
        if sec <= 0: return
        self.env.step(int(sec / self.env.dt))

    def _move_tcp(self, tcp_pose):
        move_tcp_cartesian(self.env, self.rm, tcp_pose, duration=self.move_sec)
        self._wait(self.settle_sec)
        self.viz.draw_tcp_axes()



    def _brick_xy_yaw(self):
        pos, rpy = self.vf.brick_state()
        return pos[0], pos[1], pos[2], rpy[2]

    def _get_brick_bottom_normal(self):
        """Get brick bottom surface normal vector (in world coordinate system)"""
        pos, rpy = self.vf.brick_state()
        brick_roll, brick_pitch, brick_yaw = rpy
        
        # Brick bottom normal vector in brick coordinate system is [0, 0, -1] 
        from math3d.transforms import rpy_to_mat
        R_brick = rpy_to_mat(brick_roll, brick_pitch, brick_yaw)
        bottom_normal_local = np.array([0.0, 0.0, -1.0])
        bottom_normal_world = R_brick @ bottom_normal_local
        
        return bottom_normal_world, rpy

    def _calculate_tcp_pose_for_parallel_bottom(self, target_x, target_y, target_z, target_roll=0.0, target_pitch=0.0, target_yaw=0.0):
        """
        Calculate TCP pose to ensure brick bottom surface is strictly parallel to ground
        
        Args:
            target_x, target_y, target_z: TCP target position
            target_roll: Target roll angle for brick (default 0.0)
            target_pitch: Target pitch angle for brick (default 0.0)
            target_yaw: Target yaw angle for brick (default 0.0)
            
        Returns:
            TCP pose (xyz, rpy), ensuring brick bottom is parallel to ground
        """
        # Ground normal vector 
        ground_normal = np.array([0.0, 0.0, 1.0])
        
        # Get current brick bottom normal vector and complete orientation
        current_bottom_normal, current_brick_rpy = self._get_brick_bottom_normal()
        current_roll, current_pitch, current_yaw = current_brick_rpy
        
        print(f"[BRICK_ORIENTATION] Current brick RPY: Roll={np.degrees(current_roll):.2f}°, Pitch={np.degrees(current_pitch):.2f}°, Yaw={np.degrees(current_yaw):.2f}°")
        print(f"[BRICK_ORIENTATION] Target brick RPY: Roll={np.degrees(target_roll):.2f}°, Pitch={np.degrees(target_pitch):.2f}°, Yaw={np.degrees(target_yaw):.2f}°")
        print(f"[BRICK_ORIENTATION] Current bottom normal: {current_bottom_normal}")
        print(f"[BRICK_ORIENTATION] Target ground normal: {ground_normal}")
        
        # Target: brick bottom normal vector should point to [0, 0, -1] 
        target_bottom_normal = np.array([0.0, 0.0, -1.0])
        
        # Calculate alignment between current bottom and target
        alignment_dot = np.dot(current_bottom_normal, target_bottom_normal)
        alignment_angle = np.arccos(np.clip(alignment_dot, -1.0, 1.0))
        
        print(f"[BRICK_ORIENTATION] Bottom alignment dot product: {alignment_dot:.6f}")
        print(f"[BRICK_ORIENTATION] Alignment angle: {np.degrees(alignment_angle):.2f} degrees")
        
        roll_error = current_roll - target_roll  
        pitch_error = current_pitch - target_pitch  
        yaw_error = current_yaw - target_yaw
        
        print(f"[BRICK_ORIENTATION] Brick orientation errors:")
        print(f"[BRICK_ORIENTATION]   Roll error: {np.degrees(roll_error):.2f}° (target: {np.degrees(target_roll):.2f}°)")
        print(f"[BRICK_ORIENTATION]   Pitch error: {np.degrees(pitch_error):.2f}° (target: {np.degrees(target_pitch):.2f}°)")
        print(f"[BRICK_ORIENTATION]   Yaw error: {np.degrees(yaw_error):.2f}° (target: {np.degrees(target_yaw):.2f}°)")
        
        # Calculate target TCP pose
        current_tcp_pos, current_tcp_quat = self.rm.tcp_world_pose()
        current_tcp_rpy = p.getEulerFromQuaternion(current_tcp_quat)
        current_tcp_roll, current_tcp_pitch, current_tcp_yaw = current_tcp_rpy
        
        print(f"[TCP_ORIENTATION] Current TCP RPY: Roll={np.degrees(current_tcp_roll):.2f}°, Pitch={np.degrees(current_tcp_pitch):.2f}°, Yaw={np.degrees(current_tcp_yaw):.2f}°")
        
        if abs(target_roll) < 0.01 and abs(target_pitch) < 0.01:  # Target is horizontal placement
            # High-precision horizontal strategy: stronger correction for precise parallelism
            delta_roll = target_roll - current_roll   # Brick roll error
            delta_pitch = target_pitch - current_pitch # Brick pitch error
            delta_yaw = target_yaw - current_yaw       # Brick yaw error
            
            # Use stronger adjustment gain 
            target_tcp_roll = current_tcp_roll + delta_roll * 0.5   
            target_tcp_pitch = current_tcp_pitch + delta_pitch * 0.5
            target_tcp_yaw = current_tcp_yaw + delta_yaw * 0.3    
            
            # Relax angle limits (allow larger correction amplitude for precise parallelism)
            max_adjustment = 0.35  
            target_tcp_roll = np.clip(target_tcp_roll, current_tcp_roll - max_adjustment, current_tcp_roll + max_adjustment)
            target_tcp_pitch = np.clip(target_tcp_pitch, current_tcp_pitch - max_adjustment, current_tcp_pitch + max_adjustment)
            
            print(f"[TCP_STRATEGY] Using high-precision horizontal strategy (50% adjustment, ±{np.degrees(max_adjustment):.1f}° limit)")
        else:
            # For non-horizontal targets, use similar high-precision strategy
            delta_roll = target_roll - current_roll
            delta_pitch = target_pitch - current_pitch  
            delta_yaw = target_yaw - current_yaw
            
            # Increase adjustment gain
            target_tcp_roll = current_tcp_roll + delta_roll * 0.4   
            target_tcp_pitch = current_tcp_pitch + delta_pitch * 0.4 
            target_tcp_yaw = current_tcp_yaw + delta_yaw * 0.4       
            
            # angle limits
            max_adjustment = 0.3  
            target_tcp_roll = np.clip(target_tcp_roll, current_tcp_roll - max_adjustment, current_tcp_roll + max_adjustment)
            target_tcp_pitch = np.clip(target_tcp_pitch, current_tcp_pitch - max_adjustment, current_tcp_pitch + max_adjustment)
            
            print(f"[TCP_STRATEGY] Using high-precision proportional strategy (40% adjustment, ±{np.degrees(max_adjustment):.1f}° limit)")
        
        target_tcp_rpy = [target_tcp_roll, target_tcp_pitch, target_tcp_yaw]
        
        print(f"[TCP_TARGET] Target TCP RPY: Roll={np.degrees(target_tcp_roll):.2f}°, Pitch={np.degrees(target_tcp_pitch):.2f}°, Yaw={np.degrees(target_tcp_yaw):.2f}°")
        
        # Convert to quaternion
        target_tcp_quat = p.getQuaternionFromEuler(target_tcp_rpy)
        
        # Gimbal lock detection: record TCP pose changes
        tcp_rpy_change = [
            np.degrees(target_tcp_roll - current_tcp_roll),
            np.degrees(target_tcp_pitch - current_tcp_pitch), 
            np.degrees(target_tcp_yaw - current_tcp_yaw)
        ]
        print(f"[ROLL_DETECTION] TCP pose change: ΔRoll={tcp_rpy_change[0]:.1f}°, ΔPitch={tcp_rpy_change[1]:.1f}°, ΔYaw={tcp_rpy_change[2]:.1f}°")
        
        # Detect large rotations (possible gimbal lock signs)
        tcp_warn_deg = float(self.ocfg.get("tcp_change_warning_deg", 45.0))
        large_rotation = any(abs(change) > tcp_warn_deg for change in tcp_rpy_change)
        if large_rotation:
            print(f"[ROLL_WARNING] Large TCP rotation detected! May cause gimbal lock")
            print(f"[ROLL_WARNING] Change amount: {tcp_rpy_change}")
        
        return current_tcp_pos, target_tcp_quat

    def _move_tcp_with_brick_orientation_control(self, target_xyz, brick_orientation_control=None, target_yaw=None, description=""):
        """
        Move TCP to target position while ensuring brick bottom is strictly parallel to ground
        
        Args:
            target_xyz: Target position [x, y, z]
            brick_orientation_control: Brick orientation control parameter dict, contains:
                - target_roll: Target Roll angle (default 0.0)
                - target_pitch: Target Pitch angle (default 0.0) 
                - target_yaw: Target Yaw angle
                - roll_tolerance: Roll tolerance (default 2.0 degrees)
                - pitch_tolerance: Pitch tolerance (default 2.0 degrees)
                - yaw_tolerance: Yaw tolerance (default 3.0 degrees)
            target_yaw: Compatibility parameter, used if brick_orientation_control is None
        """
        print(f"[ORIENTATION_CONTROL] {description}")
        
        # Parse parameters
        if brick_orientation_control is not None:
            target_roll = brick_orientation_control.get('target_roll', 0.0)
            target_pitch = brick_orientation_control.get('target_pitch', 0.0) 
            target_yaw = brick_orientation_control.get('target_yaw', target_yaw or 0.0)
            # Stricter tolerance requirements for precise parallel control
            roll_tolerance = brick_orientation_control.get('roll_tolerance', 1.0)  
            pitch_tolerance = brick_orientation_control.get('pitch_tolerance', 1.0)  
            yaw_tolerance = brick_orientation_control.get('yaw_tolerance', 2.0)     
        else:
            target_roll = 0.0
            target_pitch = 0.0
            target_yaw = target_yaw or 0.0
            roll_tolerance = 1.0   
            pitch_tolerance = 1.0 
            yaw_tolerance = 2.0    
        
        print(f"[ORIENTATION_CONTROL] Target RPY: Roll={target_roll:.2f}°, Pitch={target_pitch:.2f}°, Yaw={target_yaw:.4f}rad")
        print(f"[ORIENTATION_CONTROL] Tolerances: Roll=±{roll_tolerance:.1f}°, Pitch=±{pitch_tolerance:.1f}°, Yaw=±{yaw_tolerance:.1f}°")
        
        # Calculate TCP pose ensuring brick bottom is parallel
        tcp_xyz, tcp_quat = self._calculate_tcp_pose_for_parallel_bottom(
            target_xyz[0], target_xyz[1], target_xyz[2], 
            target_roll, target_pitch, target_yaw
        )
        
        # Convert quaternion to RPY angles for _move_tcp function
        tcp_rpy = p.getEulerFromQuaternion(tcp_quat)
        
        # Record brick pose before execution
        if hasattr(self, 'brick_id') and self.brick_id is not None:
            brick_pos_before, brick_quat_before = p.getBasePositionAndOrientation(self.brick_id)
            brick_rpy_before = p.getEulerFromQuaternion(brick_quat_before)
            print(f"[ROLL_MONITOR] Brick RPY before execution: Roll={np.degrees(brick_rpy_before[0]):.1f}°, Pitch={np.degrees(brick_rpy_before[1]):.1f}°, Yaw={np.degrees(brick_rpy_before[2]):.1f}°")
        else:
            brick_rpy_before = None
            print(f"[ROLL_MONITOR] No brick ID, skipping brick pose monitoring")
        
        # Execute movement
        self._move_tcp(dict(xyz=tcp_xyz, rpy=tcp_rpy))
        
        # Check brick pose changes after execution
        if brick_rpy_before is not None:
            time.sleep(0.1)  # Let physics simulation stabilize
            brick_pos_after, brick_quat_after = p.getBasePositionAndOrientation(self.brick_id)
            brick_rpy_after = p.getEulerFromQuaternion(brick_quat_after)
            
            # Calculate brick pose changes
            brick_rpy_change = [
                np.degrees(brick_rpy_after[0] - brick_rpy_before[0]),
                np.degrees(brick_rpy_after[1] - brick_rpy_before[1]),
                np.degrees(brick_rpy_after[2] - brick_rpy_before[2])
            ]
            
            # Handle angle wrapping (-180° to +180°)
            for i in range(3):
                if brick_rpy_change[i] > 180:
                    brick_rpy_change[i] -= 360
                elif brick_rpy_change[i] < -180:
                    brick_rpy_change[i] += 360
            
            print(f"[ROLL_MONITOR] Brick RPY after execution: Roll={np.degrees(brick_rpy_after[0]):.1f}°, Pitch={np.degrees(brick_rpy_after[1]):.1f}°, Yaw={np.degrees(brick_rpy_after[2]):.1f}°")
            print(f"[ROLL_MONITOR] Brick pose change: ΔRoll={brick_rpy_change[0]:.1f}°, ΔPitch={brick_rpy_change[1]:.1f}°, ΔYaw={brick_rpy_change[2]:.1f}°")
            
            # Detect gimbal lock: if brick has large rotation
            roll_check_deg = float(self.ocfg.get("roll_check_threshold_deg", 30.0))
            large_brick_rotation = any(abs(change) > roll_check_deg for change in brick_rpy_change)
            if large_brick_rotation:
                print(f"[ROLL_ALERT]  Large brick rotation detected! Possible gimbal lock!")
                print(f"[ROLL_ALERT] Change details: ΔRoll={brick_rpy_change[0]:.1f}°, ΔPitch={brick_rpy_change[1]:.1f}°, ΔYaw={brick_rpy_change[2]:.1f}°")
            else:
                print(f"[ROLL_OK] Brick pose stable, no obvious gimbal lock")
        
        # Verify results - check all three angles
        bottom_normal_after, brick_rpy_after = self._get_brick_bottom_normal()
        brick_roll_after, brick_pitch_after, brick_yaw_after = brick_rpy_after
        
        ground_normal = np.array([0.0, 0.0, 1.0])
        target_bottom_normal = np.array([0.0, 0.0, -1.0])
        final_alignment = np.dot(bottom_normal_after, target_bottom_normal)
        
        print(f"[ORIENTATION_VERIFY] Final brick RPY: Roll={np.degrees(brick_roll_after):.2f}°, Pitch={np.degrees(brick_pitch_after):.2f}°, Yaw={np.degrees(brick_yaw_after):.2f}°")
        print(f"[ORIENTATION_VERIFY] Final bottom normal: {bottom_normal_after}")
        print(f"[ORIENTATION_VERIFY] Final bottom alignment: {final_alignment:.6f}")
        
        # Verify using passed tolerance parameters
        roll_error = abs(brick_roll_after - target_roll)
        pitch_error = abs(brick_pitch_after - target_pitch)
        yaw_error = abs(brick_yaw_after - target_yaw)
        if yaw_error > np.pi:
            yaw_error = 2*np.pi - yaw_error  # Handle angle wrapping across ±π
            
        roll_ok = roll_error < np.radians(roll_tolerance)
        pitch_ok = pitch_error < np.radians(pitch_tolerance)
        yaw_ok = yaw_error < np.radians(yaw_tolerance)
        
        align_thresh = float(self.ocfg.get("alignment_threshold", 0.95))
        alignment_ok = final_alignment > align_thresh
        
        print(f"[ORIENTATION_CHECK] Roll OK: {roll_ok} (error: {np.degrees(roll_error):.2f}°, tolerance: ±{roll_tolerance:.1f}°)")
        print(f"[ORIENTATION_CHECK] Pitch OK: {pitch_ok} (error: {np.degrees(pitch_error):.2f}°, tolerance: ±{pitch_tolerance:.1f}°)")
        print(f"[ORIENTATION_CHECK] Yaw OK: {yaw_ok} (error: {np.degrees(yaw_error):.2f}°, tolerance: ±{yaw_tolerance:.1f}°)")
        print(f"[ORIENTATION_CHECK] Alignment OK: {alignment_ok} (value: {final_alignment:.6f})")
        
        success = roll_ok and pitch_ok and yaw_ok and alignment_ok
        
        if success:
            print("[ORIENTATION_SUCCESS] Brick bottom is strictly parallel to ground with correct orientation")
        else:
            print(f"[ORIENTATION_WARNING] Brick orientation control incomplete!")
            if not roll_ok:
                print(f"[ORIENTATION_WARNING]   Roll error too large: {np.degrees(roll_error):.2f}° (tolerance: ±{roll_tolerance:.1f}°)")
            if not pitch_ok:
                print(f"[ORIENTATION_WARNING]   Pitch error too large: {np.degrees(pitch_error):.2f}° (tolerance: ±{pitch_tolerance:.1f}°)")
            if not yaw_ok:
                print(f"[ORIENTATION_WARNING]   Yaw error too large: {np.degrees(yaw_error):.2f}° (tolerance: ±{yaw_tolerance:.1f}°)")
            if not alignment_ok:
                print(f"[ORIENTATION_WARNING]   Bottom alignment insufficient: {final_alignment:.6f}")
        
        return success

    def _measured_finger_ex(self):
        v = self.rm.finger_axis_world()
        return None if v is None else v

    def _arm_down_ok(self, cos_thresh=0.95):
        cosOri, cosGeo = self.vf.cos_down_metrics()
        allow_geo = bool(self.vcfg.get("allow_geo_as_fallback", True))
        ok = (cosOri >= cos_thresh) or (allow_geo and (cosGeo >= cos_thresh))
        return ok, cosOri, cosGeo

    def _force_z_down_at(self, x, y, z, yaw_hint):
        cos_down_thr = float(self.vcfg.get("tcp_z_down_cos_thresh", 0.95))
        cos_ex_thr   = float(self.vcfg.get("tcp_ex_align_cos_thresh", 0.92))
        allow_tcp_ex_fallback = bool(self.vcfg.get("allow_tcp_ex_fallback", True))

        def _ok_ex_alignment(yb, ex_meas, ex_tcp):
            cos_m = None
            cos_t = float(abs(np.dot(ex_tcp, yb)))
            if ex_meas is not None:
                cos_m = float(abs(np.dot(ex_meas, yb)))
                if cos_m >= cos_ex_thr: return True, cos_m, cos_t, "meas"
                if allow_tcp_ex_fallback and (cos_t >= cos_ex_thr): return True, cos_m, cos_t, "tcp"
                return False, cos_m, cos_t, "meas"
            else:
                return (cos_t >= cos_ex_thr), None, cos_t, "tcp"

        def _try_pose(yaw, dr=0.0, dp=0.0, draw=True):
            base = align_topdown_to_brick(yaw)
            rpy  = [base[0] + dr, base[1] + dp, base[2]]
            self._move_tcp(dict(xyz=[x,y,z], rpy=rpy))
            if draw: self.viz.draw_down_at_tcp()

            okDown, cosDownOri, cosDownGeo = self._arm_down_ok(cos_down_thr)

            ex_meas = self._measured_finger_ex()
            _, orn  = self.rm.tcp_world_pose()
            R       = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            ex_tcp  = R[:,0]
            cy, sy  = math.cos(yaw), math.sin(yaw)
            yb      = np.array([-sy, cy, 0.0])
            okEX, cosEX_m, cosEX_t, src = _ok_ex_alignment(yb, ex_meas, ex_tcp)

            print(f"[ORI] yaw={yaw:+.3f} dr={dr:+.3f} dp={dp:+.3f}  "
                  f"down(ori)={cosDownOri:.3f}|geo={cosDownGeo:.3f}  "
                  f"cosEX(m)={cosEX_m if cosEX_m is not None else 'NA'} cosEX(t)={cosEX_t:.3f} src={src} "
                  f"-> okDown={okDown} okEX={okEX}")
            return okDown and okEX

        quad0 = self.yaw_quadrant % 4
        yaw_seq = [yaw_hint + ((quad0+i) % 4) * math.pi/2 for i in range(4)]
        roll_grid  = [self.rpy_bias[0], 0.0,  math.pi/2, -math.pi/2,  math.pi/6, -math.pi/6]
        pitch_grid = [self.rpy_bias[1], 0.0,  math.pi/12, -math.pi/12]

        for yi, yaw in enumerate(yaw_seq):
            if _try_pose(yaw, self.rpy_bias[0], self.rpy_bias[1]):
                self.yaw_quadrant = (quad0 + yi) % 4
                return True
            for dr in roll_grid:
                for dp in pitch_grid:
                    if _try_pose(yaw, dr, dp):
                        self.rpy_bias = [dr, dp]
                        self.yaw_quadrant = (quad0 + yi) % 4
                        return True

        yaw = yaw_hint + (self.yaw_quadrant % 4) * (math.pi/2)
        dr, dp = self.rpy_bias[0], self.rpy_bias[1]
        step_deg_schedule = [30, 20, 12, 8, 5, 3, 2, 1]
        for step_deg in step_deg_schedule:
            step = math.radians(step_deg)
            for _ in range(6):
                base = align_topdown_to_brick(yaw)
                rpy  = [base[0] + dr, base[1] + dp, base[2]]
                self._move_tcp(dict(xyz=[x, y, z], rpy=rpy))
                self.viz.draw_down_at_tcp()

                ex_meas = self._measured_finger_ex()
                _, orn  = self.rm.tcp_world_pose()
                R       = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
                ex_tcp  = R[:,0]
                cy, sy  = math.cos(yaw), math.sin(yaw)
                yb      = np.array([-sy, cy, 0.0])

                ex_vec  = ex_tcp if (ex_meas is None) else ex_meas
                if ex_vec is None:
                    okDown, _c1, _c2 = self._arm_down_ok(cos_down_thr)
                    if okDown: return True
                    else: break

                cross_z = float(np.cross(ex_vec, yb)[2])
                dot_xy  = float(np.dot(ex_vec[:2], yb[:2]))
                ang_err = math.atan2(cross_z, dot_xy)
                yaw = yaw + math.copysign(step, ang_err)

                okDown, cosDownOri, cosDownGeo = self._arm_down_ok(cos_down_thr)
                cosEX_m = float(abs(np.dot(ex_meas, yb))) if ex_meas is not None else None
                cosEX_t = float(abs(np.dot(ex_tcp, yb)))
                print(f"[MEAS] yaw={yaw:+.3f}  cosEX(m)={cosEX_m if ex_meas is not None else 'NA'}  "
                      f"cosEX(t)={cosEX_t:.3f}  down(o)={cosDownOri:.3f}|g={cosDownGeo:.3f}")

                if ((cosEX_m if ex_meas is not None else cosEX_t) >= float(self.vcfg.get("tcp_ex_align_cos_thresh", 0.92))) and okDown:
                    self.rpy_bias = [dr, dp]
                    self.yaw_quadrant = (int(round((yaw - yaw_hint) / (math.pi/2))) % 4)
                    return True

        return False

    # Reset
    def reset_between_tasks(self):
        self.viz.banner("[phase] reset_between_tasks")
        # End-effector pose facing down: world -Z, yaw=0
        home_xyz = self.env.cfg.get("home_pose_xyz", [0.55, 0.00, 0.55])
        rpy = align_topdown_to_brick(0.0)
        self._move_tcp(dict(xyz=[home_xyz[0], home_xyz[1], home_xyz[2]], rpy=rpy))
        # Clear records
        self.rpy_bias = [0.0, 0.0]
        self.yaw_quadrant = 0
        # Release any remaining constraints + zero gripper angles + open
        if self.gripper.is_attached():
            self.gripper.detach()
            self._wait(self.timing.get("reattach_wait_sec", 0.05))
        self.rm.set_finger_sym_angles(0.0)
        self._wait(self.timing.get("gripper_open_wait_sec", 0.15))
        self.gripper.open()
        self.viz.snap("reset@ready")




    # ------------ Main FSM ------------
    def execute_fsm(self, wps, aux, assist_cfg, brick_id, ground_id, support_ids=None):
        if not self.rm.tcp_calibrated:
            self.rm.calibrate_tcp()

        if hasattr(self, 'llm_agent') and self.llm_agent is not None:
            if hasattr(self.llm_agent, 'mode') and self.llm_agent.mode == "single_agent":
                self.llm_agent.reset_cache()
                print(f"[LLM] Single-Agent mode: Cache reset (brick #{brick_id})")

        if support_ids is None:
            support_ids = [ground_id]

        ground_z     = aux["ground_z"]
        finger_depth = float(self.gcfg.get("finger_depth", 0.065))
        approach     = float(self.ccfg.get("approach", 0.08))
        lift_clear   = float(self.ccfg.get("lift", 0.15))
        width_pad    = float(self.ccfg.get("width_pad", 0.010))
        tip          = float(self.gcfg.get("tip_length_guess", 0.16))

        gx, gy = wps[4]["pose"]["xyz"][0], wps[4]["pose"]["xyz"][1]
        gz_top = aux["min_pre_place"] - approach      # Target top surface height
        yaw_place = wps[4]["pose"]["rpy"][2]
        W = float(aux["W"])
        
        restarts_left = self.max_restarts
        while restarts_left >= 0:
            # pre_grasp
            llm_pose = None
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    print("[LLM] Starting LLM-based pre_grasp planning...")
                    llm_result = self.llm_handler.plan_pre_grasp_pose(wps, aux)
                    llm_pose = llm_result["pose"]
                    print(f"[LLM] Using LLM planned pose: xyz=({llm_pose['xyz'][0]:.6f}, {llm_pose['xyz'][1]:.6f}, {llm_pose['xyz'][2]:.6f})")
                except Exception as e:
                    print(f"[LLM] LLM planning failed, using classical approach: {e}")
                    llm_pose = None
            
            for attempt in range(self.retries+1):
                self.viz.banner("[phase] pre_grasp (TCP above CoM)")
                bx, by, bz, byaw = self._brick_xy_yaw()
                z_top = bz + aux["H"]/2

                # Use LLM planned position or classical geometric position
                if llm_pose is not None:
                    target_x, target_y, target_z = llm_pose["xyz"]
                    self._force_z_down_at(target_x, target_y, target_z, byaw)
                else:
                    self._force_z_down_at(bx, by, z_top + approach + tip, byaw)
                self.viz.draw_brick_width_at_center(bx, by, bz, byaw)
                self.viz.snap("pre_grasp@align")

                tip_meas = self.rm.estimate_tip_length_from_tcp(fallback=tip)
                if abs(tip_meas - tip) > 0.001:
                    tip = tip_meas
                    print(f"[TIP] pre_grasp measured tip_finger={tip:.3f}")
                    if llm_pose is not None:
                        self._force_z_down_at(target_x, target_y, target_z, byaw)
                    else:
                        self._force_z_down_at(bx, by, z_top + approach + tip, byaw)
                    self.viz.snap("pre_grasp@re-align(tip)")

                tol_xy = float(self.vcfg.get("tol_xy", 0.006))
                for _ in range(8):
                    tx, ty, tz = self.vf.tcp_pos()
                    dx, dy = bx - tx, by - ty
                    err = math.hypot(dx, dy)
                    if err <= tol_xy: break
                    step = min(max(err*0.6, 0.003), 0.02)
                    target_z_for_xy = target_z if llm_pose is not None else (z_top + approach + tip)
                    self._force_z_down_at(tx + dx/err*step, ty + dy/err*step, target_z_for_xy, byaw)
                self.viz.snap("pre_grasp@xy-ok")

                if self.vf.check_pre_grasp(z_top, tip, approach, brick_xy=(bx,by), brick_yaw=byaw):
                    break
                if attempt < self.retries:
                    self.viz.banner("[phase] pre_grasp retry (+1cm)")
                    retry_z = (target_z + 0.01) if llm_pose is not None else (z_top + approach + tip + 0.01)
                    retry_x = target_x if llm_pose is not None else bx
                    retry_y = target_y if llm_pose is not None else by
                    self._force_z_down_at(retry_x, retry_y, retry_z, byaw)
                else:
                    break

            if not self.vf.check_pre_grasp(z_top, tip, approach, brick_xy=(bx,by), brick_yaw=byaw):
                restarts_left -= 1
                continue

            # descend
            llm_descend = None
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    print("[LLM] Starting LLM-based descend planning...")
                    llm_descend = self.llm_handler.plan_descend(wps, aux, W, ground_z)
                    if llm_descend is not None:
                        print(f"[LLM] Using LLM planned descend: gap={llm_descend['target_gap']:.6f}m")
                except Exception as e:
                    print(f"[LLM] LLM descend planning failed, using classical approach: {e}")
                    llm_descend = None

            for attempt in range(self.retries+1):
                self.viz.banner("[phase] descend (open to width, then down)")
                bx, by, bz, byaw = self._brick_xy_yaw()

                # Use LLM planned gap or classical calculation
                if llm_descend is not None:
                    required_gap = llm_descend['target_gap']
                    target_tcp_x, target_tcp_y, target_tcp_z = llm_descend['pose']['xyz']
                else:
                    required_gap = W + self.open_clearance
                    target_tcp_x, target_tcp_y = bx, by
                    desired_bottom = max(
                        ground_z + float(self.gcfg.get("ground_margin", 0.003)),
                        bz - finger_depth/2
                    )
                    target_tcp_z = desired_bottom + tip

                ok_gap, theta, gap, meta = self.gripper.open_to_gap(
                    required_gap, tol=0.001, iters=18, settle_sec=0.15, verbose=True
                )
                self.viz.snap("descend@after-open")
                if not ok_gap:
                    print(f"[WARN] gap still < target (gap={gap:.3f} < {required_gap:.3f}); "
                          f"theta={theta:.3f} rad, pair={meta.get('pair')}, signs={meta.get('signs')} "
                          f"→ will try to descend carefully (may contact).")

                print(f"[DESCEND_PLAN] W={W:.3f} need_gap={required_gap:.3f} got_gap={gap:.3f} "
                      f"→ target_tcp=({target_tcp_x:.3f}, {target_tcp_y:.3f}, {target_tcp_z:.3f})")

                ok = self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z, byaw)
                self.viz.snap("descend@at-target")
                
                # Calculate desired_bottom for state recording
                if llm_descend is not None:
                    desired_bottom = target_tcp_z - tip
                else:
                    desired_bottom = max(
                        ground_z + float(self.gcfg.get("ground_margin", 0.003)),
                        bz - finger_depth/2
                    )
                
                if ok and self.vf.check_descend_grasp(desired_bottom, tip, ground_id, brick_xy=(bx,by)):
                    break

                if attempt < self.retries:
                    self.viz.banner("[phase] descend -> back to pre_grasp]")
                    self._force_z_down_at(bx, by, (bz + aux["H"]/2) + approach + tip + 0.02, byaw)
                else:
                    return False

            if restarts_left < 0: return False
            
            # Calculate final desired_bottom for verification
            final_desired_bottom = target_tcp_z - tip if llm_descend is not None else max(
                ground_z + float(self.gcfg.get("ground_margin", 0.003)),
                bz - finger_depth/2
            )
            
            if not self.vf.check_descend_grasp(final_desired_bottom, tip, ground_id, brick_xy=(bx,by), brick_yaw=byaw):
                continue

            # close
            llm_close = None
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    print("[LLM] Starting LLM-based close planning...")
                    context = {
                        "world_physical_state": wps[0],
                        "tcp": {"xyz": self.vf.tcp_pos()},
                        "brick": {"position": [bx, by, bz], "width": W}
                    }
                    llm_close = self.llm_agent.plan_close(context, attempt_idx=0)
                    print(f"[LLM] Using LLM planned close: {llm_close['gripper_command']['action_type']}")
                except Exception as e:
                    print(f"[LLM] LLM close planning failed, using classical approach: {e}")
                    llm_close = None

            for attempt in range(self.retries+1):
                self.viz.banner("[phase] close")
                
                # Use LLM planned TCP adjustment
                if llm_close and llm_close['tcp_adjustment']['enabled']:
                    offset = llm_close['tcp_adjustment']['position_offset']
                    tx, ty, tz = self.vf.tcp_pos()
                    self._force_z_down_at(tx + offset[0], ty + offset[1], tz + offset[2], byaw)
                    print(f"[LLM_CLOSE] Applied TCP adjustment: {offset}")
                
                self.gripper.close()
                self._wait(self.timing.get("gripper_close_wait_sec", 0.25))
                
                # Use LLM planned contact assist strategy
                if assist_cfg.get("enabled", False):
                    if llm_close and llm_close['attachment_strategy']['use_contact_assist']:
                        threshold = llm_close['attachment_strategy']['contact_threshold']
                        print(f"[LLM_CLOSE] Using LLM contact threshold: {threshold:.1f}N")
                    self.gripper.try_attach_with_contact(brick_id, assist_cfg)
                self.viz.snap("close@after-close")

                if self.vf.check_close(assist_cfg):
                    break
                if attempt < self.retries:
                    bx, by, bz, byaw = self._brick_xy_yaw()
                    tx, ty, tz = self.vf.tcp_pos()
                    dx, dy = bx - tx, by - ty
                    n = math.hypot(dx, dy) + 1e-9
                    step = self.recentre_step
                    self.gripper.open()
                    self._force_z_down_at(tx + dx/n*step, ty + dy/n*step, tz, byaw)
                    self.gripper.close()
                    self._wait(0.2)
                    if assist_cfg.get("enabled", False):
                        self.gripper.try_attach_with_contact(brick_id, assist_cfg)
                else:
                    restarts_left -= 1
                    break
            if restarts_left < 0: return False
            if not self.vf.check_close(assist_cfg):
                continue

            # lift
            llm_lift = None
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    print("[LLM] Starting LLM-based lift planning...")
                    llm_lift = self.llm_handler.plan_lift(wps, aux, lift_clear, tip)
                    if llm_lift is not None:
                        print(f"[LLM] Using LLM planned lift: height={llm_lift['lift_height']:.6f}m")
                except Exception as e:
                    print(f"[LLM] LLM lift planning failed, using classical approach: {e}")
                    llm_lift = None

            self.vf.mark_before_lift()
            self.viz.banner("[phase] lift")
            bx, by, bz, byaw = self._brick_xy_yaw()
            z_top = bz + aux["H"]/2

            # Use LLM planned lift strategy or classical strategy
            if llm_lift is not None:
                target_tcp_x, target_tcp_y, target_tcp_z = llm_lift['pose']['xyz']
                lift_height = llm_lift['lift_height']
                stability_control = llm_lift.get('stability_control', {})
                fallback_strategy = llm_lift.get('fallback_strategy', {})
                
                # Execute LLM planned lift
                self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z, byaw)
                
                # Apply stability control
                if stability_control.get('verify_attachment', True):
                    print("[LLM_LIFT] Verifying attachment during lift...")
                    if not self.gripper.is_attached():
                        print("[LLM_LIFT] Warning: Attachment lost during lift")
                
            else:
                # Use classical lift strategy
                self._force_z_down_at(bx, by, z_top + lift_clear + tip, byaw)
            
            self.viz.snap("lift@at-height")

            
            if not self.vf.check_lift():
                # Use LLM fallback strategy or classical fallback
                if llm_lift is not None and llm_lift.get('fallback_strategy', {}).get('re_attach_on_failure', True):
                    print("[LLM_LIFT] Using LLM fallback: re-attachment strategy")
                    max_retries = llm_lift['fallback_strategy'].get('max_retries', 2)
                    
                    for retry in range(max_retries):
                        if assist_cfg.get("enabled", False) and (not self.gripper.is_attached()):
                            print(f"[LLM_LIFT] Re-attachment attempt {retry + 1}/{max_retries}")
                            self.gripper.try_attach_with_contact(brick_id, {"contact_gate": False})
                            self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z, byaw)
                            if self.vf.check_lift():
                                break
                    
                    if not self.vf.check_lift():
                        print("[LLM_LIFT] LLM fallback strategies exhausted")
                        restarts_left -= 1
                        continue
                else:
                    if assist_cfg.get("enabled", False) and (not self.gripper.is_attached()):
                        self.gripper.try_attach_with_contact(brick_id, {"contact_gate": False})
                        self._force_z_down_at(bx, by, z_top + lift_clear + tip, byaw)
                        if not self.vf.check_lift():
                            restarts_left -= 1
                            continue
                    else:
                        restarts_left -= 1
                        continue

            # pre_place
            llm_place_result = None
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    print("[LLM] Starting unified LLM-based place planning (pre_place + descend_place)...")
                    
                    # Build complete context
                    context = self.llm_handler.gather_llm_context(wps, aux)
                    context.update({
                        "goal": {
                            "target_xy": [gx, gy],
                            "target_z_top": gz_top,
                            "target_yaw": yaw_place,
                            "support_ids": support_ids,
                            "phase": "complete_place"  # Mark as complete place planning
                        }
                    })
                    
                    # Plan entire place phase in one go
                    llm_place_result = self.llm_agent.plan_place(context, 0, None)
                    if llm_place_result is not None:
                        print(f"[LLM] Successfully planned unified place strategy")
                        print(f"[LLM] Approach phase: {llm_place_result.get('approach_phase', {}).get('strategy', 'N/A')}")
                        print(f"[LLM] Descent phase: {llm_place_result.get('descent_phase', {}).get('action_type', 'N/A')}")
                        print(f"[LLM] Brick orientation control: {llm_place_result.get('brick_orientation_control', {})}")
                    else:
                        print("[LLM] LLM unified place planning returned None")
                except Exception as e:
                    print(f"[LLM] LLM unified place planning failed, using classical approach: {e}")
                    llm_place_result = None
            else:
                llm_place_result = None

            self.viz.banner("[phase] pre_place")
            
            # Use LLM planned position and brick orientation control
            if llm_place_result is not None:
                approach_phase = llm_place_result.get('approach_phase', {})
                brick_orientation = llm_place_result.get('brick_orientation_control', {})
                
                target_tcp_x, target_tcp_y, target_tcp_z = approach_phase['target_pose']['xyz']
                target_rpy = approach_phase['target_pose']['rpy']
                approach_height = approach_phase.get('approach_height', approach)
                
                # Use brick orientation control method
                if brick_orientation.get('bottom_parallel_to_ground', True):
                    success = self._move_tcp_with_brick_orientation_control(
                        [target_tcp_x, target_tcp_y, target_tcp_z], 
                        brick_orientation_control=brick_orientation,
                        description="Pre-place with LLM brick orientation control"
                    )
                    if not success:
                        print("[WARN] Brick orientation control failed, using traditional method")
                        self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z, target_rpy[2])
                else:
                    self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z, target_rpy[2])
            else:
                classic_orientation_control = {
                    'target_roll': 0.0,
                    'target_pitch': 0.0,
                    'target_yaw': yaw_place,
                    'bottom_parallel_to_ground': True
                }
                success = self._move_tcp_with_brick_orientation_control(
                    [gx, gy, gz_top + approach + tip], 
                    brick_orientation_control=classic_orientation_control,
                    description="Classical pre-place with brick orientation control"
                )
                if not success:
                    print("[WARN] Brick orientation control failed, using traditional force_z_down_at")
                    self._force_z_down_at(gx, gy, gz_top + approach + tip, yaw_place)
            
            self.viz.snap("pre_place@above")

            
            # Verification and fallback mechanism
            if not self.vf.check_pre_place(gz_top, tip, approach, goal_xy=(gx,gy), goal_yaw=yaw_place):
                if llm_place_result is not None:
                    # Use LLM fallback strategy, but maintain brick orientation control
                    approach_phase = llm_place_result.get('approach_phase', {})
                    fallback_strategy = approach_phase.get('fallback_strategy', {})
                    height_adjustment = fallback_strategy.get('height_adjustment', 0.02)
                    print(f"[LLM_PRE_PLACE] Using LLM fallback: height adjustment +{height_adjustment:.3f}m")
                    
                    target_tcp_x, target_tcp_y, target_tcp_z = approach_phase['target_pose']['xyz']
                    brick_orientation = llm_place_result.get('brick_orientation_control', {})
                    target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                    
                    # Use brick orientation control for fallback
                    success = self._move_tcp_with_brick_orientation_control(
                        [target_tcp_x, target_tcp_y, target_tcp_z + height_adjustment], 
                        brick_orientation_control=brick_orientation,
                        description="LLM fallback with brick orientation control"
                    )
                    if not success:
                        target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                        self._force_z_down_at(target_tcp_x, target_tcp_y, target_tcp_z + height_adjustment, target_yaw_for_brick)
                else:
                    # Classical fallback strategy, also use brick orientation control
                    classic_orientation_control = {
                        'target_roll': 0.0,
                        'target_pitch': 0.0,
                        'target_yaw': yaw_place,
                        'bottom_parallel_to_ground': True
                    }
                    success = self._move_tcp_with_brick_orientation_control(
                        [gx, gy, gz_top + approach + tip + 0.02], 
                        brick_orientation_control=classic_orientation_control,
                        description="Classical fallback with brick orientation control"
                    )
                    if not success:
                        self._force_z_down_at(gx, gy, gz_top + approach + tip + 0.02, yaw_place)
                self.viz.snap("pre_place@above(+adjustment)")


            # descend_place 
            if hasattr(self, 'llm_agent') and self.llm_agent is not None:
                try:
                    # Use first LLM call result, don't call LLM again
                    # llm_place_result = self._llm_plan_place_fusion(wps, aux, gx, gy, gz_top, yaw_place, approach, tip, support_ids, "descend_place", brick_id) 
                    print(f"[LLM] Reusing existing place planning result for descend_place phase")
                    if llm_place_result is not None:
                        descent_phase = llm_place_result.get('descent_phase', {})
                        print(f"[LLM] Using existing LLM planned descent phase:")
                        if descent_phase:
                            print(f"[LLM] Start height: {descent_phase.get('start_height', 'default')}")
                            print(f"[LLM] Step size: {descent_phase.get('step_size', 'default')}")
                            print(f"[LLM] Max range: {descent_phase.get('max_descent_range', 'default')}")
                    else:
                        print("[INFO] No LLM planning result available, will use classical approach")
                except Exception as e:
                    print(f"[INFO] Could not access existing LLM place result: {e}")
                    print("[INFO] Will use classical approach for descend_place")

            self.viz.banner("[phase] descend_place")
            self.gripper.close()

            # Baseline mode
            if self.baseline_enabled and self.skip_collision_detection:
                print("[BASELINE_PLACE] Using simplified placement - direct to target position (no collision detection)")
                
                # Move directly to target position (brick bottom + tip offset)
                target_z = gz_top + tip  # Direct to brick top position
                target_yaw_for_place = yaw_place
                
                print(f"[BASELINE_PLACE] Moving directly to target: ({gx:.6f}, {gy:.6f}, {target_z:.6f})")
                self._force_z_down_at(gx, gy, target_z, target_yaw_for_place)
                
                self.viz.snap("place@direct_target")
                print("[BASELINE_PLACE] Reached target position directly - no collision detection performed")
                

                touched_sid = support_ids_planned[0] if 'support_ids_planned' in locals() and support_ids_planned else support_ids[0]
            else:
                if llm_place_result is not None:
                    descent_phase = llm_place_result.get('descent_phase', {})
                    llm_orientation_control = descent_phase.get('orientation_control', {})
                    brick_orientation = llm_place_result.get('brick_orientation_control', {})
                    
                    if llm_orientation_control:
                        target_rpy = llm_orientation_control.get('target_rpy', [0.0, 0.0, yaw_place])
                        brick_orientation = {
                            "control_all_angles": True,
                            "target_roll": target_rpy[0],
                            "target_pitch": target_rpy[1], 
                            "target_yaw": target_rpy[2],
                "roll_tolerance": llm_orientation_control.get('roll_tolerance', 2.0),
                            "pitch_tolerance": llm_orientation_control.get('pitch_tolerance', 2.0),
                            "yaw_tolerance": llm_orientation_control.get('yaw_tolerance', 5.0),
                            "strict_parallel": llm_orientation_control.get('strict_parallel', True),
                            "bottom_parallel_to_ground": True,
                            "description": llm_orientation_control.get('description', 'LLM planned orientation control')
                        }
                    
                    start_bottom = descent_phase.get('start_height', gz_top + max(self.release_above, 0.006))
                    dz = descent_phase.get('step_size', 0.004)
                    max_extra = descent_phase.get('max_descent_range', 0.030)
                    support_ids_planned = llm_place_result.get('contact_detection', {}).get('support_ids', support_ids)
                    
                else:
                    start_bottom = gz_top + max(self.release_above, 0.006)  # Above top surface
                    dz = 0.004
                    max_extra = 0.030
                    support_ids_planned = support_ids
                    brick_orientation = {
                        "control_all_angles": True,
                        "target_roll": 0.0,
                        "target_pitch": 0.0,
                        "target_yaw": yaw_place,
                        "bottom_parallel_to_ground": True
                    }
                    

                # Initial descent position, use brick orientation control
                if brick_orientation.get('bottom_parallel_to_ground', True):
                    success = self._move_tcp_with_brick_orientation_control(
                        [gx, gy, start_bottom + tip], 
                        brick_orientation_control=brick_orientation,
                        description="Initial descend position with brick orientation control"
                    )
                    if not success:
                        print("[WARN] Initial brick orientation control failed, using traditional method")
                        target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                        self._force_z_down_at(gx, gy, start_bottom + tip, target_yaw_for_brick)
                else:
                    target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                    self._force_z_down_at(gx, gy, start_bottom + tip, target_yaw_for_brick)
                
                self.viz.snap("place@pre-support")

                steps = int(max_extra / dz) + 1
                touched_sid = None
                for i in range(steps):
                    z_next = start_bottom - (i+1)*dz
                    
                    # Maintain brick orientation control at each descent step
                    if brick_orientation.get('bottom_parallel_to_ground', True):
                        success = self._move_tcp_with_brick_orientation_control(
                            [gx, gy, z_next + tip], 
                            brick_orientation_control=brick_orientation,
                            description=f"Descend step {i+1}/{steps} with brick orientation control"
                        )
                        if not success:
                            print(f"[WARN] Brick orientation control failed at step {i+1}, using traditional method")
                            target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                            self._force_z_down_at(gx, gy, z_next + tip, target_yaw_for_brick)
                    else:
                        target_yaw_for_brick = brick_orientation.get('target_yaw', yaw_place)
                        self._force_z_down_at(gx, gy, z_next + tip, target_yaw_for_brick)
                    
                    # Check support surface contact
                    for sid in support_ids_planned:
                        if self.vf.brick_touching_body(sid):
                            touched_sid = sid
                            break
                    if touched_sid is not None:
                        break

                if touched_sid is not None:
                    print(f"[PLACE] touchdown on support body id={touched_sid}.")
                    self.viz.snap("place@touchdown")
                else:
                    print("[WARN] touchdown not detected within margin; proceed to release anyway.")
                    self.viz.snap("place@pre-release")

            # release
            self.viz.banner("[phase] release (open_to_gap to drop)")
            if assist_cfg.get("enabled", False) and self.gripper.is_attached():
                self.gripper.detach()
                self._wait(self.timing.get("reattach_wait_sec", 0.05))

            #Baseline mode
            if self.baseline_enabled and self.skip_force_feedback:
                print("[BASELINE_RELEASE] Using simplified release - direct gripper opening (no force feedback)")
                
                # Use fixed parameters for direct release
                target_gap = W + self.open_clearance + self.direct_release_gap
                
                print(f"[BASELINE_RELEASE] Direct release: gap={target_gap:.3f}m")
                
                # Directly open gripper to target gap
                ok_gap, theta, gap, meta = self.gripper.open_to_gap(
                    target_gap, tol=0.001, iters=18, settle_sec=0.12, verbose=True
                )
                self.viz.snap("place@baseline_release")
                self._wait(0.1)
                
                # Force detach constraint
                if self.gripper.is_attached():
                    self.gripper.detach()
                    self._wait(0.02)
                
                print("[BASELINE_RELEASE] Simplified release completed")
            else:
                release_params = self.llm_handler.plan_release(
                    wps, aux, gx, gy, gz_top, W, self.release_above, touched_sid
                )
                
                # Use LLM planned parameters, if LLM fails use default values
                if release_params:
                    extra = release_params.get("extra_clearance", float(self.release_open_extra))
                    tol = release_params.get("tolerance", 0.001)
                    increment_step = release_params.get("increment_step", 0.5e-3)
                    max_retries = release_params.get("max_retries", 8)
                else:
                    extra = float(self.release_open_extra)
                    tol = 0.001
                    increment_step = 0.5e-3
                    max_retries = 8
                
                target_gap = W + self.open_clearance + extra
                print(f"[RELEASE] LLM-planned quick sync release: gap={target_gap:.3f}, extra={extra:.3f}")
                

                if self.use_quick_release and hasattr(self.gripper, 'quick_sync_release'):
                    quick_success = self.gripper.quick_sync_release(target_gap, settle_sec=0.08)
                    self.viz.snap("place@quick_sync_release")
                    self._wait(0.1)  
                    

                    contacts = self.vf.finger_contacts_with(brick_id)
                    if (not self.gripper.is_attached()) and (contacts == 0):
                        print("[PLACE] Quick sync release successful!")
                    else:
                        print(f"[PLACE] Quick sync failed, fallback to gradual release")
                        quick_success = False
                else:
                    quick_success = False
                

                if not quick_success:
                    force_success = False
                    if self.use_force_feedback and hasattr(self.gripper, 'force_feedback_release'):
                        print("[PLACE] Attempting force feedback release...")
                        try:
                            # Use enhanced angle control release (only in non-baseline mode)
                            if self.angle_optimizer is not None:
                                print("[PLACE] Using enhanced release with angle control...")
                                tcp_pos, _ = self.rm.tcp_world_pose()
                                force_success, angle_error, optimization_data = self.angle_optimizer.enhanced_release_with_angle_control(
                                    brick_id, yaw_place, self.gripper, tcp_pos, self.vf
                                )
                            else:
                                force_success, final_gap = self.gripper.force_feedback_release(brick_id, self.vf)
                            
                            if force_success:
                                print("[PLACE] Force feedback release successful!")
                            else:
                                print("[PLACE] Force feedback release failed, fallback to gradual")
                        except Exception as e:
                            print(f"[PLACE] Force feedback error: {e}, using gradual release")
                            force_success = False
                    
                    # If force feedback also failed, use LLM planned gradual release
                    if not force_success:
                        print(f"[RELEASE] Using LLM-planned gradual release: max_retries={max_retries}, increment={increment_step:.3f}")
                        for k in range(max_retries):
                            target_gap = W + self.open_clearance + extra + k * increment_step
                            print(f"[RELEASE] Gradual attempt {k+1}/{max_retries}: gap={target_gap:.3f}")
                            ok_gap, theta, gap, meta = self.gripper.open_to_gap(
                                target_gap, tol=tol, iters=18, settle_sec=0.12, verbose=True
                            )
                            self.viz.snap(f"place@open_to_gap(g={target_gap:.3f})")
                            self._wait(self.timing.get("release_wait_sec", 0.06))

                            # Dual criteria: no constraint + zero finger contact
                            if self.gripper.is_attached():
                                self.gripper.detach()
                                self._wait(0.02)
                            contacts = self.vf.finger_contacts_with(brick_id)
                            if (not self.gripper.is_attached()) and (contacts == 0):
                                print(f"[PLACE] LLM-planned gradual release successful at attempt {k+1}: no contacts and no constraint.")
                                break

                            self._force_z_down_at(gx, gy, (gz_top + self.release_above) + tip + 0.010, yaw_place)
                            self._wait(self.timing.get("release_wait_sec", 0.06))
                            extra = min(extra + self.release_retry_extra, self.release_max_extra)

                # Final confirmation after release
                if self.vf.finger_contacts_with(brick_id) > 0:
                    print("[WARN] still contacting after release loop; final widen+up.")
                    self.gripper.open_to_gap(W + self.open_clearance + self.release_max_extra,
                                             tol=0.001, iters=12, settle_sec=0.10, verbose=False)
                    self._force_z_down_at(gx, gy, gz_top + self.release_above + tip + 0.015, yaw_place)
                    self._wait(self.timing.get("final_widen_wait_sec", 0.08))
            
            # retreat
            self.viz.banner("[phase] retreat")
            self._force_z_down_at(gx, gy, gz_top + approach + tip, yaw_place)
            self.viz.snap("retreat@done")
            
            return True

        # If all retries failed
        return False
