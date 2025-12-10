import pybullet as p
import numpy as np
from typing import Any, Dict, Optional
from modules.llm_planner import LLMPlanner, OpenAIChatClient
from math3d.transforms import align_topdown_to_brick

class MotionLLMHandler:
    def __init__(self, env, robot_model, gripper_helper, verifier):
        self.env = env
        self.rm = robot_model
        self.gripper = gripper_helper
        self.vf = verifier
        
        self.llm_cfg = env.cfg.get("llm", {})
        self.gcfg = env.cfg.get("gripper_geom", {})
        self.ccfg = env.cfg.get("clearance", {})
        self.vcfg = env.cfg.get("verify", {})
        
        self.open_clearance = float(self.gcfg.get("open_clearance", 0.010))
        self.release_above = float(self.ccfg.get("release_above", 0.010))
        
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Setup LLM client"""
        if not self.llm_cfg.get("enabled", False):
            self.llm_agent = None
            return
        
        client_type = self.llm_cfg.get("client", "openai")
        llm_mode = self.llm_cfg.get("mode", "multi_agent")
        
        if client_type.lower() == "openai":
            client = OpenAIChatClient(
                model=self.llm_cfg.get("model", "gpt-4o-mini"),
                temperature=self.llm_cfg.get("temperature", 0.0),
                api_key=self.llm_cfg.get("api_key")
            )
            self.llm_agent = LLMPlanner(client=client, enabled=True, mode=llm_mode)
            print(f"[LLM] Initialization complete - Mode: {llm_mode}")
        else:
            # print(f"[WARN] Unsupported LLM client: {client_type}")
            self.llm_agent = None

    def _brick_xy_yaw(self):
        pos, rpy = self.vf.brick_state()
        return pos[0], pos[1], pos[2], rpy[2]

    def gather_llm_context(self, wps, aux) -> Dict[str, Any]:
        """Gather complete context information required for LLM planning"""
        bx, by, bz, byaw = self._brick_xy_yaw()
        
        L, W, H = self.env.cfg["brick"]["size_LWH"]
        ground_z = aux.get("ground_z", 0.0)
        
        tip_guess = float(self.gcfg.get("finger_tip_length", 0.012))
        tip_measured = self.rm.estimate_tip_length_from_tcp(fallback=tip_guess)
        approach = float(self.ccfg.get("approach", 0.08))
        
        tcp_xyz, tcp_quat = self.rm.tcp_world_pose()
        tcp_rpy = p.getEulerFromQuaternion(tcp_quat)
        
        return {
            "scene": {
                "gravity": -9.81,
                "time_step": self.env.dt,
                "friction": {"ground": 0.5, "brick": 0.7, "finger": 0.8}
            },
            "control": {
                "joint_velocity_limit": 2.0,
                "joint_acc_limit": 5.0,
                "ik_max_iters": 100,
                "ik_pos_tolerance": 1e-4,
                "ik_ori_tolerance": 1e-3
            },
            "robot_specs": {
                "dof": 7,
                "arm_joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
                "arm_joint_types": ["revolute"] * 7,
                "joint_limits_rad": [[-2.97, 2.97]] * 7,
                "segment_lengths": [0.36, 0.42, 0.4, 0.081],
                "total_length_approx": 1.261,
                "base_to_tcp_length_now": np.linalg.norm(tcp_xyz),
                "tcp_calibrated": True
            },
            "gripper": {
                "tip_length_guess": tip_guess,
                "measured_tip_length": tip_measured,
                "finger_depth": float(self.gcfg.get("finger_depth", 0.035)),
                "open_clearance": self.open_clearance,
                "width_pad": float(self.gcfg.get("width_padding", 0.008)),
                "max_sym_open_rad": 0.523,
                "measured_gap_width_now": self.open_clearance
            },
            "brick": {
                "size_LWH": [L, W, H],
                "mass": 0.5,
                "pos": [bx, by, bz],
                "rpy": [0.0, 0.0, byaw]
            },
            "goal": {
                "phase": "pre_grasp",
                "description": "Grasp preparation position directly above brick"
            },
            "verify": {
                "xy_tolerance": float(self.vcfg.get("tol_xy", 0.006)),
                "z_tolerance": float(self.vcfg.get("tol_z", 0.003)),
                "angle_tolerance": float(self.vcfg.get("tol_angle", 0.05))
            },
            "constraints": {
                "approach_clearance": approach,
                "ground_z": ground_z,
                "lift_clearance": float(self.ccfg.get("lift", 0.15)),
                "safety_clearance": 0.05,
                "description": "Safety clearance and ground reference for LLM to compute target positions"
            },
            "now": {
                "tcp_xyz": tcp_xyz,
                "tcp_rpy": tcp_rpy
            }
        }

    def plan_pre_grasp_pose(self, wps, aux) -> Dict[str, Any]:
        if self.llm_agent is None:
            print("[INFO] LLM disabled, using classical planning")
            return wps[0]
        
        context = self.gather_llm_context(wps, aux)
        bx, by, bz, byaw = self._brick_xy_yaw()
        
        L, W, H = self.env.cfg["brick"]["size_LWH"]
        z_top = bz + H/2.0
        tip_measured = self.rm.estimate_tip_length_from_tcp(fallback=float(self.gcfg.get("tip_length_guess", 0.06)))
        approach = float(self.ccfg.get("approach", 0.08))
        want_tcp_z = z_top + approach + tip_measured
        
        plan_pose = {
            "xyz": [bx, by, want_tcp_z],
            "rpy": align_topdown_to_brick(byaw)
        }
        
        try:
            result = self.llm_agent.plan_pre_grasp(context, 0, None)
            llm_pose = result["pose"]
            # print(f"[LLM DEBUG] Real brick position: ({bx:.6f}, {by:.6f}, {bz:.6f})")
            # print(f"[LLM DEBUG] Want TCP z: {want_tcp_z:.6f}")
            # print(f"[LLM DEBUG] LLM output: xyz=({llm_pose['xyz'][0]:.6f}, {llm_pose['xyz'][1]:.6f}, {llm_pose['xyz'][2]:.6f})")
            return {
                "pose": llm_pose,
                "llm_result": result,
                "source": "llm_verified"
            }
        except Exception as e:
            print(f"[ERROR] LLM planning failed: {e}")
            return {
                "pose": plan_pose,
                "source": "fallback_geometric"
            }

    def plan_descend(self, wps, aux, W, ground_z) -> Dict[str, Any]:
        if self.llm_agent is None:
            print("[INFO] LLM disabled, using classical descend planning")
            return None
        
        context = self.gather_llm_context(wps, aux)
        bx, by, bz, byaw = self._brick_xy_yaw()
        
        if "computed" not in context:
            context["computed"] = {}
        context["computed"]["ground_z"] = ground_z
        H = context["brick"]["size_LWH"][2]
        context["computed"]["z_top"] = bz + H/2.0
        context["gripper"]["width_pad"] = float(self.gcfg.get("width_padding", 0.008))
        context["gripper"]["ground_margin"] = float(self.gcfg.get("ground_margin", 0.003))
        
        try:
            result = self.llm_agent.plan_descend(context, 0, None)
            # print(f"[LLM_DESCEND] Using LLM planned descend:")
            # print(f"[LLM_DESCEND] Target gap: {result['target_gap']:.6f}m")
            # print(f"[LLM_DESCEND] TCP target: ({result['pose']['xyz'][0]:.6f}, {result['pose']['xyz'][1]:.6f}, {result['pose']['xyz'][2]:.6f})")
            return {
                "pose": result["pose"],
                "target_gap": result["target_gap"],
                "llm_result": result,
                "source": "llm_verified"
            }
        except Exception as e:
            print(f"[ERROR] LLM descend planning failed: {e}")
            return None

    def plan_lift(self, wps, aux, lift_clear, tip) -> Dict[str, Any]:
        if self.llm_agent is None:
            print("[INFO] LLM disabled, using classical lift planning")
            return None
        
        context = self.gather_llm_context(wps, aux)
        
        if "computed" not in context:
            context["computed"] = {}
        context["computed"]["lift_clearance"] = lift_clear
        context["gripper"]["attachment_status"] = "attached" if self.gripper.is_attached() else "detached"
        
        try:
            result = self.llm_agent.plan_lift(context, 0, None)
            # print(f"[LLM_LIFT] Using LLM planned lift:")
            # print(f"[LLM_LIFT] Lift height: {result['lift_height']:.6f}m")
            # print(f"[LLM_LIFT] TCP target: ({result['pose']['xyz'][0]:.6f}, {result['pose']['xyz'][1]:.6f}, {result['pose']['xyz'][2]:.6f})")
            return {
                "pose": result["pose"],
                "lift_height": result["lift_height"],
                "strategy": result.get("strategy", {}),
                "stability_control": result.get("stability_control", {}),
                "fallback_strategy": result.get("fallback_strategy", {}),
                "llm_result": result,
                "source": "llm_verified"
            }
        except Exception as e:
            print(f"[ERROR] LLM lift planning failed: {e}")
            return None

    def plan_release(self, wps, aux, gx, gy, gz_top, W, release_above, touched_sid) -> Dict[str, Any]:
        if not self.llm_agent or not self.llm_agent.enabled:
            return None
        
        wps_dict = wps if isinstance(wps, dict) else {"wps": wps}
        
        current_gap = W + self.open_clearance
        if hasattr(self.gripper, 'get_current_width'):
            current_gap = self.gripper.get_current_width()
        elif hasattr(self.gripper, 'get_gap'):
            current_gap = self.gripper.get_gap()
            
        context = {
            **wps_dict,
            "goal": {
                "target_xy": [gx, gy],
                "target_z_top": gz_top,
                "release_above": release_above
            },
            "computed": {
                "required_gap": W + self.open_clearance
            },
            "current_gap": current_gap,
            "finger_contacts": self.vf.finger_contacts_with(aux["brick_id"]) if hasattr(self.vf, 'finger_contacts_with') else 0,
            "is_attached": self.gripper.is_attached() if hasattr(self.gripper, 'is_attached') else False,
            "support_detection": "contact_detected" if touched_sid is not None else "no_contact"
        }
        
        try:
            result = self.llm_agent.plan_release(context, 0, None)
            print(f"[LLM_RELEASE] ===== LLM Release Planning Results =====")
            print(f"[LLM_RELEASE] Action type: {result['release_control']['action_type']}")
            print(f"[LLM_RELEASE] Target gap: {result['release_control']['target_gap']:.6f}m")
            return result
        except Exception as e:
            print(f"[LLM_RELEASE] Error: {e}")
            return None
