import os
import re
import json
import math
from typing import Any, Dict, Optional, Tuple

from math3d.transforms import align_topdown_to_brick
from llm_prompt import (
    get_pre_grasp_prompt, PRE_GRASP_REPLY_TEMPLATE,
    get_descend_prompt, DESCEND_REPLY_TEMPLATE,
    get_close_prompt, CLOSE_REPLY_TEMPLATE,
    get_lift_prompt, LIFT_REPLY_TEMPLATE,
    get_place_prompt, PLACE_REPLY_TEMPLATE,
    get_release_prompt, RELEASE_REPLY_TEMPLATE,
    get_single_agent_prompt, SINGLE_AGENT_REPLY_TEMPLATE,  
)


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _coerce_pose(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Ensure dictionary has pose: {xyz: [x,y,z], rpy: [r,p,y]} as float."""
    if not isinstance(d, dict):
        return None
    pose = d.get("target_tcp_pose") or d.get("pose") or d.get("target_pose")
    if not isinstance(pose, dict):
        return None
    xyz = pose.get("xyz")
    rpy = pose.get("rpy")
    if (not isinstance(xyz, (list, tuple))) or (len(xyz) != 3):
        return None
    if rpy is None:
        yaw = pose.get("yaw") or d.get("yaw_hint")
        if isinstance(yaw, (int, float)):
            rpy = [0.0, 0.0, float(yaw)]
    if (not isinstance(rpy, (list, tuple))) or (len(rpy) != 3):
        return None
    try:
        xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        rpy = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
        return {"xyz": xyz, "rpy": rpy}
    except Exception:
        return None

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON from LLM text (fault-tolerant)."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# -------------------- LLM Client -------------------- #
class BaseLLMClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

class OpenAIChatClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.temperature = float(temperature)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def complete(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f'{{"error":"{type(e).__name__}: {str(e)}"}}'


# -------------------- LLM Planning Agent -------------------- #
class LLMPlanner:
    """
    Use natural language prompts (English) to provide complete context to LLM (robot arm length, joint limits, gripper and scene parameters,
    brick status, geometric planning reference poses, etc.), making LLM output TCP target pose "positioned directly above the brick".
    """
    # Import reply templates from prompt modules
    PRE_GRASP_REPLY_TEMPLATE = PRE_GRASP_REPLY_TEMPLATE
    DESCEND_REPLY_TEMPLATE = DESCEND_REPLY_TEMPLATE
    CLOSE_REPLY_TEMPLATE = CLOSE_REPLY_TEMPLATE
    LIFT_REPLY_TEMPLATE = LIFT_REPLY_TEMPLATE
    PLACE_REPLY_TEMPLATE = PLACE_REPLY_TEMPLATE
    RELEASE_REPLY_TEMPLATE = RELEASE_REPLY_TEMPLATE
    SINGLE_AGENT_REPLY_TEMPLATE = SINGLE_AGENT_REPLY_TEMPLATE

    def __init__(self, client: Optional[BaseLLMClient] = None, 
                 enabled: bool = True,
                 mode: str = "multi_agent"):
        self.client = client
        self.enabled = bool(enabled and (client is not None))
        self.mode = mode  
        self._unified_plan_cache = None  
        self.plan_pre_place = self.plan_place

    # Prompt Construction
    def _prompt_pre_grasp(self, context: Dict[str, Any],
                          attempt_idx: int, feedback: Optional[str]) -> Tuple[str, str]:
        return get_pre_grasp_prompt(context, attempt_idx, feedback)

    def _prompt_descend(self, context: Dict[str, Any],
                       attempt_idx: int, feedback: Optional[str]) -> Tuple[str, str]:
        return get_descend_prompt(context, attempt_idx, feedback)

    def _postprocess_pose(self, raw_text: str, brick_yaw: float,
                          want_tcp_z: float, plan_pose: Dict[str, Any]) -> Dict[str, Any]:
        jd = _extract_json(raw_text) or {}
        pose = _coerce_pose(jd)
        if pose is None:
            try:
                yaw = float(jd.get("yaw") or jd.get("yaw_hint"))
                rpy = align_topdown_to_brick(yaw)
                xyz = plan_pose["xyz"][:]
                xyz[2] = want_tcp_z
                return {"xyz": xyz, "rpy": rpy, "source": "llm(yaw-only)"}
            except Exception:
                return {"xyz": plan_pose["xyz"][:], "rpy": plan_pose["rpy"][:], "source": "fallback(plan)"}

        rpy = pose["rpy"]
        yaw_hint = float(rpy[2])
        rpy = align_topdown_to_brick(yaw_hint)
        xyz = pose["xyz"]
        xyz[2] = float(want_tcp_z)
        return {"xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                "source": "llm"}

    def plan_pre_grasp(self, context: Dict[str, Any],
                       attempt_idx: int = 0,
                       feedback: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("pre_grasp", {})
            return {
                "pose": result.get("pose", {}),
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }

        
        if not self.enabled or (self.client is None):

            bx, by, bz = context["brick"]["pos"]
            H = context["brick"]["size_LWH"][2]
            byaw = context["brick"]["rpy"][2]
            approach = float(context.get("constraints", {}).get("approach_clearance", 0.08))
            tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
            z_top = bz + H/2.0
            want_tcp_z = z_top + approach + tip
            
            from math3d.transforms import align_topdown_to_brick
            plan_pose = {
                "xyz": [bx, by, want_tcp_z],
                "rpy": align_topdown_to_brick(byaw)
            }
            return {"pose": plan_pose, "raw": "LLM disabled -> use classical pre_grasp.", "source": "fallback(plan)"}

        bx, by, bz = context["brick"]["pos"]
        H = context["brick"]["size_LWH"][2]
        approach = float(context.get("constraints", {}).get("approach_clearance", 0.08))
        tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
        z_top = bz + H/2.0  # brick top surface
        want_tcp_z = z_top + approach + tip
        byaw = context["brick"]["rpy"][2]
        
        from math3d.transforms import align_topdown_to_brick
        plan_pose = {
            "xyz": [bx, by, want_tcp_z],
            "rpy": align_topdown_to_brick(byaw)
        }

        sys, usr = self._prompt_pre_grasp(context, attempt_idx, feedback)
        raw = self.client.complete(sys, usr)
        pose = self._postprocess_pose(raw, byaw, want_tcp_z, plan_pose)
        
        # Debug output
        print(f"[LLM_DEBUG] ===== LLM Planning Results =====")
        print(f"[LLM_DEBUG] Real brick position: ({bx:.6f}, {by:.6f}, {bz:.6f})")
        print(f"[LLM_DEBUG] Brick height H: {H:.6f}m")
        print(f"[LLM_DEBUG] Brick top z: {z_top:.6f}m")
        print(f"[LLM_DEBUG] Approach clearance: {approach:.6f}m")
        print(f"[LLM_DEBUG] Tip length: {tip:.6f}m")
        print(f"[LLM_DEBUG] Want TCP z: {want_tcp_z:.6f}m (z_top + approach + 0.02 + tip, GraspModule formula)")
        print(f"[LLM_DEBUG] LLM output TCP: xyz=({pose['xyz'][0]:.6f}, {pose['xyz'][1]:.6f}, {pose['xyz'][2]:.6f})")
        print(f"[LLM_DEBUG] LLM output RPY: ({pose['rpy'][0]:.6f}, {pose['rpy'][1]:.6f}, {pose['rpy'][2]:.6f})")
        print(f"[LLM_DEBUG] Source: {pose.get('source', 'llm')}")
        print(f"[LLM_DEBUG] Raw LLM response: {raw[:100]}...")  

        return {"pose": {"xyz": pose["xyz"], "rpy": pose["rpy"]}, "raw": raw, "source": pose.get("source", "llm")}

    def _postprocess_descend(self, raw_text: str, brick_pos: list, 
                           want_tcp_z: float, want_gap: float, plan_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process descend phase LLM output"""
        jd = _extract_json(raw_text) or {}
        
        # Try to parse pose
        pose = _coerce_pose(jd)
        
        # Try to parse gripper gap
        gap = None
        try:
            if "gripper_command" in jd:
                gap = float(jd["gripper_command"].get("target_gap", 0))
            elif "target_gap" in jd:
                gap = float(jd["target_gap"])
        except Exception:
            pass
        
        if pose is None or gap is None or gap <= 0:
            return {
                "pose": {"xyz": plan_pose["xyz"][:], "rpy": plan_pose["rpy"][:]},
                "target_gap": want_gap,
                "source": "fallback(plan)"
            }
        
        # Validate and correct TCP position
        xyz = pose["xyz"]
        xyz[0] = float(brick_pos[0])  # Force x align to brick center
        xyz[1] = float(brick_pos[1])  # Force y align to brick center
        xyz[2] = float(want_tcp_z)    # Force z to calculated target height
        
        # Maintain pose
        rpy = pose["rpy"]
        
        return {
            "pose": {"xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])]},
            "target_gap": float(gap),
            "source": "llm"
        }

    def plan_descend(self, context: Dict[str, Any],
                    attempt_idx: int = 0,
                    feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM planning for descend phase: gripper opening and TCP descent
        Input: integrated context
        Output: { "pose": {xyz:[..], rpy:[..]}, "target_gap": float, "raw": <llm text>, "source": "llm|fallback" }
        """
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("descend", {})
            return {
                "pose": result.get("pose", {}),
                "target_gap": result.get("target_gap", 0.0),
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }
        
        if not self.enabled or (self.client is None):
            # Fall back to geometric planning
            bx, by, bz = context["brick"]["pos"]
            H = context["brick"]["size_LWH"][2]
            W = context["brick"]["size_LWH"][1]
            tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
            open_clearance = float(context["gripper"]["open_clearance"])
            finger_depth = float(context["gripper"]["finger_depth"])
            ground_z = float(context.get("constraints", {}).get("ground_z", 0.0))
            ground_margin = 0.003
            
            # Geometric calculation
            required_gap = W + open_clearance
            desired_bottom = max(ground_z + ground_margin, bz - finger_depth/2)
            tcp_z = desired_bottom + tip
            
            # Build fallback plan_pose
            from math3d.transforms import align_topdown_to_brick
            byaw = context["brick"]["rpy"][2]
            plan_pose = {"xyz": [bx, by, tcp_z], "rpy": align_topdown_to_brick(byaw)}
            
            return {
                "pose": plan_pose,
                "target_gap": required_gap,
                "raw": "LLM disabled -> use classical descend.",
                "source": "fallback(plan)"
            }

        # Prepare calculation parameters
        bx, by, bz = context["brick"]["pos"]
        H = context["brick"]["size_LWH"][2]
        W = context["brick"]["size_LWH"][1]
        tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
        open_clearance = float(context["gripper"]["open_clearance"])
        width_pad = float(context["gripper"]["width_pad"])
        finger_depth = float(context["gripper"]["finger_depth"])
        ground_z = float(context.get("constraints", {}).get("ground_z", 0.0))
        ground_margin = 0.003
        
        # Calculate target parameters
        required_gap = W + open_clearance
        desired_bottom = max(ground_z + ground_margin, bz - finger_depth/2)
        want_tcp_z = desired_bottom + tip
        byaw = context["brick"]["rpy"][2]
    
        
        from math3d.transforms import align_topdown_to_brick
        plan_pose = {"xyz": [bx, by, want_tcp_z], "rpy": align_topdown_to_brick(byaw)}

        # Call LLM
        sys, usr = self._prompt_descend(context, attempt_idx, feedback)
        raw = self.client.complete(sys, usr)
        result = self._postprocess_descend(raw, [bx, by, bz], want_tcp_z, required_gap, plan_pose)
        
        # Debug output
        print(f"[LLM_DESCEND] ===== LLM Descend Planning Results =====")
        print(f"[LLM_DESCEND] Brick position: ({bx:.6f}, {by:.6f}, {bz:.6f})")
        print(f"[LLM_DESCEND] Brick width W: {W:.6f}m")
        print(f"[LLM_DESCEND] Required gap: {required_gap:.6f}m")
        print(f"[LLM_DESCEND] Desired finger bottom: {desired_bottom:.6f}m")
        print(f"[LLM_DESCEND] Want TCP z: {want_tcp_z:.6f}m")
        print(f"[LLM_DESCEND] LLM output TCP: xyz=({result['pose']['xyz'][0]:.6f}, {result['pose']['xyz'][1]:.6f}, {result['pose']['xyz'][2]:.6f})")
        print(f"[LLM_DESCEND] LLM target gap: {result['target_gap']:.6f}m")
        print(f"[LLM_DESCEND] Source: {result.get('source', 'llm')}")
        
        return {"pose": result["pose"], "target_gap": result["target_gap"], "raw": raw, "source": result.get("source", "llm")}

    def _prompt_lift(self, context: Dict[str, Any],
                    attempt_idx: int, feedback: Optional[str]) -> Tuple[str, str]:
        """
        Lift phase LLM Prompt - responsible for safely lifting brick to specified height
        context fields: scene, control, robot_specs, gripper, brick, goal, verify, classical, computed, now
        """
        return get_lift_prompt(context, attempt_idx, feedback)

    def _postprocess_lift(self, raw_text: str, brick_pos: list, 
                         want_tcp_z: float, current_rpy: list, plan_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process lift phase LLM output"""
        jd = _extract_json(raw_text) or {}
        
        # Try to parse pose
        pose = _coerce_pose(jd)
        
        # Try to parse lift strategy
        lift_strategy = jd.get("lift_strategy", {})
        lift_height = lift_strategy.get("lift_height", 0)
        
        if pose is None or lift_height <= 0:
            return {
                "pose": {"xyz": plan_pose["xyz"][:], "rpy": plan_pose["rpy"][:]},
                "lift_height": plan_pose["lift_height"],
                "source": "fallback(plan)"
            }
        
        # Validate and correct TCP position
        xyz = pose["xyz"]
        xyz[0] = float(brick_pos[0])  # Force x align to brick center
        xyz[1] = float(brick_pos[1])  # Force y align to brick center
        xyz[2] = float(want_tcp_z)    # Force z to calculated target height
        
        rpy = current_rpy
        
        return {
            "pose": {"xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])]},
            "lift_height": float(lift_height),
            "lift_strategy": lift_strategy,
            "stability_control": jd.get("stability_control", {}),
            "fallback_strategy": jd.get("fallback_strategy", {}),
            "source": "llm"
        }

    def plan_lift(self, context: Dict[str, Any],
                  attempt_idx: int = 0,
                  feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM planning for lift phase: safely lifting brick
        Input: integrated context
        Output: { "pose": {xyz:[..], rpy:[..]}, "lift_height": float, "strategy": {...}, "raw": <llm_text>, "source": "llm|fallback" }
        """
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("lift", {})
            return {
                "pose": result.get("pose", {}),
                "lift_height": result.get("lift_height", 0.15),
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }
        
        if not self.enabled or (self.client is None):
            # Fall back to geometric planning
            bx, by, bz = context["brick"]["pos"]
            H = context["brick"]["size_LWH"][2]
            tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
            lift_clearance = float(context.get("constraints", {}).get("lift_clearance", 0.15))
            
            # Geometric calculation
            z_top = bz + H/2.0
            target_tcp_z = z_top + lift_clearance + tip
            byaw = context["brick"]["rpy"][2]
            
            from math3d.transforms import align_topdown_to_brick
            plan_pose = {
                "xyz": [bx, by, target_tcp_z], 
                "rpy": align_topdown_to_brick(byaw),
                "lift_height": lift_clearance
            }
            
            return {
                "pose": {"xyz": plan_pose["xyz"], "rpy": plan_pose["rpy"]},
                "lift_height": plan_pose["lift_height"],
                "raw": "LLM disabled -> use classical lift.",
                "source": "fallback(plan)"
            }

        # Prepare calculation parameters
        bx, by, bz = context["brick"]["pos"]
        H = context["brick"]["size_LWH"][2]
        tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
        lift_clearance = float(context.get("constraints", {}).get("lift_clearance", 0.15))
        
        # Calculate target parameters
        z_top = bz + H/2.0
        want_tcp_z = z_top + lift_clearance + tip
        current_rpy = context["now"]["tcp_rpy"]
        

        
        # Classical geometric planning as backup
        plan_pose = {
            "xyz": [bx, by, want_tcp_z], 
            "rpy": current_rpy,
            "lift_height": lift_clearance
        }

        # Call LLM
        sys, usr = self._prompt_lift(context, attempt_idx, feedback)
        raw = self.client.complete(sys, usr)
        result = self._postprocess_lift(raw, [bx, by, bz], want_tcp_z, current_rpy, plan_pose)
        
        # Debug output
        print(f"[LLM_LIFT] ===== LLM Lift Planning Results =====")
        print(f"[LLM_LIFT] Brick position: ({bx:.6f}, {by:.6f}, {bz:.6f})")
        print(f"[LLM_LIFT] Brick top z: {z_top:.6f}m")
        print(f"[LLM_LIFT] Lift clearance: {lift_clearance:.6f}m")
        print(f"[LLM_LIFT] Want TCP z: {want_tcp_z:.6f}m")
        print(f"[LLM_LIFT] LLM output TCP: xyz=({result['pose']['xyz'][0]:.6f}, {result['pose']['xyz'][1]:.6f}, {result['pose']['xyz'][2]:.6f})")
        print(f"[LLM_LIFT] LLM lift height: {result['lift_height']:.6f}m")
        print(f"[LLM_LIFT] Source: {result.get('source', 'llm')}")
        
        return {
            "pose": result["pose"], 
            "lift_height": result["lift_height"],
            "strategy": result.get("lift_strategy", {}),
            "stability_control": result.get("stability_control", {}),
            "fallback_strategy": result.get("fallback_strategy", {}),
            "raw": raw, 
            "source": result.get("source", "llm")
        }


    def plan_place(self, context: Dict[str, Any],
                  attempt_idx: int = 0,
                  feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 5-6 fusion: Use LLM planning for place phase, return different formats based on context phase field
        Merged original plan_pre_place and plan_descend_place functions
        
        Input: context including goal.phase = "pre_place" or "descend_place"
        Output: different format results based on phase
        """
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("place", {})
            return {
                **result,
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }
        
        # Get calling phase
        phase = context.get("goal", {}).get("phase", "unknown")
        
        if not self.enabled or (self.client is None):
            # LLM disabled, return classical strategy based on different phases
            return self._fallback_place_strategy(context, phase)
        
        # Get basic parameters
        gx, gy = context["goal"]["target_xy"]
        gz_top = context["goal"]["target_z_top"]
        yaw_place = context["goal"].get("target_yaw", 0.0)
        approach = context.get("constraints", {}).get("approach_clearance", 0.08)
        tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
        support_ids = context["goal"].get("support_ids", [0])
        release_above = context["goal"].get("release_above", 0.010)
        
        try:
            # Call LLM for unified planning
            sys, usr = self._prompt_unified_place(context, attempt_idx, feedback)
            raw = self.client.complete(sys, usr)
            result = self._postprocess_unified_place(raw, context)
            
            # Return to appropriate format based on calling phase
            if phase == "pre_place":
                return self._format_pre_place_result(result, gx, gy, gz_top, yaw_place, approach, tip)
            elif phase == "descend_place":
                return self._format_descend_place_result(result, gx, gy, gz_top, support_ids, release_above, approach)
            else:
                return {
                    **result,
                    "raw": raw,
                    "source": "llm"
                }
                
        except Exception as e:
            print(f"[ERROR] LLM {phase} planning failed: {e}")
            return self._fallback_place_strategy(context, phase)
    
    def _fallback_place_strategy(self, context: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """Return classical strategy based on different phases"""
        gx, gy = context["goal"]["target_xy"]
        gz_top = context["goal"]["target_z_top"]
        yaw_place = context["goal"].get("target_yaw", 0.0)
        approach = context.get("constraints", {}).get("approach_clearance", 0.08)
        tip = float(context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"])
        support_ids = context["goal"].get("support_ids", [0])
        release_above = context["goal"].get("release_above", 0.010)
        
        if phase == "pre_place":
            target_tcp_z = gz_top + approach + tip
            return {
                "pose": {
                    "xyz": [gx, gy, target_tcp_z],
                    "rpy": [0.0, 0.0, yaw_place]
                },
                "approach_height": approach,
                "strategy": "geometric_alignment",
                "description": "Classical geometric approach strategy",
                "source": "fallback(plan)",
                "raw": "LLM disabled -> use classical pre_place."
            }
        elif phase == "descend_place":
            start_height = gz_top + max(release_above, 0.006)
            return {
                "descent_control": {
                    "action_type": "gradual_descent_to_support",
                    "start_height": start_height,
                    "step_size": 0.004,
                    "max_descent_range": 0.030,
                    "orientation_control": {
                        "target_rpy": [0.0, 0.0, yaw_place],
                        "roll_tolerance": 0.002,
                        "pitch_tolerance": 0.002,
                        "yaw_tolerance": 0.005,
                        "strict_parallel": True,
                        "description": "Fallback strategy: strictly maintain brick bottom surface parallel to ground"
                    },
                    "description": "Classical geometric descent strategy"
                },
                "contact_detection": {
                    "support_ids": support_ids,
                    "detection_method": "collision_monitoring",
                    "force_threshold": "adaptive",
                    "description": "Standard contact detection"
                },
                "position_control": {
                    "xy_alignment": [gx, gy],
                    "z_adjustment": "contact_based",
                    "orientation_hold": True
                },
                "safety_monitoring": {
                    "max_force_limit": "safe_threshold",
                    "emergency_abort": True,
                    "grasp_verification": True
                },
                "fallback_strategy": {
                    "no_contact_detected": "extend_search_range",
                    "excessive_force": "abort_and_retreat",
                    "positioning_error": "micro_adjustment",
                    "max_retries": 3
                },
                "source": "fallback(plan)",
                "raw": "LLM disabled -> use classical descend_place."
            }
        else:
            target_tcp_z = gz_top + approach + tip
            start_height = gz_top + max(release_above, 0.006)
            return {
                "approach_phase": {
                    "target_pose": {
                        "xyz": [gx, gy, target_tcp_z], 
                        "rpy": [0.0, 0.0, yaw_place]
                    },
                    "approach_height": approach,
                    "strategy": "geometric_alignment",
                    "description": "Geometric alignment above target position"
                },
                "descent_phase": {
                    "action_type": "gradual_descent_to_support",
                    "start_height": start_height,
                    "step_size": 0.004,
                    "max_descent_range": 0.030,
                    "orientation_control": {
                        "target_rpy": [0.0, 0.0, yaw_place],
                        "roll_tolerance": 0.002,
                        "pitch_tolerance": 0.002,
                        "yaw_tolerance": 0.005,
                        "strict_parallel": True,
                        "description": "Complete fallback strategy: strictly maintain brick bottom surface parallel to ground"
                    },
                    "description": "Classical geometric descent strategy"
                },
                "contact_detection": {
                    "support_ids": support_ids,
                    "detection_method": "collision_monitoring",
                    "force_threshold": "adaptive",
                    "description": "Standard contact detection"
                },
                "source": "fallback(plan)",
                "raw": "LLM disabled -> use classical place strategy."
            }
    
    def _format_pre_place_result(self, result: Dict[str, Any], gx: float, gy: float, gz_top: float, 
                               yaw_place: float, approach: float, tip: float) -> Dict[str, Any]:
        """Format unified LLM result into pre_place expected format"""
        approach_phase = result.get("approach_phase", {})
        target_pose = approach_phase.get("target_pose", {
            "xyz": [gx, gy, gz_top + approach + tip],
            "rpy": [0.0, 0.0, yaw_place]
        })
        
        return {
            "pose": target_pose,
            "approach_height": approach_phase.get("approach_height", approach),
            "strategy": approach_phase.get("strategy", "llm_planned"),
            "description": approach_phase.get("description", "LLM planned approach strategy"),
            "source": "llm",
            "raw": result.get("raw", "")
        }
    
    def _format_descend_place_result(self, result: Dict[str, Any], gx: float, gy: float, gz_top: float,
                                   support_ids: list, release_above: float, approach: float) -> Dict[str, Any]:
        """Format unified LLM result into descend_place expected format"""
        descent_phase = result.get("descent_phase", {})
        contact_detection = result.get("contact_detection", {})
        
        # Force set correct descent start height: should start from target height plus small gap
        # This ensures it won't be higher than pre_place position
        correct_start_height = gz_top + max(release_above, 0.006)
        
        # Debug output
        llm_start_height = descent_phase.get("start_height", correct_start_height)
        print(f"[DESCEND_HEIGHT_DEBUG] LLM suggested height: {llm_start_height:.6f}m")
        print(f"[DESCEND_HEIGHT_DEBUG] Forced height: {correct_start_height:.6f}m")
        print(f"[DESCEND_HEIGHT_DEBUG] Target surface: {gz_top:.6f}m")
        print(f"[DESCEND_HEIGHT_DEBUG] Safety gap: {max(release_above, 0.006):.6f}m")
        
        return {
            "descent_control": {
                "action_type": descent_phase.get("action_type", "gradual_descent_to_support"),
                "start_height": correct_start_height,
                "step_size": descent_phase.get("step_size", 0.004),
                "max_descent_range": descent_phase.get("max_descent_range", 0.030),
                "description": descent_phase.get("description", "LLM planned descent strategy")
            },
            "contact_detection": {
                "support_ids": contact_detection.get("support_ids", support_ids),
                "detection_method": contact_detection.get("detection_method", "collision_monitoring"),
                "force_threshold": contact_detection.get("force_threshold", "adaptive"),
                "description": contact_detection.get("description", "LLM planned contact detection")
            },
            "position_control": {
                "xy_alignment": [gx, gy],
                "z_adjustment": "contact_based",
                "orientation_hold": True
            },
            "safety_monitoring": {
                "max_force_limit": "safe_threshold",
                "emergency_abort": True,
                "grasp_verification": True
            },
            "fallback_strategy": {
                "no_contact_detected": "extend_search_range",
                "excessive_force": "abort_and_retreat",
                "positioning_error": "micro_adjustment",
                "max_retries": 3
            },
            "source": "llm",
            "raw": result.get("raw", "")
        }

    def _prompt_unified_place(self, context: Dict[str, Any], attempt_idx: int = 0, feedback: str = "") -> Tuple[str, str]:
        """Build unified place phase prompt - using standard six-part structure"""
        return get_place_prompt(context, attempt_idx, feedback)

    def _postprocess_unified_place(self, raw_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process unified place phase LLM response"""
        jd = _extract_json(raw_text) or {}
        
        goal = context.get("goal", {})
        computed = context.get("computed", {})
        
        target_xy = goal.get("target_xy", [0, 0])
        target_z_top = goal.get("target_z_top", 0)
        target_yaw = goal.get("target_yaw", 0)
        approach = computed.get("approach", 0.08)
        support_ids = goal.get("support_ids", [0])
        
        # Process approach_phase
        approach_phase = jd.get("approach_phase", {})
        if not approach_phase:
            target_tcp_z = target_z_top + approach + 0.048
            approach_phase = {
                "target_pose": {
                    "xyz": [target_xy[0], target_xy[1], target_tcp_z],
                    "rpy": [0.0, 0.0, target_yaw]
                },
                "approach_height": approach,
                "strategy": "fallback_geometric",
                "description": "Fallback to geometric strategy"
            }
        
        # Process descent_phase
        descent_phase = jd.get("descent_phase", {})
        if not descent_phase:
            descent_phase = {
                "action_type": "gradual_descent_to_support",
                "start_height": target_z_top + 0.010,
                "step_size": 0.004,
                "max_descent_range": 0.030,
                "orientation_control": {
                    "target_rpy": [0.0, 0.0, target_yaw],
                    "roll_tolerance": 0.002,
                    "pitch_tolerance": 0.002, 
                    "yaw_tolerance": 0.005,
                    "strict_parallel": True,
                    "description": "Fallback strategy: strictly maintain brick bottom parallel to ground"
                },
                "description": "Fallback to geometric descent"
            }
        else:
            if "orientation_control" not in descent_phase:
                descent_phase["orientation_control"] = {
                    "target_rpy": [0.0, 0.0, target_yaw],
                    "roll_tolerance": 0.002,
                    "pitch_tolerance": 0.002,
                    "yaw_tolerance": 0.005,
                    "strict_parallel": True,
                    "description": "LLM planning: strictly maintain brick bottom parallel to ground"
                }
        
        # Process contact_detection
        contact_detection = jd.get("contact_detection", {})
        if not contact_detection:
            contact_detection = {
                "support_ids": support_ids,
                "detection_method": "collision_monitoring", 
                "force_threshold": "adaptive",
                "description": "Standard contact detection"
            }
        
        return {
            "approach_phase": approach_phase,
            "descent_phase": descent_phase,
            "contact_detection": contact_detection,
            "source": "llm",
            "raw": raw_text
        }


    def _prompt_release(self, context: Dict[str, Any], 
                       attempt_idx: int = 0,
                       feedback: Optional[str] = None) -> Tuple[str, str]:
        """Generate release phase prompt using 6-part structure"""
        
        # Extract context information
        tcp_xyz = context["now"]["tcp_xyz"]
        brick_pos = context["brick"]["pos"]
        target_pos = context["goal"]["target_xy"] + [context["goal"]["target_z_top"]]
        brick_size = context["brick"]["size_LWH"]
        
        # Gripper and release parameters
        current_gap = context.get("current_gap", 0.105)
        required_gap = 0.105
        release_above = context["goal"].get("release_above", 0.010)
        max_opening = context["gripper"].get("max_opening", 0.200)
        
        # Contact and status information
        finger_contacts = context.get("finger_contacts", 0)
        is_attached = context.get("is_attached", False)
        support_detection = context.get("support_detection", "contact_detected")
        
        # Physical parameters
        gravity = context["scene"]["gravity"]
        friction_info = context["scene"]["friction"]
        
        # -- Release Phase 6-Part Structured Prompt --
        task = f"""
## (1) Current Environment State

**Robot Arm Status:**
- Current TCP position: ({tcp_xyz[0]:.6f}, {tcp_xyz[1]:.6f}, {tcp_xyz[2]:.6f})m
- TCP positioning accuracy: target position reached
- Descent phase: completed, brick in contact with support surface

**Gripper Status:**
- Current gripper gap: {current_gap:.6f}m
- Required brick clearance: {required_gap:.6f}m
- Maximum opening capacity: {max_opening:.6f}m
- Finger contact count: {finger_contacts}
- Physical attachment status: {is_attached}

**Brick Status:**
- Brick dimensions LWH: {brick_size[0]:.3f} x {brick_size[1]:.3f} x {brick_size[2]:.3f}m
- Current brick position: ({brick_pos[0]:.6f}, {brick_pos[1]:.6f}, {brick_pos[2]:.6f})m
- Target placement position: ({target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f})m
- Support surface contact: {support_detection}

**Physical Environment:**
- Gravity: {gravity}
- Friction coefficients: {friction_info}
- Release height clearance: {release_above:.6f}m

## (2) Memory Information

**Task Progress:**
- Current execution phase: Brick release (Release)
- Completed phases: ✓ Approach → ✓ Descent → ✓ Contact with support surface
- Currently processing: Safe brick release and gripper opening

**Completed Steps:**
- ✓ Pre-grasp pose planning and movement
- ✓ Descent to grasp position
- ✓ Gripper closing and brick grasp
- ✓ Lifting to safe height
- ✓ Movement to placement position
- ✓ Precise descent to support surface
- ✓ Contact detection confirmation

**Current Task Status:**
- Phase: release (brick release)
- Objective: Safely open gripper to release brick while maintaining placement precision
- Next step: gripper retreat and position verification

**Error Feedback Information:**
{('Previous attempt rejected, reason: ' + feedback) if feedback else 'No historical error feedback'}

## (3) Role Definition

You are a **Brick Release Control Expert Agent**, specifically responsible for safely releasing bricks after precise placement while ensuring placement quality.

**Main Responsibilities:**
- Design gradual gripper opening strategy ensuring brick remains stable during release
- Monitor contact status preventing brick displacement or adhesion
- Plan gripper retreat path avoiding collision with placed brick
- Verify successful release and placement accuracy

**Specific Tasks:**
1. **Gradual Opening Control**: Calculate optimal gripper opening sequence preventing sudden brick movement
2. **Contact Monitoring**: Real-time monitoring of finger-brick contact during opening process
3. **Anti-adhesion Strategy**: Prevent brick sticking to gripper fingers through micro-movements
4. **Retreat Planning**: Design safe gripper withdrawal path maintaining brick stability

## (4) Knowledge Base

**Release Dynamics Knowledge:**
- Gradual opening prevents sudden force changes that could displace brick
- Contact monitoring detects when brick fully separates from gripper
- Anti-stick strategies include slight lifting and micro-vibrations
- Opening speed affects release stability: too fast causes displacement, too slow wastes time

**Gripper Opening Strategy:**
- Target gap = brick width + extra clearance + safety margin
- Opening increment: 0.005-0.015m steps for smooth release
- Contact threshold monitoring: detect force drop indicating separation
- Maximum opening limit: stay within gripper capability range

**Safety and Precision Control:**
- Maintain TCP position during opening to prevent brick push/pull
- Monitor placement accuracy throughout release process
- Emergency procedures for stuck or misaligned scenarios
- Verification criteria for successful release completion

**Physical Constraints:**
- Current gripper gap: {current_gap:.6f}m
- Required clearance: {required_gap:.6f}m
- Maximum opening: {max_opening:.6f}m
- Opening tolerance: ±0.001m precision

## (5) Thinking Chain

**Step 1: Assess Current Release Conditions**
- Current gripper gap: {current_gap:.6f}m
- Required release gap: minimum {required_gap + 0.015:.6f}m for safe clearance
- Contact status: {finger_contacts} contacts detected
- Placement verification: support surface contact confirmed

**Step 2: Calculate Release Parameters**
- Target opening gap: {required_gap + 0.015:.6f}m (brick width + clearance)
- Opening increment: 0.008m steps for smooth release
- Total opening distance: {(required_gap + 0.015) - current_gap:.6f}m
- Estimated opening steps: {int(((required_gap + 0.015) - current_gap) / 0.008) + 1}

**Step 3: Design Opening Strategy**
- Progressive opening: start with small increments
- Contact monitoring: detect separation at each step
- Position stability: maintain TCP coordinates during opening
- Anti-stick measures: slight upward movement if needed

**Step 4: Plan Retreat Sequence**
- Verify zero contact before retreat
- Lift distance: {release_above:.6f}m above brick
- Retreat distance: 0.050m horizontal withdrawal
- Final position: safe observation height for verification

## (6) Output Format

**Output Description:**
Your output will directly control gripper opening and retreat sequence for safe brick release:
1. release_control - Gripper opening control parameters
2. contact_monitoring - Contact detection and separation verification
3. position_adjustment - TCP position maintenance and retreat planning
4. safety_verification - Release success criteria and verification methods

**Strict Constraints:**
1. target_gap ≥ current_gap + 0.010 (minimum additional opening)
2. target_gap ≤ max_opening * 0.9 (stay within gripper limits)
3. increment_step: 0.005-0.015m (smooth progressive opening)
4. lift_distance ≥ 0.005m (minimum clearance for retreat)
5. **Only JSON format output allowed, no markdown code blocks or additional text**

**Output Template:**
{{
  "release_control": {{
    "action_type": "gradual_release",
    "target_gap": {required_gap + 0.020:.6f},
    "release_speed": "medium",
    "initial_gap": {current_gap:.6f},
    "increment_step": 0.008,
    "max_gap": {min(max_opening * 0.9, required_gap + 0.050):.6f},
    "tolerance": 0.001,
    "description": "Progressive gripper opening for safe brick release"
  }},
  "contact_monitoring": {{
    "detection_method": "combined_check",
    "contact_threshold": 0.5,
    "detach_verification": true,
    "anti_stick_enabled": true,
    "lift_distance": 0.010,
    "shake_amplitude": 0.002,
    "description": "Real-time contact detection and anti-adhesion control"
  }},
  "position_adjustment": {{
    "xy_maintain": true,
    "z_lift_after": {release_above:.6f},
    "orientation_lock": true,
    "retreat_distance": 0.050,
    "safe_height": 0.200,
    "path_planning": "adaptive"
  }},
  "safety_verification": {{
    "success_criteria": ["zero_contact", "no_constraint", "stable_placement"],
    "verification_timeout": 2.0,
    "force_monitoring": true,
    "emergency_procedures": ["large_opening", "force_detach"]
  }}
}}

**Important Reminders:**
- Gap and distance units: meters (m), must be positive numbers
- Contact threshold units: force (N) or contact count
- All parameters must be within gripper physical limits
- reasoning field should explain core considerations of release strategy
""".strip()

        system = (
            "You are a robotic gripper release control expert. "
            "Your output controls safe brick release after precise placement. "
            "Always output valid JSON only, with precise numbers in meters."
        )
        user = task
        return system, user

    def _postprocess_release(self, raw_text: str, current_gap: float,
                           max_opening: float, target_gap: float) -> Dict[str, Any]:
        """Post-process release phase LLM output"""
        jd = _extract_json(raw_text) or {}
        
        release_control = jd.get("release_control", {})
        target_gap_llm = release_control.get("target_gap", 0)
        initial_gap = release_control.get("initial_gap", 0)
        increment_step = release_control.get("increment_step", 0)
        max_gap = release_control.get("max_gap", 0)
        
        contact_monitoring = jd.get("contact_monitoring", {})
        contact_threshold = contact_monitoring.get("contact_threshold", 0.5)
        lift_distance = contact_monitoring.get("lift_distance", 0)
        
        if (target_gap_llm <= 0 or initial_gap <= 0 or increment_step <= 0 or 
            max_gap <= 0 or lift_distance <= 0):
            fallback_target_gap = max(target_gap * 1.2, current_gap + 0.020)
            fallback_target_gap = min(fallback_target_gap, max_opening * 0.9)
            
            return {
                "release_control": {
                    "action_type": "gradual_release",
                    "target_gap": fallback_target_gap,
                    "release_speed": "medium",
                    "initial_gap": current_gap,
                    "increment_step": 0.010,
                    "max_gap": max_opening * 0.9,
                    "tolerance": 0.001,
                    "description": "Fallback to classical progressive release"
                },
                "contact_monitoring": {
                    "detection_method": "combined_check",
                    "contact_threshold": 0.5,
                    "detach_verification": True,
                    "anti_stick_enabled": True,
                    "lift_distance": 0.010,
                    "description": "Classical contact detection and anti-stick"
                },
                "position_adjustment": {
                    "xy_maintain": True,
                    "z_lift_after": 0.015,
                    "orientation_lock": True,
                    "retreat_distance": 0.050,
                    "safe_height": 0.200,
                    "path_planning": "straight"
                },
                "safety_verification": {
                    "success_criteria": ["zero_contact", "no_constraint"],
                    "verification_timeout": 2.0,
                    "force_monitoring": True,
                    "emergency_procedures": ["large_opening", "force_detach"]
                },
                "fallback_strategy": {
                    "max_retries": 5,
                    "retry_strategy": "increase_gap",
                    "gap_increment": 0.010,
                    "final_action": "max_opening_lift",
                    "emergency_gap": max_opening
                },
                "source": "fallback(plan)"
            }
        
        target_gap_llm = max(current_gap + 0.005, min(target_gap_llm, max_opening * 0.9))
        increment_step = max(0.005, min(increment_step, 0.020))
        lift_distance = max(0.005, min(lift_distance, 0.030))
        
        if isinstance(contact_threshold, str):
            contact_threshold = 0.5
        else:
            contact_threshold = max(0.1, min(float(contact_threshold), 10.0))
        
        return {
            "release_control": {
                "action_type": release_control.get("action_type", "gradual_release"),
                "target_gap": float(target_gap_llm),
                "release_speed": release_control.get("release_speed", "medium"),
                "initial_gap": max(current_gap, float(initial_gap)),
                "increment_step": float(increment_step),
                "max_gap": min(float(max_gap), max_opening),
                "tolerance": float(release_control.get("tolerance", 0.001)),
                "description": release_control.get("description", "LLM planned release strategy")
            },
            "contact_monitoring": {
                "detection_method": contact_monitoring.get("detection_method", "combined_check"),
                "contact_threshold": float(contact_threshold),
                "detach_verification": contact_monitoring.get("detach_verification", True),
                "anti_stick_enabled": contact_monitoring.get("anti_stick_enabled", True),
                "lift_distance": float(lift_distance),
                "shake_amplitude": float(contact_monitoring.get("shake_amplitude", 0.002)),
                "description": contact_monitoring.get("description", "LLM configured contact monitoring")
            },
            "position_adjustment": jd.get("position_adjustment", {
                "xy_maintain": True,
                "z_lift_after": 0.015,
                "orientation_lock": True,
                "retreat_distance": 0.050,
                "safe_height": 0.200,
                "path_planning": "adaptive"
            }),
            "safety_verification": jd.get("safety_verification", {
                "success_criteria": ["zero_contact", "no_constraint", "stable_placement"],
                "verification_timeout": 2.0,
                "force_monitoring": True,
                "emergency_procedures": ["large_opening", "force_detach"]
            }),
            "fallback_strategy": jd.get("fallback_strategy", {
                "max_retries": 5,
                "retry_strategy": "increase_gap",
                "gap_increment": 0.010,
                "final_action": "max_opening_lift",
                "emergency_gap": max_opening
            }),
            "source": "llm"
        }

    def plan_release(self, context: Dict[str, Any],
                    attempt_idx: int = 0,
                    feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM planning for release phase: brick release
        Input: integrated context
        Output: { "release_control": {...}, "contact_monitoring": {...}, "strategy": {...}, "raw": <llm_text>, "source": "llm|fallback" }
        """
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("release", {})
            return {
                **result,
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }
        
        if not self.enabled or (self.client is None):
            current_gap = context.get("current_gap", 0.105)
            max_opening = context["gripper"].get("max_opening", 0.200)
            target_gap = max(current_gap + 0.020, 0.105 * 1.2)
            target_gap = min(target_gap, max_opening * 0.9)
            
            plan_strategy = {
                "release_control": {
                    "action_type": "gradual_release",
                    "target_gap": target_gap,
                    "release_speed": "medium",
                    "initial_gap": current_gap,
                    "increment_step": 0.010,
                    "max_gap": max_opening * 0.9,
                    "tolerance": 0.001,
                    "description": "Classical progressive release"
                },
                "contact_monitoring": {
                    "detection_method": "combined_check",
                    "contact_threshold": 0.5,
                    "detach_verification": True,
                    "anti_stick_enabled": True,
                    "lift_distance": 0.010
                },
                "position_adjustment": {
                    "xy_maintain": True,
                    "z_lift_after": 0.015,
                    "orientation_lock": True,
                    "retreat_distance": 0.050,
                    "safe_height": 0.200,
                    "path_planning": "straight"
                },
                "safety_verification": {
                    "success_criteria": ["zero_contact", "no_constraint"],
                    "verification_timeout": 2.0,
                    "force_monitoring": True,
                    "emergency_procedures": ["large_opening", "force_detach"]
                },
                "fallback_strategy": {
                    "max_retries": 5,
                    "retry_strategy": "increase_gap",
                    "gap_increment": 0.010,
                    "final_action": "max_opening_lift",
                    "emergency_gap": max_opening
                },
                "source": "fallback(plan)"
            }
            return {
                **plan_strategy,
                "raw": "LLM disabled -> use classical release.",
            }
        
        current_gap = context.get("current_gap", 0.105)
        max_opening = context["gripper"].get("max_opening", 0.200)
        target_gap = max(current_gap + 0.020, 0.105 * 1.2)
        sys, usr = self._prompt_release(context, attempt_idx, feedback)
        raw = self.client.complete(sys, usr)
        result = self._postprocess_release(raw, current_gap, max_opening, target_gap)
        
        # Debug output
        print(f"[LLM_RELEASE] ===== LLM Release Planning Results =====")
        print(f"[LLM_RELEASE] Action type: {result['release_control']['action_type']}")
        print(f"[LLM_RELEASE] Target gap: {result['release_control']['target_gap']:.6f}m")
        print(f"[LLM_RELEASE] Initial gap: {result['release_control']['initial_gap']:.6f}m")
        print(f"[LLM_RELEASE] Increment step: {result['release_control']['increment_step']:.6f}m")
        print(f"[LLM_RELEASE] Detection method: {result['contact_monitoring']['detection_method']}")
        print(f"[LLM_RELEASE] Contact threshold: {result['contact_monitoring']['contact_threshold']:.3f}N")
        print(f"[LLM_RELEASE] Anti-stick lift: {result['contact_monitoring']['lift_distance']:.6f}m")
        print(f"[LLM_RELEASE] Source: {result.get('source', 'llm')}")
        
        return {
            **result,
            "raw": raw
        }

    def _prompt_close(self, context: Dict[str, Any], attempt_idx: int = 0, feedback: str = "") -> Tuple[str, str]:
        return get_close_prompt(context, attempt_idx, feedback)

    def _postprocess_close(self, raw_response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process close phase LLM response"""
        try:
            result = json.loads(raw_response)
            
            required_fields = ["gripper_command", "tcp_adjustment", "attachment_strategy", "reasoning"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            if "tcp_adjustment" in result and result["tcp_adjustment"].get("enabled", False):
                offset = result["tcp_adjustment"].get("position_offset", [0, 0, 0])
                for i in range(3):
                    offset[i] = max(-0.01, min(0.01, offset[i]))
                result["tcp_adjustment"]["position_offset"] = offset
            
            if "attachment_strategy" in result:
                threshold = result["attachment_strategy"].get("contact_threshold", 5.0)
                result["attachment_strategy"]["contact_threshold"] = max(0.1, min(20.0, threshold))
            
            return result
        
        except Exception as e:
            print(f"[LLM_CLOSE] JSON parsing failed: {e}, using classical close strategy")
            return {
                "gripper_command": {
                    "action_type": "close_grasp",
                    "description": "Standard gripper closing"
                },
                "tcp_adjustment": {
                    "enabled": False,
                    "position_offset": [0, 0, 0],
                    "description": "No TCP adjustment"
                },
                "attachment_strategy": {
                    "use_contact_assist": True,
                    "contact_threshold": 5.0,
                    "description": "Use contact assist grasp"
                },
                "reasoning": "LLM parsing failed, using classical closing strategy",
                "compliance_check": {
                    "gripper_closed_successfully": True,
                    "brick_securely_grasped": True,
                    "tcp_centered_on_brick": True,
                    "attachment_confirmed": True,
                    "no_excessive_force": True
                },
                "source": "fallback"
            }

    def plan_close(self, context: Dict[str, Any], attempt_idx: int = 0, feedback: str = "") -> Dict[str, Any]:
        """Close phase planning - gripper closing to grasp brick"""
        if self.mode == "single_agent":
            if self._unified_plan_cache is None:
                self._unified_plan_cache = self.plan_unified(context, attempt_idx, feedback)
            result = self._unified_plan_cache.get("close", {})
            return {
                **result,
                "raw": self._unified_plan_cache.get("raw", ""),
                "source": result.get("source", "single_agent")
            }
        
        if not self.enabled:
            return {
                "gripper_command": {
                    "action_type": "close_grasp",
                    "description": "Classical gripper closing"
                },
                "tcp_adjustment": {
                    "enabled": False,
                    "position_offset": [0, 0, 0],
                    "description": "No TCP adjustment"
                },
                "attachment_strategy": {
                    "use_contact_assist": True,
                    "contact_threshold": 5.0,
                    "description": "Standard contact assist"
                },
                "reasoning": "LLM disabled -> use classical close.",
                "source": "fallback(disabled)"
            }
        
        sys, usr = self._prompt_close(context, attempt_idx, feedback)
        raw = self.client.complete(sys, usr)
        result = self._postprocess_close(raw, context)
        
        # Debug output
        print(f"[LLM_CLOSE] ===== LLM Close Planning Results =====")
        print(f"[LLM_CLOSE] Action type: {result['gripper_command']['action_type']}")
        print(f"[LLM_CLOSE] TCP adjustment: {result['tcp_adjustment']['enabled']}")
        if result['tcp_adjustment']['enabled']:
            print(f"[LLM_CLOSE] Position offset: {result['tcp_adjustment']['position_offset']}")
        print(f"[LLM_CLOSE] Contact assist: {result['attachment_strategy']['use_contact_assist']}")
        print(f"[LLM_CLOSE] Contact threshold: {result['attachment_strategy']['contact_threshold']:.1f}N")
        print(f"[LLM_CLOSE] Source: {result.get('source', 'llm')}")
        
        return {
            **result,
            "raw": raw
        }

    def reset_cache(self):
        """Clear unified planning cache (called when processing new brick)"""
        self._unified_plan_cache = None

    def plan_unified(self, context: Dict[str, Any],
                    attempt_idx: int = 0,
                    feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Single Agent mode: plan all phases at once
        
        Returns:
            Dictionary containing all 6 phase plans
        """
        if not self.enabled or (self.client is None):
            return self._fallback_classical_planning(context)
        
        try:
            sys, usr = get_single_agent_prompt(context, attempt_idx, feedback)
            
            raw = self.client.complete(sys, usr)
            
            result = _extract_json(raw) or {}
            
            required_phases = ["pre_grasp", "descend", "close", "lift", "place", "release"]
            missing = [p for p in required_phases if p not in result]
            
            if missing:
                print(f"[SINGLE_AGENT] Missing phases: {missing}, using fallback strategy")
                return self._fallback_classical_planning(context)
            
            validated = self._validate_unified_result(result, context)
            
            print("[SINGLE_AGENT] Unified planning succeeded")
            return {
                **validated,
                "raw": raw,
                "source": "single_agent_llm",
                "mode": "single_agent"
            }
            
        except Exception as e:
            print(f"[SINGLE_AGENT] Unified planning failed: {e}")
            return self._fallback_classical_planning(context)

    def _validate_unified_result(self, result: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process unified planning result"""
        from math3d.transforms import align_topdown_to_brick
        
        brick_pos = context["brick"]["pos"]
        brick_size = context["brick"]["size_LWH"]
        brick_yaw = context["brick"]["rpy"][2]
        
        approach = context.get("constraints", {}).get("approach_clearance", 0.08)
        lift_clearance = context.get("constraints", {}).get("lift_clearance", 0.15)
        ground_z = context.get("constraints", {}).get("ground_z", 0.0)
        tip_length = context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"]
        goal_xy = context.get("goal", {}).get("target_xy", [brick_pos[0], brick_pos[1]])
        goal_z_top = context.get("goal", {}).get("target_z_top", brick_pos[2] + brick_size[2]/2)
        goal_yaw = context.get("goal", {}).get("target_yaw", brick_yaw)
        
        validated = {}
        
        z_top = brick_pos[2] + brick_size[2]/2
        validated["pre_grasp"] = {
            "pose": {
                "xyz": [
                    float(brick_pos[0]),
                    float(brick_pos[1]),
                    float(z_top + approach + tip_length)
                ],
                "rpy": align_topdown_to_brick(brick_yaw)
            },
            "source": "single_agent_validated"
        }
        
        ground_z = context.get("constraints", {}).get("ground_z", 0.0)
        desired_bottom = max(ground_z + 0.003, brick_pos[2] - 0.0175)
        validated["descend"] = {
            "pose": {
                "xyz": [
                    float(brick_pos[0]),
                    float(brick_pos[1]),
                    float(desired_bottom + tip_length)
                ],
                "rpy": validated["pre_grasp"]["pose"]["rpy"]
            },
            "target_gap": float(brick_size[1] + context["gripper"]["open_clearance"] + 0.008),
            "source": "single_agent_validated"
        }
        
        validated["close"] = {
            "gripper_command": result.get("close", {}).get("gripper_command", {
                "action_type": "close_grasp"
            }),
            "tcp_adjustment": result.get("close", {}).get("tcp_adjustment", {
                "enabled": False
            }),
            "attachment_strategy": result.get("close", {}).get("attachment_strategy", {
                "use_contact_assist": True
            }),
            "source": "single_agent_validated"
        }
        
        validated["lift"] = {
            "pose": {
                "xyz": [
                    float(brick_pos[0]),
                    float(brick_pos[1]),
                    float(brick_pos[2] + brick_size[2]/2 + lift_clearance + tip_length)
                ],
                "rpy": validated["pre_grasp"]["pose"]["rpy"]
            },
            "lift_height": float(lift_clearance),
            "source": "single_agent_validated"
        }
        validated["place"] = {
            "approach_phase": {
                "target_pose": {
                    "xyz": [
                        float(goal_xy[0]),
                        float(goal_xy[1]),
                        float(goal_z_top + approach + tip_length)
                    ],
                    "rpy": [0.0, 0.0, float(goal_yaw)]
                },
                "approach_height": approach
            },
            "descent_phase": {
                "action_type": "gradual_descent_to_support",
                "start_height": float(goal_z_top + max(0.006, 0.010)),
                "step_size": 0.004,
                "max_descent_range": 0.030,
                "orientation_control": {
                    "target_rpy": [0.0, 0.0, float(goal_yaw)],
                    "roll_tolerance": 0.002,
                    "pitch_tolerance": 0.002,
                    "yaw_tolerance": 0.005,
                    "strict_parallel": True
                }
            },
            "contact_detection": result.get("place", {}).get("contact_detection", {
                "support_ids": context.get("goal", {}).get("support_ids", [0]),
                "detection_method": "collision_monitoring"
            }),
            "source": "single_agent_validated"
        }
        
        required_gap = brick_size[1] + context["gripper"]["open_clearance"]
        validated["release"] = {
            "release_control": {
                "action_type": "gradual_release",
                "target_gap": float(max(required_gap * 1.2, required_gap + 0.020)),
                "increment_step": 0.008
            },
            "contact_monitoring": result.get("release", {}).get("contact_monitoring", {
                "detection_method": "force_and_contact",
                "separation_verification": True
            }),
            "retreat_strategy": result.get("release", {}).get("retreat_strategy", {
                "lift_distance": 0.015
            }),
            "source": "single_agent_validated"
        }
        
        return validated

    def _fallback_classical_planning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classical planning fallback strategy (used when single agent fails)"""
        from math3d.transforms import align_topdown_to_brick
        
        print("[PLANNING] Using classical geometric planning fallback strategy")
        
        brick_pos = context["brick"]["pos"]
        brick_size = context["brick"]["size_LWH"]
        brick_yaw = context["brick"]["rpy"][2]
        approach = context.get("constraints", {}).get("approach_clearance", 0.08)
        lift_clearance = context.get("constraints", {}).get("lift_clearance", 0.15)
        ground_z = context.get("constraints", {}).get("ground_z", 0.0)
        tip_length = context["gripper"]["measured_tip_length"] or context["gripper"]["tip_length_guess"]
        
        goal_xy = context.get("goal", {}).get("target_xy", [brick_pos[0], brick_pos[1]])
        goal_z_top = context.get("goal", {}).get("target_z_top", brick_pos[2] + brick_size[2]/2)
        goal_yaw = context.get("goal", {}).get("target_yaw", brick_yaw)
        
        return {
            "pre_grasp": {
                "pose": {
                    "xyz": [brick_pos[0], brick_pos[1], brick_pos[2] + brick_size[2]/2 + approach + tip_length],
                    "rpy": align_topdown_to_brick(brick_yaw)
                },
                "source": "classical_fallback"
            },
            "descend": {
                "pose": {
                    "xyz": [brick_pos[0], brick_pos[1], 
                           max(ground_z + 0.003, brick_pos[2] - 0.0175) + tip_length],
                    "rpy": align_topdown_to_brick(brick_yaw)
                },
                "target_gap": brick_size[1] + context["gripper"]["open_clearance"] + 0.008,
                "source": "classical_fallback"
            },
            "close": {
                "gripper_command": {"action_type": "close_grasp"},
                "source": "classical_fallback"
            },
            "lift": {
                "pose": {
                    "xyz": [brick_pos[0], brick_pos[1], 
                           brick_pos[2] + brick_size[2]/2 + lift_clearance + tip_length],
                    "rpy": align_topdown_to_brick(brick_yaw)
                },
                "lift_height": lift_clearance,
                "source": "classical_fallback"
            },
            "place": {
                "approach_phase": {
                    "target_pose": {
                        "xyz": [goal_xy[0], goal_xy[1], goal_z_top + approach + tip_length],
                        "rpy": [0.0, 0.0, goal_yaw]
                    },
                    "approach_height": approach
                },
                "descent_phase": {
                    "start_height": goal_z_top + 0.010,
                    "step_size": 0.004,
                    "max_descent_range": 0.030,
                    "orientation_control": {
                        "target_rpy": [0.0, 0.0, goal_yaw],
                        "roll_tolerance": 0.002,
                        "pitch_tolerance": 0.002,
                        "yaw_tolerance": 0.005,
                        "strict_parallel": True
                    }
                },
                "source": "classical_fallback"
            },
            "release": {
                "release_control": {
                    "target_gap": (brick_size[1] + context["gripper"]["open_clearance"]) * 1.2,
                    "increment_step": 0.008
                },
                "source": "classical_fallback"
            },
            "source": "classical_fallback",
            "mode": "fallback"
        }



# Use setattr to set method alias at class level
def _set_method_alias():
    LLMPlanner.plan_pre_place = LLMPlanner.plan_place

_set_method_alias()
