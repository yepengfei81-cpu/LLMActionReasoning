import math
import numpy as np
import pybullet as p
from typing import Tuple, Dict, List
from math3d.transforms import angle_difference

class AngleOptimizer:
    def __init__(self, robot_model, pb_client):
        self.rm = robot_model
        self.pb = pb_client
        
        # Angle control parameters
        self.angle_tolerance = 0.03 
        self.max_correction_iterations = 10
        self.correction_step_size = 0.01  
        self.min_correction_threshold = 0.005 
        
        # Stacking detection parameters
        self.stacking_detection_height = 0.08  
        self.stacking_angle_tolerance = 0.05  
        self.stacking_force_multiplier = 0.5  
        
    def get_brick_orientation(self, brick_id: int) -> float:
        pos, ori = self.pb.getBasePositionAndOrientation(brick_id)
        euler = self.pb.getEulerFromQuaternion(ori)
        return euler[2]  # yaw
    
    def calculate_angle_error(self, current_yaw: float, target_yaw: float) -> float:
        return angle_difference(current_yaw, target_yaw)
    
    def is_stacking_placement(self, brick_id: int) -> bool:
        try:
            pos, _ = self.pb.getBasePositionAndOrientation(brick_id)
            return pos[2] > self.stacking_detection_height
        except:
            return False
    
    def get_effective_tolerance(self, brick_id: int) -> float:
        if self.is_stacking_placement(brick_id):
            return self.stacking_angle_tolerance  
        return self.angle_tolerance
    
    def optimize_placement_angle(self, brick_id: int, target_yaw: float, 
                                gripper_controller, tcp_position: Tuple[float, float, float],
                                verbose: bool = True) -> Tuple[bool, float, List[Dict]]:
        # if verbose:
        #     print(f"[ANGLE_OPT] Starting angle optimization, target: {math.degrees(target_yaw):.2f}°")
        
        # Check if it is a stacking scenario
        is_stacking = self.is_stacking_placement(brick_id)
        effective_tolerance = self.get_effective_tolerance(brick_id)
        force_multiplier = self.stacking_force_multiplier if is_stacking else 1.0
        
        # if verbose:
        #     placement_type = "Stacking" if is_stacking else "Base layer"
            # print(f"[ANGLE_OPT] {placement_type} placement, angle tolerance: {math.degrees(effective_tolerance):.1f}°")
        
        optimization_data = []
        
        for iteration in range(self.max_correction_iterations):
            # Get current brick orientation
            current_yaw = self.get_brick_orientation(brick_id)
            angle_error = self.calculate_angle_error(current_yaw, target_yaw)
            
            step_data = {
                'iteration': iteration,
                'current_yaw_deg': math.degrees(current_yaw),
                'target_yaw_deg': math.degrees(target_yaw),
                'angle_error_deg': math.degrees(angle_error),
                'angle_error_rad': angle_error
            }
            
            # if verbose:
            #     print(f"[ANGLE_OPT] Iter {iteration}: current={math.degrees(current_yaw):.2f}°, "
            #           f"error={math.degrees(angle_error):.2f}°")
            
            if abs(angle_error) < effective_tolerance:
                step_data['converged'] = True
                optimization_data.append(step_data)
                # if verbose:
                #     print(f"[ANGLE_OPT] Converged! Final error: {math.degrees(angle_error):.2f}°")
                return True, angle_error, optimization_data
            
            # If error is too small, skip correcting
            if abs(angle_error) < self.min_correction_threshold:
                step_data['too_small_to_correct'] = True
                optimization_data.append(step_data)
                break
            
            # Calculate correction action
            correction_success = self._apply_angle_correction(
                brick_id, angle_error, gripper_controller, tcp_position, 
                force_multiplier, verbose
            )
            
            step_data['correction_applied'] = correction_success
            optimization_data.append(step_data)
            
            if not correction_success:
                # if verbose:
                #     print(f"[ANGLE_OPT] Failed to apply correction at iteration {iteration}")
                break
            
            for _ in range(10):
                self.pb.stepSimulation()
        
        final_yaw = self.get_brick_orientation(brick_id)
        final_error = self.calculate_angle_error(final_yaw, target_yaw)
        
        success = abs(final_error) < effective_tolerance
        
        # if verbose:
        #     print(f"[ANGLE_OPT] Optimization {'succeeded' if success else 'failed'}. "
        #           f"Final error: {math.degrees(final_error):.2f}°")
        
        return success, final_error, optimization_data
    
    def _apply_angle_correction(self, brick_id: int, angle_error: float, 
                              gripper_controller, tcp_position: Tuple[float, float, float],
                              force_multiplier: float = 1.0, verbose: bool = True) -> bool:
        """
        Apply angle correction
        Correct brick angle by fine-tuning TCP orientation
        """
        try:
            correction_magnitude = min(abs(angle_error), self.correction_step_size)
            correction_direction = 1 if angle_error > 0 else -1
            
            # if verbose:
            #     print(f"[ANGLE_OPT] Applying correction: {math.degrees(correction_magnitude * correction_direction):.2f}°")

            current_gap = gripper_controller.last_sym_theta() * 0.2 
            
            # Fine-tune asymmetry of left and right fingers to affect brick orientation
            asymmetry_factor = correction_direction * 0.01 * force_multiplier  # Adjust force based on stacking condition
            base_force = 20.0 * force_multiplier  # Base force adjustment
            restore_force = 40.0 * force_multiplier  # Restore force adjustment
            
            # Apply asymmetric control
            for joint_id in self.rm.finger_joint_indices:
                joint_info = self.pb.getJointInfo(self.rm.id, joint_id)
                joint_name = joint_info[1].decode('utf-8').lower()
                
                if 'left' in joint_name:
                    target_pos = -(current_gap/0.2) * (1 + asymmetry_factor)
                elif 'right' in joint_name:
                    target_pos = (current_gap/0.2) * (1 - asymmetry_factor)
                else:
                    continue
                    
                self.pb.setJointMotorControl2(
                    self.rm.id,
                    joint_id,
                    self.pb.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=base_force
                )
            
            for _ in range(15):
                self.pb.stepSimulation()
            
            for joint_id in self.rm.finger_joint_indices:
                joint_info = self.pb.getJointInfo(self.rm.id, joint_id)
                joint_name = joint_info[1].decode('utf-8').lower()
                
                if 'left' in joint_name:
                    target_pos = -(current_gap/0.2)
                elif 'right' in joint_name:
                    target_pos = (current_gap/0.2)
                else:
                    continue
                    
                self.pb.setJointMotorControl2(
                    self.rm.id,
                    joint_id,
                    self.pb.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=restore_force
                )
            
            for _ in range(10):
                self.pb.stepSimulation()
            
            return True
            
        except Exception as e:
            # if verbose:
            #     print(f"[ANGLE_OPT] Error applying correction: {e}")
            return False
    

    def enhanced_release_with_angle_control(self, brick_id: int, target_yaw: float, 
                                          gripper_controller, tcp_position: Tuple[float, float, float], vf_checker=None) -> Tuple[bool, float, Dict]:
        """
        Enhanced release, combining angle control and force feedback
        """
        # print("[ANGLE_OPT] Starting enhanced release with angle control...")
        
        # Pre-optimize angle
        pre_optimization_success, pre_error, pre_data = self.optimize_placement_angle(
            brick_id, target_yaw, gripper_controller, tcp_position
        )
        
        # Execute force feedback release
        try:
            force_success, final_gap = gripper_controller.force_feedback_release(brick_id, vf_checker)
            force_data = []  
        except Exception as e:
            # print(f"[ANGLE_OPT] Force feedback error: {e}, using standard release")
            force_success = True
            final_gap = 0.15
            force_data = []
        
        # Post-optimize angle
        post_optimization_success, post_error, post_data = False, pre_error, []
        if force_success and abs(pre_error) > self.angle_tolerance:
            post_optimization_success, post_error, post_data = self.optimize_placement_angle(
                brick_id, target_yaw, gripper_controller, tcp_position
            )
        
        final_success = force_success and abs(post_error) < self.angle_tolerance
        
        optimization_summary = {
            'pre_optimization': {
                'success': pre_optimization_success,
                'error_deg': math.degrees(pre_error),
                'iterations': len(pre_data)
            },
            'force_release': {
                'success': force_success,
                'final_gap': final_gap,
                'force_samples': len(force_data) if force_data else 0
            },
            'post_optimization': {
                'success': post_optimization_success,
                'error_deg': math.degrees(post_error),
                'iterations': len(post_data)
            },
            'final_success': final_success,
            'final_angle_error_deg': math.degrees(post_error)
        }
        
        # print(f"[ANGLE_OPT] Enhanced release {'succeeded' if final_success else 'failed'}. "
        #       f"Final angle error: {math.degrees(post_error):.2f}°")
        
        return final_success, post_error, optimization_summary
