import numpy as np
import pybullet as p

class ForceController:
    def __init__(self, pb_client, finger_joint_ids):
        self.pb = pb_client
        self.finger_joint_ids = finger_joint_ids
        self.force_history = []
        self.contact_force_threshold = 5.0  # N
        self.force_change_threshold = 2.0   # N
        
    def get_joint_forces(self, robot_id):
        """Get gripper joint forces"""
        forces = []
        for joint_id in self.finger_joint_ids:
            joint_state = self.pb.getJointState(robot_id, joint_id)
            # joint_state[3] is the joint reaction force
            forces.append(abs(joint_state[3]))
        return forces
    
    def get_contact_forces(self, robot_id, target_body_id):
        """Get contact force with target object"""
        contact_points = self.pb.getContactPoints(robot_id, target_body_id)
        total_force = 0.0
        force_positions = []
        
        for contact in contact_points:
            force_magnitude = abs(contact[9])
            total_force += force_magnitude
            force_positions.append({
                'position': contact[5],  
                'force': force_magnitude,
                'link': contact[3]  
            })
            
        return total_force, force_positions
    
    def monitor_release_forces(self, robot_id, brick_id, duration=0.5):
        """Monitor force changes during release"""
        start_time = self.pb.getPhysicsEngineParameters()['fixedTimeStep'] * self.pb.getPhysicsEngineParameters()['numSubSteps']
        force_samples = []
        
        for _ in range(int(duration * 240)): 
            total_force, contact_positions = self.get_contact_forces(robot_id, brick_id)
            joint_forces = self.get_joint_forces(robot_id)
            
            force_samples.append({
                'contact_force': total_force,
                'joint_forces': joint_forces,
                'contact_count': len(contact_positions)
            })
            
            self.pb.stepSimulation()
            
        return force_samples
    
    def detect_force_drop(self, force_samples, window_size=10):
        """Detect sudden force drop (release signal)"""
        if len(force_samples) < window_size * 2:
            return False, 0
            
        # Calculate moving average force
        recent_forces = [s['contact_force'] for s in force_samples[-window_size:]]
        previous_forces = [s['contact_force'] for s in force_samples[-window_size*2:-window_size]]
        
        recent_avg = np.mean(recent_forces)
        previous_avg = np.mean(previous_forces)
        
        force_drop = previous_avg - recent_avg
        
        if force_drop > self.force_change_threshold and recent_avg < self.contact_force_threshold:
            return True, force_drop
            
        return False, force_drop
    
    def adaptive_release_strategy(self, robot_id, brick_id, gripper_controller):
        """Adaptive release strategy based on force feedback"""
        print("[FORCE_FEEDBACK] Starting adaptive release...")
        
        # Record initial contact force
        initial_force, _ = self.get_contact_forces(robot_id, brick_id)
        print(f"[FORCE_FEEDBACK] Initial contact force: {initial_force:.2f}N")
        
        # Gradually release, monitor force changes
        current_gap = gripper_controller.get_current_width()
        target_gap = current_gap + 0.020 
        step_size = 0.002  
        
        force_samples = []
        steps = 0
        max_steps = int((target_gap - current_gap) / step_size)
        
        while current_gap < target_gap and steps < max_steps:
            # Increase opening with small steps
            next_gap = min(current_gap + step_size, target_gap)
            gripper_controller.set_width(next_gap)
            
            # Wait briefly for stabilization
            for _ in range(6):  
                self.pb.stepSimulation()
            
            # Monitor force changes
            current_force, contact_positions = self.get_contact_forces(robot_id, brick_id)
            joint_forces = self.get_joint_forces(robot_id)
            
            force_samples.append({
                'step': steps,
                'gap': next_gap,
                'contact_force': current_force,
                'joint_forces': joint_forces,
                'contact_count': len(contact_positions)
            })
            
            if len(contact_positions) == 0:
                print(f"[FORCE_FEEDBACK] Release detected at step {steps}, gap {next_gap:.3f}")
                return True, next_gap, force_samples
            
            if len(force_samples) >= 10:
                force_dropped, drop_amount = self.detect_force_drop(force_samples)
                if force_dropped:
                    print(f"[FORCE_FEEDBACK] Force drop detected: {drop_amount:.2f}N at step {steps}")
                    final_gap = next_gap + 0.005
                    gripper_controller.set_width(final_gap)
                    for _ in range(12):  
                        self.pb.stepSimulation()
                    return True, final_gap, force_samples
            
            current_gap = gripper_controller.get_current_width()
            steps += 1
            
            # Report every 5 steps
            if steps % 5 == 0:
                print(f"[FORCE_FEEDBACK] Step {steps}/{max_steps}: gap={next_gap:.3f}, force={current_force:.2f}N, contacts={len(contact_positions)}")
        
        print(f"[FORCE_FEEDBACK] Release completed after {steps} steps")
        return True, current_gap, force_samples
