import pybullet as p
from env.pyb_env import BulletEnv
from modules.grasp_module import GraspModule
from control.gripper import GripperHelper
from modules.state_verifier import StateVerifier
from modules.motion_executor import MotionExecutor
from modules.qp_scheduler import QPTaskScheduler

def main():
    env = BulletEnv("configs/kuka_six_bricks.yaml", use_gui=True)
    rm  = env.robot_model

    gripper = GripperHelper(rm)
    grasp = GraspModule(env) 
    assist_cfg = env.cfg.get("assist_grasp", {})

    ground_z = env.get_ground_top()

    placement_sequence = env.get_brick_placement_sequence()
    # placed_supports = []  
    placed_bricks_info = []
    success_count = 0

    qp_scheduler = QPTaskScheduler(env, threshold_low=0.005, threshold_critical=0.015)
    
    for seq_idx, brick_idx in enumerate(placement_sequence):
        brick_id = env.brick_ids[brick_idx]
        goal_pose = env.compute_goal_pose_from_layout(brick_idx)
        vf = StateVerifier(env, rm, gripper, brick_id)
        motion = MotionExecutor(env, rm, gripper, vf)

        brick_state = env.get_brick_state(brick_id=brick_id)
        wps, aux = grasp.plan(brick_state, [*goal_pose], ground_z, brick_id=brick_id)

        print(f"\n{'='*60}")
        print(f"Start processing brick {seq_idx+1}/{len(placement_sequence)}")
        print(f"Brick Index: {brick_idx}, Brick ID: {brick_id}")
        print(f"Target Pose: {goal_pose}")
        

        level_name = env.get_level_name(brick_idx)
        support_ids = env.get_related_support_ids(brick_idx)
        
        print(f"Level: {level_name}")
        print(f"Support IDs: {support_ids}")
        if placed_bricks_info:
            remaining = len(placement_sequence) - seq_idx
            decision = qp_scheduler.solve(
                current_task_idx=seq_idx,
                placed_bricks=placed_bricks_info,
                remaining_bricks=remaining
            )
            
            print(f"[QP Scheduler] Decision: {decision['action']}, Reason: {decision['reason']}")
            
            if decision["action"] == "REPAIR":
                print(f"⚠️  Interrupting to repair brick {decision['repair_brick_id']}")
                print(f"    Deviation: {decision['deviation']*1000:.2f} mm")
                repair_ok = repair_brick(
                    env, rm, gripper, grasp, 
                    decision["repair_brick_id"], 
                    decision["repair_target"],
                    ground_z, assist_cfg,
                    placed_bricks_info=placed_bricks_info  # 新增参数
                )
                
                if repair_ok:
                    print(f"✅ Repair successful!")
                else:
                    print(f"❌ Repair failed!")
        print(f"{'='*60}")
        
        ok = motion.execute_fsm(wps, aux, assist_cfg, brick_id, env.ground_id, support_ids=support_ids)
        
        if ok:
            success_count += 1
            # placed_supports.append(brick_id)
            placed_bricks_info.append({
                "brick_id": brick_id,
                "expected_pos": goal_pose[:3],
                "level": seq_idx  # 简单用序号表示层级
            })            
            print(f"✅ [SUCCESS] {level_name} (Seq {seq_idx+1}) Placement Successful!")
        else:
            print(f"❌ [FAILED] {level_name} (Seq {seq_idx+1}) Placement Failed!")
        
        print(f"[Progress] Completed: {success_count}/{seq_idx+1}, Success Rate: {success_count/(seq_idx+1)*100:.1f}%")

        # Allow time for object to settle (read from config)
        settle_steps = int(env.cfg["timing"].get("brick_settle_sec", 2.0) / env.dt)
        for _ in range(settle_steps):
            env.step(1)

        # Reset between tasks (for next brick)
        if seq_idx < (len(placement_sequence) - 1):
            print(f"Preparing for next brick, resetting...")
            motion.reset_between_tasks()
            reset_steps = int(env.cfg["timing"].get("reset_wait_sec", 1.5) / env.dt)
            for _ in range(reset_steps):
                env.step(1)

    print(f"\n{'='*60}")
    print(f"🎯 Stacking task completed!")
    print(f"📊 Final Statistics:")
    print(f"   - Total Bricks: {len(env.brick_ids)}")
    print(f"   - Successful Placements: {success_count}")
    print(f"   - Failed Count: {len(env.brick_ids) - success_count}")
    print(f"   - Total Success Rate: {success_count/len(env.brick_ids)*100:.1f}%")
    
    if success_count == len(env.brick_ids):
        print("🎉 Perfect! All bricks placed successfully!")
    elif success_count >= len(env.brick_ids) * 0.8:
        print("👍 Great! Most bricks placed successfully!")
    else:
        print("🤔 Parameters and strategy need further optimization.")
    
    print(f"{'='*60}")
    print("Keeping scene for inspection...")
    
    final_steps = int(env.cfg["timing"].get("final_wait_sec", 10.0) / env.dt)
    for _ in range(final_steps):
        env.step(1)

    env.disconnect()

def repair_brick(env, rm, gripper, grasp, brick_id, target_pos, ground_z, assist_cfg, placed_bricks_info=None):
    """
    修复偏移的砖块
    重新抓取并放置到目标位置
    """
    from modules.state_verifier import StateVerifier
    from modules.motion_executor import MotionExecutor
    
    print(f"[Repair] Starting repair for brick {brick_id}")
    brick_state = env.get_brick_state(brick_id=brick_id)
    
    goal_pose = list(target_pos) + [0, 0, 0]
    wps, aux = grasp.plan(brick_state, goal_pose, ground_z, brick_id=brick_id)
    
    vf = StateVerifier(env, rm, gripper, brick_id)
    motion = MotionExecutor(env, rm, gripper, vf)
    
    # 构建支撑列表：地面 + 其他已放置的砖块（排除当前修复的砖块）
    support_ids = [env.ground_id]
    if placed_bricks_info:
        for info in placed_bricks_info:
            if info["brick_id"] != brick_id:
                support_ids.append(info["brick_id"])
    
    ok = motion.execute_fsm(wps, aux, assist_cfg, brick_id, env.ground_id, support_ids=support_ids)
    
    return ok
if __name__ == "__main__":
    main()
