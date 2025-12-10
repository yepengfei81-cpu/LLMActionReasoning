import pybullet as p
from env.pyb_env import BulletEnv
from modules.grasp_module import GraspModule
from control.gripper import GripperHelper
from modules.state_verifier import StateVerifier
from modules.motion_executor import MotionExecutor
from modules.qp_scheduler import QPTaskScheduler, TaskType, TaskItem


def main():
    # ============ 初始化 ============
    env = BulletEnv("configs/kuka_six_bricks.yaml", use_gui=True)
    rm = env.robot_model
    gripper = GripperHelper(rm)
    grasp = GraspModule(env)
    assist_cfg = env.cfg.get("assist_grasp", {})
    ground_z = env.get_ground_top()

    # ============ QP 调度器初始化 ============
    qp_scheduler = QPTaskScheduler(
        env, 
        threshold_low=0.05,      # 50mm 以下不修复
        threshold_critical=0.1   # 100mm 以上必须修复
    )

    # ============ 任务状态跟踪 ============
    original_sequence = env.get_brick_placement_sequence()
    
    # 已放置砖块信息
    placed_bricks_info = []
    
    # 已完成的砖块集合
    completed_bricks = set()
    
    # 统计
    success_count = 0
    failed_count = 0
    repair_count = 0
    temp_count = 0
    total_tasks_executed = 0
    
    # 任务队列
    task_queue = []
    
    # 当前任务索引
    original_idx = 0

    # ============ 主循环 ============
    while original_idx < len(original_sequence) or len(task_queue) > 0:
        
        # ======== 步骤 1: 规划/更新任务队列 ========
        if len(task_queue) == 0:
            if original_idx < len(original_sequence):
                remaining = original_sequence[original_idx:]
                current_brick_idx = remaining[0] if remaining else None
                
                qp_scheduler.update_placed_bricks(placed_bricks_info)
                
                task_queue = qp_scheduler.plan_task_sequence(
                    current_brick_idx=current_brick_idx,
                    remaining_sequence=remaining,
                    is_holding_brick=False
                )
            else:
                qp_scheduler.update_placed_bricks(placed_bricks_info)
                if qp_scheduler.should_replan():
                    bricks_to_repair = qp_scheduler.get_bricks_needing_repair()
                    for d in bricks_to_repair:
                        task_queue.append(TaskItem(
                            task_type=TaskType.REPAIR_PLACE,
                            brick_idx=d["brick_idx"],
                            brick_id=d["brick_id"],
                            target_pos=tuple(d["expected_pos"]),
                            target_orn=d["expected_orn"],
                            level=d["level"],
                            priority=0,
                            reason="Final repair pass"
                        ))
                else:
                    break
        
        if len(task_queue) == 0:
            break
        
        # ======== 步骤 2: 取出下一个任务 ========
        current_task = task_queue.pop(0)
        total_tasks_executed += 1
        
        brick_idx = current_task.brick_idx
        brick_id = current_task.brick_id
        goal_pose = current_task.to_goal_pose()
        task_type = current_task.task_type
        level = current_task.level
        is_temp = current_task.is_temp
        level_name = env.get_level_name(brick_idx)
        
        print(f"\n{'='*60}")
        print(f"[TASK #{total_tasks_executed}] {task_type.value.upper()}")
        print(f"   Brick Index: {brick_idx}, Brick ID: {brick_id}")
        print(f"   Level: {level_name}")
        print(f"   Target Pose: {goal_pose}")
        print(f"   Is Temp Position: {is_temp}")
        print(f"   Reason: {current_task.reason}")
        print(f"   Queue remaining: {len(task_queue)}")
        
        # ======== 步骤 3: 执行任务 ========
        vf = StateVerifier(env, rm, gripper, brick_id)
        motion = MotionExecutor(env, rm, gripper, vf)
        
        brick_state = env.get_brick_state(brick_id=brick_id)
        wps, aux = grasp.plan(brick_state, [*goal_pose], ground_z, brick_id=brick_id)
        
        # 临时放置使用地面作为支撑
        if task_type == TaskType.TEMP_PLACE:
            support_ids = [env.ground_id]
        else:
            support_ids = env.get_related_support_ids(brick_idx)
        
        ok = motion.execute_fsm(wps, aux, assist_cfg, brick_id, env.ground_id, support_ids=support_ids)
        
        # ======== 步骤 4: 处理结果 ========
        if ok:
            if task_type == TaskType.TEMP_PLACE:
                temp_count += 1
                print(f"📦 [TEMP SUCCESS] Brick idx={brick_idx} moved to temp position!")
                
                # 标记砖块在临时位置
                qp_scheduler.mark_brick_in_temp(brick_idx, goal_pose[:3])
                
                # 更新 placed_bricks_info（临时位置也要记录）
                exists = False
                for info in placed_bricks_info:
                    if info["brick_idx"] == brick_idx:
                        info["expected_pos"] = goal_pose[:3]
                        info["expected_orn"] = goal_pose[3:]
                        info["is_temp"] = True
                        exists = True
                        break
                if not exists:
                    placed_bricks_info.append({
                        "brick_id": brick_id,
                        "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3],
                        "expected_orn": goal_pose[3:],
                        "level": level,
                        "is_temp": True
                    })
                    
            elif task_type == TaskType.REPAIR_PLACE:
                repair_count += 1
                print(f"✅ [REPAIR SUCCESS] Brick idx={brick_idx} repaired!")
                
                # 如果之前在临时位置，取消标记
                qp_scheduler.unmark_brick_from_temp(brick_idx)
                
                # 更新信息
                updated = False
                for info in placed_bricks_info:
                    if info["brick_idx"] == brick_idx:
                        info["expected_pos"] = goal_pose[:3]
                        info["expected_orn"] = goal_pose[3:]
                        info["is_temp"] = False
                        updated = True
                        break
                
                if not updated:
                    placed_bricks_info.append({
                        "brick_id": brick_id,
                        "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3],
                        "expected_orn": goal_pose[3:],
                        "level": level,
                        "is_temp": False
                    })
                    
            else:  # NORMAL_PLACE
                success_count += 1
                completed_bricks.add(brick_idx)
                print(f"✅ [SUCCESS] {level_name} (brick {brick_idx}) Placement Successful!")
                
                exists = any(info["brick_idx"] == brick_idx for info in placed_bricks_info)
                if not exists:
                    placed_bricks_info.append({
                        "brick_id": brick_id,
                        "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3],
                        "expected_orn": goal_pose[3:],
                        "level": level,
                        "is_temp": False
                    })
                
                if brick_idx in original_sequence[original_idx:]:
                    for i, idx in enumerate(original_sequence[original_idx:]):
                        if idx == brick_idx:
                            original_idx = original_idx + i + 1
                            break
        else:
            failed_count += 1
            print(f"❌ [FAILED] {level_name} (brick {brick_idx}) Failed!")
        
        # 进度
        print(f"[Progress] Success: {success_count}, Failed: {failed_count}, "
              f"Repairs: {repair_count}, Temp moves: {temp_count}, "
              f"Total executed: {total_tasks_executed}")
        
        # 等待稳定
        settle_sec = env.cfg["timing"].get("brick_settle_sec", 2.0)
        env.step(int(settle_sec / env.dt))
        
        # ======== 步骤 5: 检查是否需要重新规划 ========
        qp_scheduler.update_placed_bricks(placed_bricks_info)
        
        if qp_scheduler.should_replan() and len(task_queue) > 0:
            print(f"\n[QP] ⚠️ Deviation detected! Re-planning task sequence...")
            
            next_brick_idx = task_queue[0].brick_idx if task_queue else None
            
            remaining_original = [idx for idx in original_sequence[original_idx:] 
                                 if idx not in completed_bricks]
            
            task_queue = qp_scheduler.plan_task_sequence(
                current_brick_idx=next_brick_idx,
                remaining_sequence=remaining_original,
                is_holding_brick=False
            )
        
        # 重置机械臂
        if len(task_queue) > 0 or original_idx < len(original_sequence):
            print("Preparing for next task, resetting...")
            motion.reset_between_tasks()
            reset_sec = env.cfg["timing"].get("reset_wait_sec", 1.5)
            env.step(int(reset_sec / env.dt))

    # ============ 结束统计 ============
    print(f"\n{'='*60}")
    print(f"🎯 Stacking task completed!")
    print(f"📊 Final Statistics:")
    print(f"   - Original Tasks: {len(original_sequence)}")
    print(f"   - Total Tasks Executed: {total_tasks_executed}")
    print(f"   - Successful Placements: {success_count}")
    print(f"   - Failed: {failed_count}")
    print(f"   - Repairs Performed: {repair_count}")
    print(f"   - Temp Moves: {temp_count}")
    print(f"   - Efficiency: {len(original_sequence)/total_tasks_executed*100:.1f}%" 
          if total_tasks_executed > 0 else "N/A")
    
    # 检查是否有砖块还在临时位置
    if qp_scheduler.bricks_in_temp:
        print(f"   ⚠️ Bricks still in temp: {list(qp_scheduler.bricks_in_temp.keys())}")
    
    if success_count == len(original_sequence):
        print("🎉 Perfect! All bricks placed successfully!")
    elif success_count >= len(original_sequence) * 0.8:
        print("👍 Great! Most bricks placed successfully!")
    else:
        print("🤔 Parameters and strategy need further optimization.")
    
    print(f"{'='*60}")
    print("Keeping scene for inspection...")
    
    final_sec = env.cfg["timing"].get("final_wait_sec", 10.0)
    env.step(int(final_sec / env.dt))
    
    env.disconnect()


if __name__ == "__main__":
    main()