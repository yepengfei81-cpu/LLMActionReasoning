import pybullet as p
import numpy as np
import time
from env.pyb_env import BulletEnv
from modules.grasp_module import GraspModule
from control.gripper import GripperHelper
from modules.state_verifier import StateVerifier
from modules.motion_executor import MotionExecutor
from modules.qp_scheduler import QPTaskScheduler, TaskType, TaskItem
from modules.sam3_segment import SAM3BrickSegmenter, EyeInHandCamera, CameraDisplayManager


def main():
    # ============ 初始化环境 ============
    env = BulletEnv("configs/kuka_six_bricks.yaml", use_gui=True)
    rm = env.robot_model
    gripper = GripperHelper(rm)
    grasp = GraspModule(env)
    assist_cfg = env.cfg.get("assist_grasp", {})
    ground_z = env.get_ground_top()

    # ============ 获取砖块信息 ============
    original_sequence = env.get_brick_placement_sequence()
    brick_body_ids = env.brick_ids
    brick_height = env.cfg["brick"]["size_LWH"][2]
    
    print(f"[INIT] 砖块放置序列: {original_sequence}")
    print(f"[INIT] 砖块 Body IDs: {brick_body_ids}")
    print(f"[INIT] 砖块高度: {brick_height}")

    # ============ 初始化 SAM3 实时分割系统 ============
    sam3_segmenter = SAM3BrickSegmenter(
        camera_position=(0.0, 0.0, 2.0),
        camera_target=(0.0, 0.0, 0.2),
        width=640,
        height=480,
        fov=78.0,
        checkpoint_path="/home/ypf/sam3-main/checkpoint/sam3.pt",
        text_prompt="red building block",
        sam_resolution=1008,
        confidence_threshold=0.4,
        use_opengl=True,
        brick_body_ids=brick_body_ids,
        brick_height=brick_height,
    )    
    sam3_segmenter.start()

    # ============ 初始化手眼相机 ============
    eye_in_hand = EyeInHandCamera(
        robot_model=rm,
        width=640,
        height=480,
        fov=78.0,
        near=0.01,
        far=2.0,
        local_position=(0.0, -0.16, -0.1),
        local_orientation_rpy=(np.pi * 3/4, 0.0, 0.0),
        use_opengl=True,
    )
    eye_in_hand.start()

    # ============ 初始化统一显示管理器 ============
    display_manager = CameraDisplayManager(
        sam3_segmenter=sam3_segmenter,
        eye_in_hand=eye_in_hand,
        display_fps=15,
        combined_view=True
    )
    display_manager.start()

    print("\n[INIT] 执行初始 SAM3 分割，获取砖块位置...")
    sam3_segmenter.trigger_segment()
    time.sleep(1.5)  # 等待 SAM3 分割完成
    print("[INIT] 初始分割完成\n")

    # ============ 【新增】初始姿态检测和修复 ============
    print("[INIT] 检查砖块初始姿态...")
    # 创建临时 MotionExecutor 用于初始姿态检测
    init_vf = StateVerifier(env, rm, gripper, env.ground_id)
    init_motion = MotionExecutor(
        env, rm, gripper, init_vf,
        sam3_segmenter=sam3_segmenter,
        eye_in_hand_camera=eye_in_hand
    )
    
    init_pose_result = init_motion.check_and_correct_all_brick_poses(max_corrections=6)
    
    if init_pose_result["corrections_made"] > 0:
        print(f"[INIT] 初始姿态修复完成: {init_pose_result['corrections_made']} 次修复")
        for detail in init_pose_result["details"]:
            status = "✓" if detail['result'].get('success') else "✗"
            print(f"   {status} Brick {detail['brick_id']}: {detail['original_pose']}")
        
        # 修复后回到初始位置，并重新触发 SAM3
        init_motion.reset_between_tasks()
        reset_sec = env.cfg["timing"].get("reset_wait_sec", 1.5)
        env.step(int(reset_sec / env.dt))
    
    if not init_pose_result["all_flat"]:
        print("[INIT] ⚠️ 部分砖块仍未恢复平放状态，继续执行任务...")
    else:
        print("[INIT] ✓ 所有砖块姿态正常，开始任务执行\n")

    # ============ QP 调度器初始化 ============
    qp_scheduler = QPTaskScheduler(
        env, 
        threshold_low=0.055,
        threshold_critical=0.1
    )

    # ============ 任务状态跟踪 ============
    placed_bricks_info = []
    completed_bricks = set()
    success_count = 0
    failed_count = 0
    repair_count = 0
    temp_count = 0
    total_tasks_executed = 0
    task_queue = []
    is_holding_brick = False
    held_brick_idx = None

    # ============ 主循环 ============
    while True:
        if not display_manager.is_running():
            print("[MAIN] Display manager stopped, exiting...")
            break
            
        # ======== 【新增】步骤 0: 每轮循环开始前检测砖块姿态 ========
        # 利用前一轮 reset_between_tasks 或初始化时触发的 SAM3 缓存
        if sam3_segmenter is not None and not is_holding_brick:
            # 等待 SAM3 分割完成
            time.sleep(0.5)
            
            # 创建临时 MotionExecutor 用于姿态检测
            temp_vf = StateVerifier(env, rm, gripper, env.ground_id)
            temp_motion = MotionExecutor(
                env, rm, gripper, temp_vf,
                sam3_segmenter=sam3_segmenter,
                eye_in_hand_camera=eye_in_hand
            )
            
            print(f"\n[POSE_CHECK] 分析 SAM3 缓存中的砖块姿态...")
            pose_check_result = temp_motion.check_and_correct_all_brick_poses(max_corrections=3)
            
            if pose_check_result["corrections_made"] > 0:
                print(f"[POSE_CHECK] 完成 {pose_check_result['corrections_made']} 次姿态修复")
                for detail in pose_check_result["details"]:
                    status = "✓" if detail['result'].get('success') else "✗"
                    print(f"   {status} Brick {detail['brick_id']}: {detail['original_pose']}")
                
                # 修复后回到初始位置
                temp_motion.reset_between_tasks()
                reset_sec = env.cfg["timing"].get("reset_wait_sec", 1.5)
                env.step(int(reset_sec / env.dt))
                # 注意：reset_between_tasks 会触发 SAM3，下次循环会用新缓存
            
            if not pose_check_result["all_flat"]:
                print(f"[POSE_CHECK] ⚠️ 部分砖块仍未平放，继续执行任务...")
            else:
                print(f"[POSE_CHECK] ✓ 所有砖块姿态正常")
        
        # 【修复】检查是否真正完成
        all_placed = len(completed_bricks) >= len(original_sequence)
        no_temp_bricks = len(qp_scheduler.bricks_in_temp) == 0
        no_pending_tasks = len(task_queue) == 0
        
        if all_placed and no_temp_bricks and no_pending_tasks:
            print("[MAIN] All conditions met: all placed, no temp bricks, no pending tasks")
            break
        
        # ======== 步骤 1: 规划/更新任务队列 ========
        if len(task_queue) == 0:
            remaining = [idx for idx in original_sequence if idx not in completed_bricks]
            temp_bricks_to_restore = list(qp_scheduler.bricks_in_temp.keys())
            
            if remaining or temp_bricks_to_restore:
                current_brick_idx = remaining[0] if remaining else temp_bricks_to_restore[0]
                qp_scheduler.update_placed_bricks(placed_bricks_info)
                
                try:
                    task_queue = qp_scheduler.plan_task_sequence(
                        current_brick_idx=current_brick_idx,
                        remaining_sequence=remaining,
                        is_holding_brick=is_holding_brick
                    )
                except RuntimeError as e:
                    print(f"[ERROR] MILP solver failed: {e}")
                    break
            else:
                qp_scheduler.update_placed_bricks(placed_bricks_info)
                if qp_scheduler.should_replan():
                    bricks_to_repair = qp_scheduler.get_bricks_needing_repair()
                    remaining_bricks = [d["brick_idx"] for d in bricks_to_repair]
                    
                    try:
                        task_queue = qp_scheduler.plan_task_sequence(
                            current_brick_idx=remaining_bricks[0] if remaining_bricks else None,
                            remaining_sequence=remaining_bricks,
                            is_holding_brick=False
                        )
                    except RuntimeError as e:
                        print(f"[ERROR] Final repair MILP failed: {e}")
                        break
                else:
                    break
        
        if len(task_queue) == 0:
            if len(qp_scheduler.bricks_in_temp) > 0:
                print(f"[WARNING] Still have bricks in temp: {list(qp_scheduler.bricks_in_temp.keys())}")
                remaining = [idx for idx in original_sequence if idx not in completed_bricks]
                try:
                    task_queue = qp_scheduler.plan_task_sequence(
                        current_brick_idx=list(qp_scheduler.bricks_in_temp.keys())[0],
                        remaining_sequence=remaining,
                        is_holding_brick=is_holding_brick
                    )
                except RuntimeError as e:
                    print(f"[ERROR] Temp restore MILP failed: {e}")
                    break
            else:
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
        
        # ======== 步骤 2.5: 执行前检测 ========
        if task_type in [TaskType.NORMAL_PLACE] and not is_temp and not is_holding_brick:
            print(f"\n[PRE-CHECK] Checking placed bricks before executing task...")
            qp_scheduler.update_placed_bricks(placed_bricks_info)
            
            ancestors = qp_scheduler.get_all_ancestors(brick_idx)
            bricks_needing_repair = qp_scheduler.get_bricks_needing_repair()
            repair_set = {d["brick_idx"] for d in bricks_needing_repair}
            temp_set = set(qp_scheduler.bricks_in_temp.keys())
            problem_ancestors = ancestors & (repair_set | temp_set)
            
            if problem_ancestors:
                print(f"[PRE-CHECK] ⚠️ Dependencies {problem_ancestors} have problems!")
                remaining = [idx for idx in original_sequence if idx not in completed_bricks]
                
                try:
                    new_task_queue = qp_scheduler.plan_task_sequence(
                        current_brick_idx=brick_idx,
                        remaining_sequence=remaining,
                        is_holding_brick=is_holding_brick
                    )
                    task_queue = new_task_queue
                    
                    if task_queue:
                        current_task = task_queue.pop(0)
                        brick_idx = current_task.brick_idx
                        brick_id = current_task.brick_id
                        goal_pose = current_task.to_goal_pose()
                        task_type = current_task.task_type
                        level = current_task.level
                        is_temp = current_task.is_temp
                        level_name = env.get_level_name(brick_idx)
                        print(f"[PRE-CHECK] ✓ New first task: {task_type.value} brick={brick_idx}")
                    else:
                        continue
                except RuntimeError as e:
                    print(f"[PRE-CHECK] ❌ Re-planning failed: {e}")
            else:
                print(f"[PRE-CHECK] ✓ All dependencies OK")
        
        # ======== 步骤 3: 准备并执行任务 ========
        vf = StateVerifier(env, rm, gripper, brick_id)
        motion = MotionExecutor(
            env, rm, gripper, vf, 
            sam3_segmenter=sam3_segmenter,
            eye_in_hand_camera=eye_in_hand
        )
        brick_state = env.get_brick_state(brick_id=brick_id)
        wps, aux = grasp.plan(brick_state, [*goal_pose], ground_z, brick_id=brick_id)
        
        if task_type == TaskType.TEMP_PLACE:
            support_ids = [env.ground_id]
        else:
            support_ids = env.get_related_support_ids(brick_idx)
        
        result = motion.execute_fsm(wps, aux, assist_cfg, brick_id, env.ground_id, support_ids=support_ids)
        
        if isinstance(result, bool):
            result = {"success": result, "holding_brick": False, "failed_phase": None, "brick_released": result}
        
        ok = result["success"]
        is_holding_brick = result["holding_brick"]
        held_brick_idx = brick_idx if is_holding_brick else None
        
        # ======== 步骤 4: 处理结果 ========
        if ok:
            is_holding_brick = False
            held_brick_idx = None
            
            if task_type == TaskType.TEMP_PLACE:
                temp_count += 1
                print(f"📦 [TEMP SUCCESS] Brick idx={brick_idx} moved to temp position!")
                qp_scheduler.mark_brick_in_temp(brick_idx, goal_pose[:3])
                
                found = False
                for info in placed_bricks_info:
                    if info["brick_idx"] == brick_idx:
                        info["expected_pos"] = goal_pose[:3]
                        info["expected_orn"] = goal_pose[3:]
                        info["is_temp"] = True
                        found = True
                        break
                if not found:
                    placed_bricks_info.append({
                        "brick_id": brick_id, "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3], "expected_orn": goal_pose[3:],
                        "level": level, "is_temp": True
                    })
                    
            elif task_type == TaskType.REPAIR_PLACE:
                repair_count += 1
                print(f"✅ [REPAIR SUCCESS] Brick idx={brick_idx} repaired!")
                qp_scheduler.unmark_brick_from_temp(brick_idx)
                
                found = False
                for info in placed_bricks_info:
                    if info["brick_idx"] == brick_idx:
                        info["expected_pos"] = goal_pose[:3]
                        info["expected_orn"] = goal_pose[3:]
                        info["is_temp"] = False
                        found = True
                        break
                if not found:
                    placed_bricks_info.append({
                        "brick_id": brick_id, "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3], "expected_orn": goal_pose[3:],
                        "level": level, "is_temp": False
                    })
                
                if brick_idx in original_sequence:
                    completed_bricks.add(brick_idx)
                    
            else:  # NORMAL_PLACE
                success_count += 1
                completed_bricks.add(brick_idx)
                print(f"✅ [SUCCESS] {level_name} (brick {brick_idx}) Placement Successful!")
                
                found = any(info["brick_idx"] == brick_idx for info in placed_bricks_info)
                if not found:
                    placed_bricks_info.append({
                        "brick_id": brick_id, "brick_idx": brick_idx,
                        "expected_pos": goal_pose[:3], "expected_orn": goal_pose[3:],
                        "level": level, "is_temp": False
                    })
        else:
            failed_count += 1
            print(f"❌ [FAILED] {level_name} (brick {brick_idx}) Failed at phase: {result.get('failed_phase', 'unknown')}")
            
            if result["holding_brick"]:
                print(f"⚠️ [WARNING] Still holding brick {brick_idx}!")
                is_holding_brick = True
                held_brick_idx = brick_idx
            else:
                is_holding_brick = False
                held_brick_idx = None
        
        print(f"[Progress] Completed: {len(completed_bricks)}/{len(original_sequence)}, "
              f"Failed: {failed_count}, Repairs: {repair_count}, Temp: {temp_count}")
        
        # 等待稳定
        settle_sec = env.cfg["timing"].get("brick_settle_sec", 2.0)
        env.step(int(settle_sec / env.dt))
        
        # ======== 步骤 5: 重新规划检查 ========
        qp_scheduler.update_placed_bricks(placed_bricks_info)
        
        if qp_scheduler.should_replan() and len(task_queue) > 0:
            print(f"\n[QP] ⚠️ Deviation detected! Re-planning...")
            next_brick_idx = task_queue[0].brick_idx if task_queue else None
            remaining = [idx for idx in original_sequence if idx not in completed_bricks]
            
            try:
                task_queue = qp_scheduler.plan_task_sequence(
                    current_brick_idx=next_brick_idx,
                    remaining_sequence=remaining,
                    is_holding_brick=is_holding_brick
                )
            except RuntimeError as e:
                print(f"[ERROR] Re-planning MILP failed: {e}")
        
        # ======== 步骤 6: 重置机械臂（触发 SAM3） ========
        if len(task_queue) > 0 or len(completed_bricks) < len(original_sequence):
            print("Preparing for next task, resetting...")
            motion.reset_between_tasks()  # ← 这里会触发 SAM3
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
    
    if len(completed_bricks) == len(original_sequence):
        print("🎉 Perfect! All bricks placed successfully!")
    
    print(f"{'='*60}")
    
    final_sec = env.cfg["timing"].get("final_wait_sec", 10.0)
    env.step(int(final_sec / env.dt))

    display_manager.close()
    sam3_segmenter.close()
    eye_in_hand.close()
    env.disconnect()


if __name__ == "__main__":
    main()