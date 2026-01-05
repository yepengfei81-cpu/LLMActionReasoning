import pybullet as p
import numpy as np
import time
from env.pyb_env import BulletEnv
from modules.grasp_module import GraspModule
from control.gripper import GripperHelper
from modules.state_verifier import StateVerifier
from modules.motion_executor import MotionExecutor
from modules.qp_scheduler import QPTaskScheduler, TaskType
from modules.sam3_segment import SAM3BrickSegmenter, EyeInHandCamera, CameraDisplayManager


def main():
    # ============ 初始化环境 ============
    env = BulletEnv("configs/kuka_six_bricks.yaml", use_gui=True)
    rm = env.robot_model
    gripper = GripperHelper(rm)
    grasp = GraspModule(env)
    assist_cfg = env.cfg.get("assist_grasp", {})
    ground_z = env.get_ground_top()

    brick_body_ids = env.brick_ids
    brick_height = env.cfg["brick"]["size_LWH"][2]
    
    print(f"[INIT] 砖块数量: {len(brick_body_ids)}")

    # ============ 初始化视觉系统 ============
    sam3_segmenter = SAM3BrickSegmenter(
        camera_position=(0.0, 0.0, 2.0),
        camera_target=(0.0, 0.0, 0.2),
        width=640, height=480, fov=78.0,
        checkpoint_path="/home/ypf/sam3-main/checkpoint/sam3.pt",
        text_prompt="red building block",
        sam_resolution=1008, confidence_threshold=0.4,
        use_opengl=True,
        brick_body_ids=brick_body_ids,
        brick_height=brick_height,
    )    
    sam3_segmenter.start()

    eye_in_hand = EyeInHandCamera(
        robot_model=rm,
        width=640, height=480, fov=78.0,
        near=0.01, far=2.0,
        local_position=(0.0, -0.16, -0.1),
        local_orientation_rpy=(np.pi * 3/4, 0.0, 0.0),
        use_opengl=True,
    )
    eye_in_hand.start()

    display_manager = CameraDisplayManager(
        sam3_segmenter=sam3_segmenter,
        eye_in_hand=eye_in_hand,
        display_fps=15, combined_view=True
    )
    display_manager.start()

    sam3_segmenter.trigger_segment()
    time.sleep(1.5)

    # ============ 初始姿态检测 ============
    init_vf = StateVerifier(env, rm, gripper, env.ground_id)
    init_motion = MotionExecutor(env, rm, gripper, init_vf,
                                  sam3_segmenter=sam3_segmenter,
                                  eye_in_hand_camera=eye_in_hand)
    
    init_result = init_motion.check_and_correct_all_brick_poses(max_corrections=6)
    if init_result["corrections_made"] > 0:
        init_motion.reset_between_tasks()
        env.step(int(env.cfg["timing"].get("reset_wait_sec", 1.5) / env.dt))

    # ============ QP 调度器 ============
    scheduler = QPTaskScheduler(env, fill_threshold=0.12)

    # ============ 统计 ============
    success_count = 0
    failed_count = 0
    total_tasks = 0

    # ============ 主循环 ============
    while display_manager.is_running():
        # 姿态检测
        time.sleep(0.3)
        temp_vf = StateVerifier(env, rm, gripper, env.ground_id)
        temp_motion = MotionExecutor(env, rm, gripper, temp_vf,
                                      sam3_segmenter=sam3_segmenter,
                                      eye_in_hand_camera=eye_in_hand)
        pose_result = temp_motion.check_and_correct_all_brick_poses(max_corrections=3)
        if pose_result["corrections_made"] > 0:
            temp_motion.reset_between_tasks()
            env.step(int(env.cfg["timing"].get("reset_wait_sec", 1.5) / env.dt))
        
        # 获取下一个任务
        task = scheduler.get_next_task()
        
        if task is None:
            if scheduler.all_slots_filled():
                print("[MAIN] ✅ All slots filled!")
                break
            print("[MAIN] ⚠️ No task, retrying...")
            continue
        
        total_tasks += 1
        
        # ======== 【关键修复】基于位置找到最近的砖块 ========
        # 不依赖 task.pybullet_id，而是根据 grasp_position 找最近的砖块
        grasp_target = np.array(task.grasp_position)
        
        best_brick_id = None
        best_dist = float('inf')
        best_pos = None
        best_orn = None
        
        for bid in env.brick_ids:
            try:
                pos, orn = p.getBasePositionAndOrientation(bid)
                pos = np.array(pos)
                dist = np.linalg.norm(pos[:2] - grasp_target[:2])
                
                if dist < best_dist:
                    best_dist = dist
                    best_brick_id = bid
                    best_pos = pos
                    best_orn = orn
            except:
                continue
        
        if best_brick_id is None or best_dist > 0.1:  # 10cm 容差
            print(f"[MAIN] ⚠️ No brick found near grasp position {task.grasp_position}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[TASK #{total_tasks}]")
        print(f"   Target grasp: ({task.grasp_position[0]:.3f}, {task.grasp_position[1]:.3f}, {task.grasp_position[2]:.3f})")
        print(f"   Actual brick: ({best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}) [dist={best_dist*1000:.1f}mm]")
        print(f"   Target: Slot {task.slot_idx} (Level {task.level})")
        print(f"   Cost: {task.estimated_cost:.2f}s")
        
        # 【修复】使用实际找到的砖块
        actual_brick_id = best_brick_id
        actual_grasp_pos = tuple(best_pos)
        
        # 执行任务
        vf = StateVerifier(env, rm, gripper, actual_brick_id)
        motion = MotionExecutor(env, rm, gripper, vf,
                                 sam3_segmenter=sam3_segmenter,
                                 eye_in_hand_camera=eye_in_hand)
        
        # 用实际位置构造 brick_state
        brick_state = {
            "pos": actual_grasp_pos,
            "orn": best_orn
        }
        
        goal_pose = task.to_goal_pose()
        wps, aux = grasp.plan(brick_state, [*goal_pose], ground_z, brick_id=actual_brick_id)
        
        # 支撑面
        support_ids = [env.ground_id]
        if task.level > 0:
            for slot in scheduler.slots:
                if slot.level == task.level - 1 and slot.status.value == "filled":
                    for bid in env.brick_ids:
                        try:
                            pos, _ = p.getBasePositionAndOrientation(bid)
                            if np.linalg.norm(np.array(pos[:2]) - slot.position[:2]) < 0.05:
                                support_ids.append(bid)
                                break
                        except:
                            pass
        
        result = motion.execute_fsm(wps, aux, assist_cfg, actual_brick_id, env.ground_id, support_ids=support_ids)
        
        ok = result.get("success", False) if isinstance(result, dict) else result
        
        if ok:
            success_count += 1
            print(f"✅ [SUCCESS]")
        else:
            failed_count += 1
            print(f"❌ [FAILED]")
        
        progress = scheduler.get_progress()
        print(f"[Progress] {progress['filled']}/{progress['total']} slots, Success: {success_count}, Failed: {failed_count}")
        
        # 等待稳定
        env.step(int(env.cfg["timing"].get("brick_settle_sec", 2.0) / env.dt))
        
        # 重置
        if not scheduler.all_slots_filled():
            motion.reset_between_tasks()
            env.step(int(env.cfg["timing"].get("reset_wait_sec", 1.5) / env.dt))

    # ============ 结束 ============
    print(f"\n{'='*60}")
    print(f"🎯 Task Complete!")
    print(f"   Total: {total_tasks}, Success: {success_count}, Failed: {failed_count}")
    scheduler.print_status()
    
    env.step(int(env.cfg["timing"].get("final_wait_sec", 10.0) / env.dt))

    display_manager.close()
    sam3_segmenter.close()
    eye_in_hand.close()
    env.disconnect()


if __name__ == "__main__":
    main()