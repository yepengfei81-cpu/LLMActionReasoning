import pybullet as p
import numpy as np

def euler_to_quat(rpy):
    return p.getQuaternionFromEuler(rpy)

def _ik_bullet(robot_model, target_xyz, target_rpy, max_iters):
    """PyBullet IK 求解"""
    q = euler_to_quat(target_rpy)
    
    if hasattr(p, "calculateInverseKinematics2"):
        try:
            sol = p.calculateInverseKinematics2(
                bodyUniqueId=robot_model.id,
                endEffectorLinkIndex=robot_model.ee_link,
                targetPosition=target_xyz,
                targetOrientation=q,
                lowerLimits=robot_model.ik_chain_ll,
                upperLimits=robot_model.ik_chain_ul,
                jointRanges=robot_model.ik_chain_jr,
                restPoses=robot_model.ik_chain_rp,
                jointDamping=robot_model.ik_chain_jd,
                maxNumIterations=int(max_iters),
                residualThreshold=1e-4
            )
            return sol, robot_model.ik_chain_indices
        except Exception:
            pass

    sol = p.calculateInverseKinematics(
        bodyUniqueId=robot_model.id,
        endEffectorLinkIndex=robot_model.ee_link,
        targetPosition=target_xyz,
        targetOrientation=q,
        lowerLimits=robot_model.ik_full_ll,
        upperLimits=robot_model.ik_full_ul,
        jointRanges=robot_model.ik_full_jr,
        restPoses=robot_model.ik_full_rp,
        jointDamping=robot_model.ik_full_jd,
        maxNumIterations=int(max_iters),
        residualThreshold=1e-4
    )
    return sol, robot_model.ik_full_indices

def _extract_joint_angles(robot_model, sol, used_indices):
    """从 IK 解中提取关节角度"""
    idx2angle = {used_indices[k]: sol[k] for k in range(min(len(sol), len(used_indices)))}
    return [idx2angle.get(j, p.getJointState(robot_model.id, j)[0]) for j in robot_model.arm_joint_indices]

def _check_ik_error(robot_model, q_goal, target_xyz):
    """检查 IK 解的误差（不改变实际关节状态）"""
    q_cur = [p.getJointState(robot_model.id, j)[0] for j in robot_model.arm_joint_indices]
    
    # 临时应用解
    for i, j in enumerate(robot_model.arm_joint_indices):
        p.resetJointState(robot_model.id, j, q_goal[i])
    
    ee_pos = p.getLinkState(robot_model.id, robot_model.ee_link)[0]
    error = np.linalg.norm(np.array(ee_pos) - np.array(target_xyz))
    
    # 恢复
    for i, j in enumerate(robot_model.arm_joint_indices):
        p.resetJointState(robot_model.id, j, q_cur[i])
    
    return error

def ik_solve(robot_model, target_xyz, target_rpy, max_iters=180, max_error=0.05):
    """
    IK 求解，失败时自动从零位重试
    """
    # 第一次尝试
    sol, used_indices = _ik_bullet(robot_model, target_xyz, target_rpy, max_iters)
    q_goal = _extract_joint_angles(robot_model, sol, used_indices)
    error = _check_ik_error(robot_model, q_goal, target_xyz)
    
    if error <= max_error:
        return q_goal
    
    # 从零位重试
    q_cur = [p.getJointState(robot_model.id, j)[0] for j in robot_model.arm_joint_indices]
    for j in robot_model.arm_joint_indices:
        p.resetJointState(robot_model.id, j, 0.0)
    
    sol_retry, used_indices_retry = _ik_bullet(robot_model, target_xyz, target_rpy, max_iters)
    q_retry = _extract_joint_angles(robot_model, sol_retry, used_indices_retry)
    error_retry = _check_ik_error(robot_model, q_retry, target_xyz)
    
    # 恢复原始状态
    for i, j in enumerate(robot_model.arm_joint_indices):
        p.resetJointState(robot_model.id, j, q_cur[i])
    
    return q_retry if error_retry < error else q_goal

def move_ee_cartesian(env, robot_model, pose, duration=1.3, steps=None):
    """移动 EE 到目标位姿"""
    if steps is None:
        steps = max(1, int(duration / env.dt))
    
    q_goal = ik_solve(robot_model, pose["xyz"], pose["rpy"], 
                      max_iters=robot_model.cfg["control"]["ik_max_iters"])
    q_cur = [p.getJointState(robot_model.id, j)[0] for j in robot_model.arm_joint_indices]
    
    for q in np.linspace(q_cur, q_goal, steps):
        robot_model.set_arm_positions(q.tolist())
        env.step(1)

def move_tcp_cartesian(env, robot_model, tcp_pose, duration=1.3, steps=None):
    """移动 TCP 到目标位姿"""
    ee_pose = robot_model.ee_pose_from_tcp(tcp_pose["xyz"], tcp_pose["rpy"])
    move_ee_cartesian(env, robot_model, ee_pose, duration=duration, steps=steps)