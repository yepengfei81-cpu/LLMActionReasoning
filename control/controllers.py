import pybullet as p
import numpy as np

def euler_to_quat(rpy):
    return p.getQuaternionFromEuler(rpy)

def _ik_bullet(robot_model, target_xyz, target_rpy, max_iters):
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

def ik_solve(robot_model, target_xyz, target_rpy, max_iters=180):
    sol, used_indices = _ik_bullet(robot_model, target_xyz, target_rpy, max_iters)
    idx2angle = {}
    n = min(len(sol), len(used_indices))
    for k in range(n):
        idx2angle[used_indices[k]] = sol[k]

    q_cmd = []
    for j in robot_model.arm_joint_indices:
        q_cmd.append(idx2angle.get(j, p.getJointState(robot_model.id, j)[0]))
    return q_cmd

def move_ee_cartesian(env, robot_model, pose, duration=1.3, steps=None):
    if steps is None:
        steps = max(1, int(duration / env.dt))
    xyz = pose["xyz"]; rpy = pose["rpy"]
    
    q_goal = ik_solve(robot_model, xyz, rpy, max_iters=robot_model.cfg["control"]["ik_max_iters"])
    q_cur = [p.getJointState(robot_model.id, j)[0] for j in robot_model.arm_joint_indices]
    
    qs = np.linspace(q_cur, q_goal, steps)
    for q in qs:
        robot_model.set_arm_positions(q.tolist())
        env.step(1)

def move_tcp_cartesian(env, robot_model, tcp_pose, duration=1.3, steps=None):
    """
    Convert TCP pose to end-effector pose and execute movement.
    Use robot_model's built-in conversion function to ensure correct conversion.
    """
    ee_pose = robot_model.ee_pose_from_tcp(tcp_pose["xyz"], tcp_pose["rpy"])
    move_ee_cartesian(env, robot_model, ee_pose, duration=duration, steps=steps)

