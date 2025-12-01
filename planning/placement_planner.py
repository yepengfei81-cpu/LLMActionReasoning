def place_waypoints(goal_pose_xyzrpy, brick_H, approach_clearance=0.08, place_gap=0.001):
    gx, gy, gz, gr, gp, gyaw = goal_pose_xyzrpy
    z_top = gz
    z_pre = z_top + approach_clearance
    z_put = z_top + place_gap
    rpy = [0.0, 0.0, gyaw]

    wps = [
      dict(name="pre_place", pose=dict(xyz=[gx,gy,z_pre], rpy=rpy), open_gripper=None, wait_sec=0.01),
      dict(name="descend",   pose=dict(xyz=[gx,gy,z_put], rpy=rpy), open_gripper=None, wait_sec=0.01),
      dict(name="open",      pose=dict(xyz=[gx,gy,z_put], rpy=rpy), open_gripper="OPEN_FOR_RELEASE", wait_sec=0.05),
      dict(name="retreat",   pose=dict(xyz=[gx,gy,z_pre], rpy=rpy), open_gripper=None, wait_sec=0.01),
    ]
    return wps
