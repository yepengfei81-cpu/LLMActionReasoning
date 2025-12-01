import numpy as np
from math import cos, sin, pi

def rpy_to_mat(roll, pitch, yaw):
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def mat_to_rpy(R):
    pitch = -np.arcsin(R[2, 0])
    roll  =  np.arctan2(R[2, 1], R[2, 2])
    yaw   =  np.arctan2(R[1, 0], R[0, 0])
    return [float(roll), float(pitch), float(yaw)]

def normalize_angle(angle):
    while angle > pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle

def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    return normalize_angle(diff)

def align_topdown_to_brick(brick_yaw):
    Rbr = rpy_to_mat(0, 0, brick_yaw)
    yb = Rbr[:, 1]
    ez = np.array([0.0, 0.0, -1.0])
    ex = yb - ez * float(yb @ ez)
    n  = np.linalg.norm(ex)
    if n < 1e-9:
        ex = np.array([1.0, 0.0, 0.0])
    else:
        ex /= n
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
    R = np.column_stack([ex, ey, ez])
    return mat_to_rpy(R)
