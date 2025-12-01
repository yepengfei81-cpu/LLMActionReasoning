import pybullet as p
import pybullet_data
import os

def set_search_path():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

def load_table_at(x, y, z=0.0):
    table_urdf = os.path.join("table", "table.urdf")
    table_id = p.loadURDF(table_urdf, [x, y, z], useFixedBase=True)
    return table_id

def aabb_top_z(body_id, link_id=-1):
    aabb_min, aabb_max = p.getAABB(body_id, link_id)
    return aabb_max[2]

def get_all_joint_info(body_id):
    infos = []
    for j in range(p.getNumJoints(body_id)):
        ji = p.getJointInfo(body_id, j)
        name = ji[1].decode("utf-8")
        jtype = ji[2]
        parent = ji[16]
        link_name = ji[12].decode("utf-8")
        lo, hi = ji[8], ji[9]
        infos.append(dict(
            index=j, name=name, type=jtype, parent=parent,
            link_name=link_name, lower=lo, upper=hi
        ))
    return infos
