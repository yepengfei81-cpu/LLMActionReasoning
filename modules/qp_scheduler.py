"""
QP 任务调度器 (槽位填充模式 - 简化版)

核心思想：
- 砖块没有ID，只有位置
- 距离槽位 < fill_threshold = 已填充（不可抓取）
- 距离槽位 >= fill_threshold = 可抓取
- 用 MILP 优化砖块位置到槽位的分配，最小化总成本
"""

import numpy as np
import pybullet as p
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    raise ImportError("cvxpy is required. Install with: pip install cvxpy")


class TaskType(Enum):
    NORMAL_PLACE = "normal_place"


class SlotStatus(Enum):
    EMPTY = "empty"
    FILLED = "filled"


@dataclass
class Slot:
    """槽位（目标位置）"""
    slot_idx: int
    level: int
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [r, p, y]
    status: SlotStatus = SlotStatus.EMPTY


@dataclass
class GraspableObject:
    """可抓取物体（只有位置，没有ID）"""
    position: np.ndarray  # [x, y, z]
    pybullet_id: int  # 仅用于执行抓取，不用于规划逻辑


@dataclass
class TaskItem:
    """任务项"""
    task_type: TaskType
    grasp_position: Tuple[float, float, float]  # 抓取位置
    target_position: Tuple[float, float, float]  # 目标位置
    target_orientation: Tuple[float, float, float]  # 目标姿态
    level: int
    slot_idx: int
    pybullet_id: int  # 仅用于执行
    estimated_cost: float = 0.0
    
    def to_goal_pose(self) -> Tuple[float, float, float, float, float, float]:
        return (*self.target_position, *self.target_orientation)


# 成本参数
VERTICAL_COST = 5.0  # 固定垂直运动成本（秒）
ALPHA = 2.0  # 距离-时间转换系数（秒/米）


class QPTaskScheduler:
    """基于位置的槽位填充调度器"""
    
    def __init__(self, env, fill_threshold: float = 0.05):
        """
        Args:
            env: BulletEnv 环境
            fill_threshold: 槽位填充阈值（米）
                - 距离 < fill_threshold: 视为已填充，不可抓取
                - 距离 >= fill_threshold: 视为可抓取
        """
        self.env = env
        self.fill_threshold = fill_threshold
        
        # 砖块尺寸
        self.brick_L, self.brick_W, self.brick_H = env.cfg["brick"]["size_LWH"]
        self.ground_z = env.get_ground_top() if hasattr(env, 'get_ground_top') else 0.0
        
        # Home 位置（XY）
        home_cfg = env.cfg.get("home_pose_xyz", [0.55, 0.0, 0.55])
        self.home_xy = np.array(home_cfg[:2])
        
        # 初始化槽位
        self._init_slots()
        self._print_init_info()
    
    def _init_slots(self):
        """从 layout 配置初始化槽位"""
        self.slots: List[Slot] = []
        
        if not hasattr(self.env, 'layout_targets'):
            self.env._parse_layout()
        
        yaw = self.env.cfg["goal"]["yaw"]
        
        for idx, target in enumerate(self.env.layout_targets):
            level = target["level"]
            xy = target["xy"]
            z = self.ground_z + self.brick_H / 2 + level * self.brick_H
            
            self.slots.append(Slot(
                slot_idx=idx,
                level=level,
                position=np.array([xy[0], xy[1], z]),
                orientation=np.array([0.0, 0.0, yaw]),
                status=SlotStatus.EMPTY
            ))
        
        self.slots.sort(key=lambda s: (s.level, s.slot_idx))
        self.max_level = max(s.level for s in self.slots) if self.slots else 0
    
    def _print_init_info(self):
        print(f"\n[QP] ═══════════════════════════════════════════════════")
        print(f"[QP] 简化版槽位填充调度器")
        print(f"[QP] 填充阈值: {self.fill_threshold*100:.1f}cm")
        print(f"[QP] 槽位数量: {len(self.slots)}")
        for level in range(self.max_level + 1):
            count = sum(1 for s in self.slots if s.level == level)
            print(f"     Level {level}: {count} 个槽位")
        print(f"[QP] ═══════════════════════════════════════════════════\n")
    
    # ================== 核心：基于位置的状态检测 ==================
    
    def _get_all_brick_positions(self) -> List[Tuple[np.ndarray, int]]:
        """
        获取所有砖块的当前位置
        
        Returns:
            [(position, pybullet_id), ...]
        """
        positions = []
        for brick_id in self.env.brick_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(brick_id)
                positions.append((np.array(pos), brick_id))
            except:
                pass
        return positions
    
    def _update_world_state(self) -> Tuple[List[GraspableObject], List[Slot]]:
        """
        更新世界状态：检测哪些砖块在槽位中，哪些可抓取
        
        核心逻辑：
        - 遍历所有砖块位置
        - 如果砖块距离某个槽位 < fill_threshold → 该槽位已填充
        - 否则 → 该砖块可抓取
        
        Returns:
            (可抓取物体列表, 更新后的槽位列表)
        """
        # 重置槽位状态
        for slot in self.slots:
            slot.status = SlotStatus.EMPTY
        
        all_bricks = self._get_all_brick_positions()
        graspable = []
        
        for pos, pybullet_id in all_bricks:
            is_in_slot = False
            
            # 检查是否在某个槽位中
            for slot in self.slots:
                if slot.status == SlotStatus.FILLED:
                    continue
                
                xy_dist = np.linalg.norm(pos[:2] - slot.position[:2])
                z_diff = abs(pos[2] - slot.position[2])
                
                # 判断是否填充该槽位
                if xy_dist < self.fill_threshold and z_diff < self.brick_H * 0.8:
                    slot.status = SlotStatus.FILLED
                    is_in_slot = True
                    break
            
            # 不在任何槽位中 → 可抓取
            if not is_in_slot:
                # 额外检查：Z 高度合理（在地面附近，排除飞出去的）
                if self.ground_z - 0.05 < pos[2] < self.ground_z + self.brick_H * 3:
                    graspable.append(GraspableObject(
                        position=pos,
                        pybullet_id=pybullet_id
                    ))
        
        return graspable, self.slots
    
    # ================== 成本计算 ==================
    
    def _compute_cost(self, grasp_pos: np.ndarray, slot_pos: np.ndarray) -> float:
        """
        计算从 grasp_pos 抓取并放到 slot_pos 的成本
        
        成本 = α * (d_home→brick + d_brick→slot + d_slot→home) + C_vertical
        """
        d1 = np.linalg.norm(self.home_xy - grasp_pos[:2])
        d2 = np.linalg.norm(grasp_pos[:2] - slot_pos[:2])
        d3 = np.linalg.norm(slot_pos[:2] - self.home_xy)
        
        return ALPHA * (d1 + d2 + d3) + VERTICAL_COST
    
    # ================== MILP 求解 ==================
    
    def _solve_assignment(self, 
                          graspable: List[GraspableObject],
                          empty_slots: List[Slot]) -> List[Tuple[GraspableObject, Slot]]:
        """
        MILP 求解最优分配
        
        目标: min Σ cost(i,j) * x_ij
        约束:
            - 每个槽位最多分配一个物体
            - 每个物体最多分配到一个槽位
        """
        n = len(graspable)
        m = len(empty_slots)
        
        if n == 0 or m == 0:
            return []
        
        print(f"[QP-MILP] Solving: {n} graspable → {m} empty slots")
        
        # 构建成本矩阵
        cost = np.zeros((n, m))
        for i, obj in enumerate(graspable):
            for j, slot in enumerate(empty_slots):
                cost[i, j] = self._compute_cost(obj.position, slot.position)
        
        # MILP
        x = cp.Variable((n, m), boolean=True)
        objective = cp.Minimize(cp.sum(cp.multiply(cost, x)))
        
        constraints = [
            cp.sum(x, axis=0) <= 1,  # 每个槽位最多一个
            cp.sum(x, axis=1) <= 1,  # 每个物体最多一个槽位
            cp.sum(x) == min(n, m)   # 尽可能多地分配
        ]
        
        prob = cp.Problem(objective, constraints)
        
        # 尝试多个求解器
        for solver in [cp.GLPK_MI, cp.CBC, cp.SCIP, cp.ECOS_BB]:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    break
            except:
                continue
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"[QP-MILP] Warning: solver status = {prob.status}, using greedy")
            return self._greedy_assignment(graspable, empty_slots, cost)
        
        # 解析结果
        assignments = []
        x_val = x.value
        for i in range(n):
            for j in range(m):
                if x_val[i, j] > 0.5:
                    assignments.append((graspable[i], empty_slots[j]))
                    print(f"[QP-MILP] Assign pos ({graspable[i].position[0]:.2f}, "
                          f"{graspable[i].position[1]:.2f}) → Slot {empty_slots[j].slot_idx}")
        
        return assignments
    
    def _greedy_assignment(self, 
                           graspable: List[GraspableObject],
                           empty_slots: List[Slot],
                           cost: np.ndarray) -> List[Tuple[GraspableObject, Slot]]:
        """贪心分配（备用）"""
        pairs = []
        for i, obj in enumerate(graspable):
            for j, slot in enumerate(empty_slots):
                pairs.append((cost[i, j], i, j))
        pairs.sort()
        
        used_i, used_j = set(), set()
        assignments = []
        
        for c, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            assignments.append((graspable[i], empty_slots[j]))
            used_i.add(i)
            used_j.add(j)
        
        return assignments
    
    # ================== 主接口 ==================
    
    def get_next_task(self) -> Optional[TaskItem]:
        """
        获取下一个任务
        
        流程：
        1. 检测世界状态（哪些可抓取，哪些槽位空）
        2. 找当前层级的空槽位
        3. MILP 求解最优分配
        4. 返回成本最低的任务
        """
        # 更新世界状态
        graspable, slots = self._update_world_state()
        
        # 打印状态
        filled = sum(1 for s in slots if s.status == SlotStatus.FILLED)
        print(f"[QP] State: {filled}/{len(slots)} slots filled, {len(graspable)} graspable")
        
        # 检查完成
        if all(s.status == SlotStatus.FILLED for s in slots):
            print(f"[QP] ✅ All slots filled!")
            return None
        
        # 找当前层级（最低未完成层）
        current_level = 0
        for level in range(self.max_level + 1):
            level_slots = [s for s in slots if s.level == level]
            if not all(s.status == SlotStatus.FILLED for s in level_slots):
                current_level = level
                break
        
        # 获取当前层的空槽位
        empty_slots = [s for s in slots 
                       if s.level == current_level and s.status == SlotStatus.EMPTY]
        
        if not empty_slots:
            print(f"[QP] No empty slots in Level {current_level}")
            return None
        
        if not graspable:
            print(f"[QP] ⚠️ No graspable objects!")
            return None
        
        print(f"[QP] Level {current_level}: {len(empty_slots)} empty, {len(graspable)} graspable")
        
        # MILP 求解
        assignments = self._solve_assignment(graspable, empty_slots)
        
        if not assignments:
            print(f"[QP] ⚠️ No valid assignments!")
            return None
        
        # 选择成本最低的
        best = min(assignments, key=lambda x: self._compute_cost(x[0].position, x[1].position))
        obj, slot = best
        cost = self._compute_cost(obj.position, slot.position)
        
        print(f"[QP] Next task: grasp at ({obj.position[0]:.3f}, {obj.position[1]:.3f}, "
              f"{obj.position[2]:.3f}) → Slot {slot.slot_idx} (cost: {cost:.2f}s)")
        
        return TaskItem(
            task_type=TaskType.NORMAL_PLACE,
            grasp_position=tuple(obj.position),
            target_position=tuple(slot.position),
            target_orientation=tuple(slot.orientation),
            level=slot.level,
            slot_idx=slot.slot_idx,
            pybullet_id=obj.pybullet_id,
            estimated_cost=cost
        )
    
    def all_slots_filled(self) -> bool:
        """检查是否全部完成"""
        self._update_world_state()
        return all(s.status == SlotStatus.FILLED for s in self.slots)
    
    def get_progress(self) -> Dict:
        """获取进度"""
        self._update_world_state()
        filled = sum(1 for s in self.slots if s.status == SlotStatus.FILLED)
        return {
            "filled": filled,
            "total": len(self.slots),
            "complete": filled == len(self.slots)
        }
    
    def print_status(self):
        """打印状态"""
        graspable, _ = self._update_world_state()
        
        print(f"\n[QP] ═══════════════════════════════════════════════════")
        for level in range(self.max_level + 1):
            level_slots = [s for s in self.slots if s.level == level]
            status_str = " ".join([
                f"S{s.slot_idx}:{'✓' if s.status == SlotStatus.FILLED else '○'}"
                for s in level_slots
            ])
            print(f"     Level {level}: {status_str}")
        print(f"[QP] Graspable: {len(graspable)}")
        print(f"[QP] ═══════════════════════════════════════════════════\n")


# 兼容旧接口
class ActionType(Enum):
    PRE_GRASP = "pre_grasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    PRE_PLACE = "pre_place"
    DESCEND_PLACE = "descend_place"
    RELEASE = "release"