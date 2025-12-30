"""
QP 任务调度器 (带依赖约束)
功能: 检测已放置砖块的偏移，动态调整任务序列
关键: 使用 MILP 优化求解最优任务序列
"""

import numpy as np
import pybullet as p
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    raise ImportError("cvxpy is required for QP optimization. Install with: pip install cvxpy")


class TaskType(Enum):
    """任务类型枚举"""
    NORMAL_PLACE = "normal_place"      # 正常放置新砖块
    REPAIR_PLACE = "repair_place"      # 修复已放置的砖块
    TEMP_PLACE = "temp_place"          # 临时放置（移开碍事的砖块）


class ActionType(Enum):
    """原子动作类型"""
    PRE_GRASP = "pre_grasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    PRE_PLACE = "pre_place"
    DESCEND_PLACE = "descend_place"
    RELEASE = "release"


# 每个动作的估计时间成本（秒）
ACTION_COSTS = {
    ActionType.PRE_GRASP: 1.5,
    ActionType.DESCEND: 1.0,
    ActionType.CLOSE: 0.5,
    ActionType.LIFT: 1.0,
    ActionType.PRE_PLACE: 1.5,
    ActionType.DESCEND_PLACE: 1.0,
    ActionType.RELEASE: 0.5,
}

# 成本常量
PLACE_ONLY_COST = (ACTION_COSTS[ActionType.PRE_PLACE] + 
                   ACTION_COSTS[ActionType.DESCEND_PLACE] + 
                   ACTION_COSTS[ActionType.RELEASE])  # ~3秒

FULL_PICK_PLACE_COST = sum(ACTION_COSTS.values())  # ~7秒


@dataclass
class TaskItem:
    """任务项"""
    task_type: TaskType
    brick_idx: int
    brick_id: int
    target_pos: Tuple[float, float, float]
    target_orn: Tuple[float, float, float]
    level: int
    priority: int = 0
    reason: str = ""
    is_temp: bool = False
    estimated_cost: float = 0.0
    
    def to_goal_pose(self) -> Tuple[float, float, float, float, float, float]:
        return (*self.target_pos, *self.target_orn)


class QPTaskScheduler:
    """基于 MILP 的动态任务调度器"""
    
    def __init__(self, env, threshold_low=0.015, threshold_critical=0.03):
        if not HAS_CVXPY:
            raise ImportError("cvxpy is required. Install with: pip install cvxpy")
        
        self.env = env
        self.threshold_low = threshold_low
        self.threshold_critical = threshold_critical
        
        self.dependency_map = self._build_dependency_map()
        self.placed_bricks_info: List[Dict] = []
        self.temp_positions = self._generate_temp_positions()
        self.used_temp_positions: Set[int] = set()
        self.bricks_in_temp: Dict[int, Tuple[float, float, float]] = {}
        
    def _build_dependency_map(self) -> Dict[int, List[int]]:
        """构建依赖关系图"""
        if hasattr(self.env, 'get_brick_dependencies'):
            dep_map = self.env.get_brick_dependencies()
            
            print(f"\n[QP] ═══════════════════════════════════════════════════")
            print(f"[QP] Brick Dependency Map:")
            for brick_idx in sorted(dep_map.keys()):
                deps = dep_map[brick_idx]
                if deps:
                    print(f"     Brick {brick_idx} depends on: {deps}")
                else:
                    print(f"     Brick {brick_idx} depends on: [] (base layer)")
            print(f"[QP] ═══════════════════════════════════════════════════\n")
            
            return dep_map
        
        raise ValueError("[QP] Environment must provide get_brick_dependencies()")
    
    def _generate_temp_positions(self) -> List[Tuple[float, float, float]]:
        """生成临时放置位置列表"""
        ground_z = 0.0
        if hasattr(self.env, 'get_ground_top'):
            ground_z = self.env.get_ground_top()
        
        L, W, H = 0.20, 0.10, 0.035
        if hasattr(self.env, 'cfg') and 'brick' in self.env.cfg:
            L, W, H = self.env.cfg['brick']['size_LWH']
        
        self.brick_L = L
        self.brick_W = W
        self.brick_H = H
        self.ground_z = ground_z
        self.temp_z = ground_z + H / 2
        self.temp_offset_distance = L + 0.1
        
        if not hasattr(self.env, 'layout_targets'):
            self.env._parse_layout()
        
        layout_targets = self.env.layout_targets
        
        if not layout_targets:
            self.stack_center_x = 0.0
            self.stack_center_y = 0.0
            return []
        
        xs = [t['xy'][0] for t in layout_targets]
        ys = [t['xy'][1] for t in layout_targets]
        
        self.stack_center_x = (min(xs) + max(xs)) / 2
        self.stack_center_y = (min(ys) + max(ys)) / 2
        
        fallback_positions = []
        for i in range(len(layout_targets)):
            if i % 2 == 0:
                tx = min(xs) - self.temp_offset_distance - (i // 2) * (L + 0.05)
            else:
                tx = max(xs) + self.temp_offset_distance + (i // 2) * (L + 0.05)
            ty = self.stack_center_y
            fallback_positions.append((tx, ty, self.temp_z))
        
        return fallback_positions
    
    # ================== 状态查询方法 ==================
    
    def get_temp_position_for_brick(self, brick_idx: int) -> Tuple[float, float, float]:
        """根据砖块期望位置计算临时位置（远离堆叠中心）"""
        if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
            target = self.env.layout_targets[brick_idx]
            expected_x, expected_y = target['xy']
            
            # 向远离中心的方向偏移
            if expected_x >= self.stack_center_x:
                temp_x = expected_x + self.temp_offset_distance
            else:
                temp_x = expected_x - self.temp_offset_distance
            
            temp_y = expected_y
            temp_z = self.temp_z
            
            # 冲突检测
            for other_idx, other_pos in self.bricks_in_temp.items():
                if other_idx != brick_idx:
                    dist = np.sqrt((temp_x - other_pos[0])**2 + (temp_y - other_pos[1])**2)
                    if dist < self.brick_L * 0.8:
                        if expected_x >= self.stack_center_x:
                            temp_x += self.brick_L + 0.05
                        else:
                            temp_x -= self.brick_L + 0.05
            
            return (temp_x, temp_y, temp_z)
        
        # 使用后备位置
        for i, pos in enumerate(self.temp_positions):
            if i not in self.used_temp_positions:
                self.used_temp_positions.add(i)
                return pos
        
        offset = len(self.used_temp_positions) * 0.15
        return (-0.4 - offset, 0.0, self.temp_z)
    
    def release_temp_position(self, pos: Tuple[float, float, float]):
        for i, temp_pos in enumerate(self.temp_positions):
            if np.allclose(pos, temp_pos, atol=0.01):
                self.used_temp_positions.discard(i)
                break
    
    def mark_brick_in_temp(self, brick_idx: int, temp_pos: Tuple[float, float, float]):
        self.bricks_in_temp[brick_idx] = temp_pos
        print(f"[QP] Marked brick {brick_idx} in temp position")
    
    def unmark_brick_from_temp(self, brick_idx: int):
        if brick_idx in self.bricks_in_temp:
            temp_pos = self.bricks_in_temp.pop(brick_idx)
            self.release_temp_position(temp_pos)
            print(f"[QP] Unmarked brick {brick_idx} from temp position")
    
    def is_brick_in_temp(self, brick_idx: int) -> bool:
        return brick_idx in self.bricks_in_temp
    
    def get_dependencies_for_brick(self, brick_idx: int) -> List[int]:
        return self.dependency_map.get(brick_idx, [])
    
    def get_all_ancestors(self, brick_idx: int) -> Set[int]:
        """递归获取所有祖先依赖"""
        ancestors = set()
        direct_deps = self.get_dependencies_for_brick(brick_idx)
        for dep in direct_deps:
            ancestors.add(dep)
            ancestors.update(self.get_all_ancestors(dep))
        return ancestors
    
    def get_all_dependents(self, brick_idx: int) -> Set[int]:
        """获取所有后代依赖（压在这个砖块上面的）"""
        dependents = set()
        for idx, deps in self.dependency_map.items():
            if brick_idx in deps:
                dependents.add(idx)
                dependents.update(self.get_all_dependents(idx))
        return dependents
    
    def check_brick_deviation(self, brick_id: int, expected_pos: np.ndarray) -> float:
        current_pos, _ = p.getBasePositionAndOrientation(brick_id)
        current_pos = np.array(current_pos)
        return np.linalg.norm(current_pos[:2] - expected_pos[:2])
    
    def check_all_placed_bricks(self) -> List[Dict]:
        """检查所有已放置砖块的偏差"""
        deviations = []
        for brick_info in self.placed_bricks_info:
            brick_id = brick_info["brick_id"]
            expected_pos = np.array(brick_info["expected_pos"])
            deviation = self.check_brick_deviation(brick_id, expected_pos)
            brick_idx = brick_info.get("brick_idx")
            is_in_temp = self.is_brick_in_temp(brick_idx)
            
            deviations.append({
                "brick_id": brick_id,
                "brick_idx": brick_idx,
                "deviation": deviation,
                "expected_pos": expected_pos,
                "expected_orn": brick_info.get("expected_orn", (0.0, 0.0, 0.0)),
                "level": brick_info.get("level", 0),
                "needs_repair": deviation > self.threshold_low and not is_in_temp,
                "is_in_temp": is_in_temp
            })
        
        return deviations
    
    def update_placed_bricks(self, placed_bricks_info: List[Dict]):
        self.placed_bricks_info = placed_bricks_info
    
    def get_bricks_needing_repair(self) -> List[Dict]:
        deviations = self.check_all_placed_bricks()
        return [d for d in deviations if d["needs_repair"]]
    
    def _get_brick_level(self, brick_idx: int) -> int:
        if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
            return self.env.layout_targets[brick_idx]["level"]
        return 0
    
    def should_replan(self) -> bool:
        return len(self.get_bricks_needing_repair()) > 0
    
    def _solve_with_milp(self, 
                         current_brick_idx: Optional[int],
                         remaining_sequence: List[int],
                         bricks_needing_repair: List[Dict],
                         is_holding_brick: bool) -> List[TaskItem]:
        """
        使用 MILP 求解最优任务序列
        """
        
        print("\n[QP-MILP] ═══════════════════════════════════════════════")
        print("[QP-MILP] Building MILP problem...")
        
        # ========== Step 1: 分析当前状态 ==========
        deviations = self.check_all_placed_bricks()
        deviation_map = {d["brick_idx"]: d for d in deviations}
        repair_set = {d["brick_idx"] for d in bricks_needing_repair}
        
        # 已正确放置的砖块
        placed_correctly = set()
        for d in deviations:
            if not d["needs_repair"] and not d["is_in_temp"]:
                placed_correctly.add(d["brick_idx"])
        
        print(f"[QP-MILP] Placed correctly: {placed_correctly}")
        print(f"[QP-MILP] Needs repair: {repair_set}")
        print(f"[QP-MILP] In temp: {set(self.bricks_in_temp.keys())}")
        print(f"[QP-MILP] Remaining to place: {remaining_sequence}")
        print(f"[QP-MILP] Currently holding: {current_brick_idx if is_holding_brick else 'None'}")
        
        # ========== Step 2: 确定需要处理的任务 ==========
        tasks_to_schedule = []  # [(brick_idx, task_type, must_use_temp)]
        scheduled_set = set()  # 用于避免重复: (brick_idx, task_type)
        
        # 如果正在抓着砖块
        held_brick = current_brick_idx if is_holding_brick else None
        
        # 检查抓着的砖块的依赖是否满足
        held_deps_ok = True
        if held_brick is not None:
            ancestors = self.get_all_ancestors(held_brick)
            ancestors_needing_repair = ancestors & repair_set
            ancestors_in_temp = ancestors & set(self.bricks_in_temp.keys())
            held_deps_ok = len(ancestors_needing_repair) == 0 and len(ancestors_in_temp) == 0
            
            if not held_deps_ok:
                tasks_to_schedule.append((held_brick, "TEMP", True))
                scheduled_set.add((held_brick, "TEMP"))
                tasks_to_schedule.append((held_brick, "RESTORE_HELD", False))
                scheduled_set.add((held_brick, "RESTORE_HELD"))
                print(f"[QP-MILP] Held brick {held_brick}: deps not OK, TEMP then RESTORE_HELD")
            else:
                tasks_to_schedule.append((held_brick, "PLACE", False))
                scheduled_set.add((held_brick, "PLACE"))
                print(f"[QP-MILP] Held brick {held_brick}: deps OK, direct PLACE")
        
        # 需要修复的砖块
        for brick_idx in repair_set:
            if brick_idx == held_brick:
                continue
            
            dependents = self.get_all_dependents(brick_idx)
            blocking = []
            for d in deviations:
                if d["brick_idx"] in dependents and not d["is_in_temp"]:
                    blocking.append(d["brick_idx"])
            
            for blocker in blocking:
                if (blocker, "TEMP") not in scheduled_set:
                    tasks_to_schedule.append((blocker, "TEMP", True))
                    scheduled_set.add((blocker, "TEMP"))
                    print(f"[QP-MILP] Brick {blocker}: blocking repair of {brick_idx}, TEMP")
            
            if (brick_idx, "REPAIR") not in scheduled_set:
                tasks_to_schedule.append((brick_idx, "REPAIR", False))
                scheduled_set.add((brick_idx, "REPAIR"))
                print(f"[QP-MILP] Brick {brick_idx}: REPAIR task")
            
            for blocker in blocking:
                if (blocker, "RESTORE") not in scheduled_set:
                    tasks_to_schedule.append((blocker, "RESTORE", False))
                    scheduled_set.add((blocker, "RESTORE"))
                    print(f"[QP-MILP] Brick {blocker}: RESTORE after repair")
        
        # 【关键修复】临时位置的砖块必须全部恢复，不管有没有后续依赖
        for temp_brick in self.bricks_in_temp.keys():
            if temp_brick == held_brick:
                continue
            if (temp_brick, "RESTORE") in scheduled_set:
                continue
            if (temp_brick, "RESTORE_HELD") in scheduled_set:
                continue
            
            # 【修复】无条件添加 RESTORE 任务 - 临时位置的砖块必须恢复到正确位置
            tasks_to_schedule.append((temp_brick, "RESTORE", False))
            scheduled_set.add((temp_brick, "RESTORE"))
            print(f"[QP-MILP] Brick {temp_brick}: RESTORE from temp (MANDATORY)")
        
        # 正常放置任务
        for brick_idx in remaining_sequence:
            if brick_idx == held_brick:
                continue
            if brick_idx in repair_set:
                continue
            if brick_idx in self.bricks_in_temp:
                continue
            if (brick_idx, "NORMAL") not in scheduled_set:
                tasks_to_schedule.append((brick_idx, "NORMAL", False))
                scheduled_set.add((brick_idx, "NORMAL"))
                print(f"[QP-MILP] Brick {brick_idx}: NORMAL place task")
        
        n_tasks = len(tasks_to_schedule)
        
        if n_tasks == 0:
            print("[QP-MILP] No tasks to schedule")
            return []
        
        print(f"\n[QP-MILP] Total tasks to schedule: {n_tasks}")
        for i, (brick_idx, task_type, must_temp) in enumerate(tasks_to_schedule):
            print(f"     Task {i}: brick={brick_idx}, type={task_type}, must_temp={must_temp}")
        
        # ========== Step 3: 构建 MILP 问题 ==========
        
        # 决策变量
        order = cp.Variable(n_tasks, integer=True)
        
        # 目标函数: 最小化总执行时间
        costs = []
        for i, (brick_idx, task_type, must_temp) in enumerate(tasks_to_schedule):
            # 第一个任务如果是放下手中砖块，只需要放置成本
            if i == 0 and held_brick is not None and brick_idx == held_brick and task_type in ["PLACE", "TEMP"]:
                base_cost = PLACE_ONLY_COST
            else:
                base_cost = FULL_PICK_PLACE_COST
            costs.append(base_cost)
        
        total_cost = sum(costs)
        objective = cp.Minimize(total_cost)
        
        # 约束
        constraints = []
        
        # 约束 1: 顺序范围
        constraints.append(order >= 0)
        constraints.append(order <= n_tasks - 1)
        
        # 约束 2: AllDifferent (顺序互不相同)
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                z = cp.Variable(boolean=True)
                M = n_tasks
                constraints.append(order[i] - order[j] >= 1 - M * z)
                constraints.append(order[j] - order[i] >= 1 - M * (1 - z))
        
        # 约束 3: 如果手持砖块，第一个任务必须是处理它 (PLACE 或 TEMP)
        if held_brick is not None:
            for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
                if brick_idx == held_brick and task_type in ["PLACE", "TEMP"]:
                    constraints.append(order[i] == 0)
                    print(f"[QP-MILP] Constraint: Task {i} (held brick {task_type}) must be order=0")
                    break
        
        # 构建任务索引映射
        task_indices = {}  # brick_idx -> [(task_idx, task_type)]
        for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
            if brick_idx not in task_indices:
                task_indices[brick_idx] = []
            task_indices[brick_idx].append((i, task_type))
        
        # 约束 4: 放置依赖 (NORMAL, PLACE, REPAIR, RESTORE, RESTORE_HELD 都需要依赖满足)
        for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
            if task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
                deps = self.get_dependencies_for_brick(brick_idx)
                for dep in deps:
                    # 依赖砖块必须已经在正确位置
                    if dep in placed_correctly:
                        # 已经正确放置，无需约束
                        continue
                    
                    if dep in task_indices:
                        # 找到依赖砖块的放置任务 (NORMAL, PLACE, REPAIR, RESTORE)
                        for dep_task_idx, dep_task_type in task_indices[dep]:
                            if dep_task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
                                constraints.append(order[dep_task_idx] <= order[i] - 1)
                                print(f"[QP-MILP] Dep constraint: Task {dep_task_idx} ({dep}.{dep_task_type}) "
                                      f"before Task {i} ({brick_idx}.{task_type})")
        
        # 约束 5: TEMP 必须在对应 REPAIR 之前，RESTORE 必须在 REPAIR 之后
        for repair_brick in repair_set:
            if repair_brick not in task_indices:
                continue
            
            repair_task_idx = None
            for idx, ttype in task_indices[repair_brick]:
                if ttype == "REPAIR":
                    repair_task_idx = idx
                    break
            
            if repair_task_idx is None:
                continue
            
            # 找出阻挡这个修复的砖块
            dependents = self.get_all_dependents(repair_brick)
            for blocker in dependents:
                if blocker in task_indices:
                    for idx, ttype in task_indices[blocker]:
                        if ttype == "TEMP":
                            constraints.append(order[idx] <= order[repair_task_idx] - 1)
                            print(f"[QP-MILP] Constraint: TEMP {idx} before REPAIR {repair_task_idx}")
                        elif ttype == "RESTORE":
                            constraints.append(order[idx] >= order[repair_task_idx] + 1)
                            print(f"[QP-MILP] Constraint: RESTORE {idx} after REPAIR {repair_task_idx}")
        
        # 约束 6: RESTORE_HELD 必须在所有阻挡它的 REPAIR 完成之后
        if held_brick is not None and not held_deps_ok:
            restore_held_idx = None
            for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
                if brick_idx == held_brick and task_type == "RESTORE_HELD":
                    restore_held_idx = i
                    break
            
            if restore_held_idx is not None:
                ancestors = self.get_all_ancestors(held_brick)
                for ancestor in ancestors:
                    if ancestor in repair_set and ancestor in task_indices:
                        for idx, ttype in task_indices[ancestor]:
                            if ttype == "REPAIR":
                                constraints.append(order[restore_held_idx] >= order[idx] + 1)
                                print(f"[QP-MILP] Constraint: RESTORE_HELD {restore_held_idx} "
                                      f"after REPAIR {idx} (ancestor {ancestor})")
                    # 也要在临时位置砖块恢复之后
                    if ancestor in self.bricks_in_temp and ancestor in task_indices:
                        for idx, ttype in task_indices[ancestor]:
                            if ttype == "RESTORE":
                                constraints.append(order[restore_held_idx] >= order[idx] + 1)
                                print(f"[QP-MILP] Constraint: RESTORE_HELD {restore_held_idx} "
                                      f"after RESTORE {idx} (ancestor {ancestor} in temp)")

        # ========== 【新增】约束 7: 依赖临时位置砖块的任务必须在 RESTORE 之后 ==========
        # 如果砖块 A 被移到临时位置，那么所有依赖 A 的砖块必须在 A 恢复之后才能放置
        for temp_brick in self.bricks_in_temp.keys():
            if temp_brick not in task_indices:
                continue
            
            # 找到这个砖块的 RESTORE 任务
            restore_task_idx = None
            for idx, ttype in task_indices[temp_brick]:
                if ttype in ["RESTORE", "RESTORE_HELD"]:
                    restore_task_idx = idx
                    break
            
            if restore_task_idx is None:
                continue
            
            # 找出所有依赖这个临时砖块的任务
            for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
                if task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
                    # 检查这个任务的砖块是否依赖临时位置的砖块
                    deps = self.get_dependencies_for_brick(brick_idx)
                    if temp_brick in deps:
                        # 这个任务依赖临时位置的砖块，必须在 RESTORE 之后
                        if i != restore_task_idx:  # 不是自己
                            constraints.append(order[i] >= order[restore_task_idx] + 1)
                            print(f"[QP-MILP] Temp-dep constraint: Task {i} ({brick_idx}.{task_type}) "
                                  f"must be after RESTORE {restore_task_idx} (temp brick {temp_brick})")
        
        # ========== 【新增】约束 8: TEMP 任务中，依赖关系也要考虑 ==========
        # 如果将要被 TEMP 的砖块被其他任务依赖，那些任务必须等 RESTORE 完成
        for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
            if task_type == "TEMP":
                # 找到对应的 RESTORE 任务
                restore_task_idx = None
                for idx, ttype in task_indices.get(brick_idx, []):
                    if ttype in ["RESTORE", "RESTORE_HELD"]:
                        restore_task_idx = idx
                        break
                
                if restore_task_idx is None:
                    continue
                
                # 找出所有依赖这个砖块的其他任务
                for j, (other_brick, other_type, _) in enumerate(tasks_to_schedule):
                    if other_type in ["NORMAL", "PLACE"]:
                        deps = self.get_dependencies_for_brick(other_brick)
                        if brick_idx in deps:
                            # 这个任务依赖将被 TEMP 的砖块
                            constraints.append(order[j] >= order[restore_task_idx] + 1)
                            print(f"[QP-MILP] Future-temp-dep: Task {j} ({other_brick}.{other_type}) "
                                  f"must wait for RESTORE {restore_task_idx} of brick {brick_idx}")
                                    
        # ========== Step 4: 求解 ==========
        print(f"\n[QP-MILP] Solving MILP with {len(constraints)} constraints...")
        
        prob = cp.Problem(objective, constraints)
        
        solvers_to_try = [cp.GLPK_MI, cp.CBC, cp.SCIP, cp.ECOS_BB]
        solved = False
        
        for solver in solvers_to_try:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    solved = True
                    print(f"[QP-MILP] Solved with {solver}! Status: {prob.status}")
                    break
            except Exception as e:
                print(f"[QP-MILP] Solver {solver} failed: {e}")
                continue
        
        if not solved:
            raise RuntimeError(f"[QP-MILP] Failed to solve! Status: {prob.status}")
        
        # ========== Step 5: 构建任务序列 ==========
        # 【修复】对 order 值取整
        order_values = [int(round(v)) for v in order.value]
        print(f"\n[QP-MILP] Solution order: {order_values}")
        
        # 按顺序排列任务
        sorted_indices = sorted(range(n_tasks), key=lambda i: order_values[i])
        
        task_sequence = []
        
# 在 _solve_with_milp 方法的 Step 5 中，修改 REPAIR/RESTORE 部分

        for task_idx in sorted_indices:
            brick_idx, task_type, must_temp = tasks_to_schedule[task_idx]
            goal = self.env.compute_goal_pose_from_layout(brick_idx)
            level = self._get_brick_level(brick_idx)
            
            if task_type == "TEMP":
                # 放到临时位置
                temp_pos = self.get_temp_position_for_brick(brick_idx)
                is_held_first = (brick_idx == held_brick and order_values[task_idx] == 0)
                task_sequence.append(TaskItem(
                    task_type=TaskType.TEMP_PLACE,
                    brick_idx=brick_idx,
                    brick_id=self.env.brick_ids[brick_idx],
                    target_pos=temp_pos,
                    target_orn=(0.0, 0.0, 0.0),
                    level=level,
                    priority=len(task_sequence),
                    reason=f"MILP: temp placement",
                    is_temp=True,
                    estimated_cost=PLACE_ONLY_COST if is_held_first else FULL_PICK_PLACE_COST
                ))
                
            elif task_type in ["NORMAL", "PLACE"]:
                # 正常放置
                is_held_first = (brick_idx == held_brick and order_values[task_idx] == 0)
                task_sequence.append(TaskItem(
                    task_type=TaskType.NORMAL_PLACE,
                    brick_idx=brick_idx,
                    brick_id=self.env.brick_ids[brick_idx],
                    target_pos=goal[:3],
                    target_orn=goal[3:],
                    level=level,
                    priority=len(task_sequence),
                    reason=f"MILP: normal placement",
                    estimated_cost=PLACE_ONLY_COST if is_held_first else FULL_PICK_PLACE_COST
                ))
                
            elif task_type in ["REPAIR", "RESTORE", "RESTORE_HELD"]:
                # 【关键修复】区分 REPAIR 和 RESTORE
                if task_type == "REPAIR":
                    # REPAIR: 砖块在原位但偏移了，使用 deviation_map 中的期望位置
                    if brick_idx in deviation_map:
                        d = deviation_map[brick_idx]
                        # 【修复】检查 deviation_map 中的位置是否是临时位置
                        # 如果是临时位置，应该使用 layout 中的正确位置
                        if d.get("is_in_temp", False):
                            target_pos = goal[:3]
                            target_orn = goal[3:]
                            reason = f"MILP: repair (from temp)"
                        else:
                            target_pos = tuple(d["expected_pos"])
                            target_orn = d["expected_orn"]
                            reason = f"MILP: repair (dev={d['deviation']*1000:.1f}mm)"
                    else:
                        target_pos = goal[:3]
                        target_orn = goal[3:]
                        reason = f"MILP: repair"
                else:
                    # RESTORE / RESTORE_HELD: 砖块在临时位置，需要恢复到正确位置
                    # 【关键】始终使用 layout 中定义的正确目标位置
                    target_pos = goal[:3]
                    target_orn = goal[3:]
                    reason = f"MILP: {task_type.lower()} (to correct position)"
                
                task_sequence.append(TaskItem(
                    task_type=TaskType.REPAIR_PLACE,
                    brick_idx=brick_idx,
                    brick_id=self.env.brick_ids[brick_idx],
                    target_pos=target_pos,
                    target_orn=target_orn,
                    level=level,
                    priority=len(task_sequence),
                    reason=reason,
                    estimated_cost=FULL_PICK_PLACE_COST
                ))
        
        print(f"\n[QP-MILP] Generated {len(task_sequence)} tasks")
        print("[QP-MILP] ═══════════════════════════════════════════════\n")
        
        return task_sequence
    # ================== 主入口 ==================
    
    def plan_task_sequence(self, 
                          current_brick_idx: int,
                          remaining_sequence: List[int],
                          is_holding_brick: bool = False) -> List[TaskItem]:
        """
        规划任务序列（主入口）
        
        使用 MILP 优化求解最优任务序列，最小化执行时间
        """
        print(f"\n[QP] ═══════════════════════════════════════════════════")
        print(f"[QP] Planning task sequence with MILP optimization...")
        print(f"[QP] Current brick: {current_brick_idx}")
        print(f"[QP] Remaining sequence: {remaining_sequence}")
        print(f"[QP] Is holding brick: {is_holding_brick}")
        print(f"[QP] Bricks in temp: {list(self.bricks_in_temp.keys())}")
        
        # 获取需要修复的砖块
        bricks_needing_repair = self.get_bricks_needing_repair()
        
        # 打印当前状态
        deviations = self.check_all_placed_bricks()
        print(f"[QP] Checking {len(deviations)} placed bricks:")
        for d in deviations:
            if d["is_in_temp"]:
                status = "📦 IN TEMP"
            elif d["needs_repair"]:
                status = "⚠️ NEED REPAIR"
            else:
                status = "✓ OK"
            print(f"     Brick {d['brick_idx']}: deviation={d['deviation']*1000:.2f}mm {status}")
        
        # 使用 MILP 求解
        task_sequence = self._solve_with_milp(
            current_brick_idx, remaining_sequence,
            bricks_needing_repair, is_holding_brick
        )
        
        # 计算总成本
        total_cost = sum(t.estimated_cost for t in task_sequence)
        
        # 打印结果
        print(f"\n[QP] Planned {len(task_sequence)} tasks (est. time: {total_cost:.1f}s):")
        for i, task in enumerate(task_sequence):
            temp_marker = " [TEMP]" if task.is_temp else ""
            print(f"     [{i}] {task.task_type.value}: brick={task.brick_idx}, "
                  f"cost={task.estimated_cost:.1f}s{temp_marker}")
            print(f"         reason: {task.reason}")
        print(f"[QP] ═══════════════════════════════════════════════════\n")
        
        return task_sequence

# """
# 简化版任务调度器
# 功能: 基于距离选择最近的可用砖块，按 Level 顺序填充槽位
# """

# import numpy as np
# import pybullet as p
# from typing import List, Dict, Set, Optional, Tuple
# from enum import Enum
# from dataclasses import dataclass


# class TaskType(Enum):
#     """任务类型枚举"""
#     NORMAL_PLACE = "normal_place"      # 正常放置新砖块


# class SlotStatus(Enum):
#     """槽位状态"""
#     EMPTY = "empty"
#     FILLED = "filled"


# @dataclass
# class Slot:
#     """目标槽位"""
#     slot_idx: int
#     level: int
#     goal_pos: np.ndarray
#     goal_orn: np.ndarray
#     status: SlotStatus = SlotStatus.EMPTY
#     filled_brick_id: Optional[int] = None


# @dataclass
# class TaskItem:
#     """任务项"""
#     task_type: TaskType
#     brick_idx: int
#     brick_id: int
#     source_pos: Tuple[float, float, float]
#     target_pos: Tuple[float, float, float]
#     target_orn: Tuple[float, float, float]
#     level: int
#     slot_idx: int
#     reason: str = ""
#     estimated_cost: float = 0.0
    
#     def to_goal_pose(self) -> Tuple[float, float, float, float, float, float]:
#         return (*self.target_pos, *self.target_orn)


# class QPTaskScheduler:
#     """简化版任务调度器：距离优先 + Level 顺序"""
    
#     def __init__(self, env, 
#                  fill_threshold=0.05):  # 5cm 以内视为已填充
        
#         self.env = env
#         self.fill_threshold = fill_threshold
        
#         # 砖块尺寸
#         self.brick_L, self.brick_W, self.brick_H = env.cfg["brick"]["size_LWH"]
#         self.ground_z = env.get_ground_top() if hasattr(env, 'get_ground_top') else 0.0
        
#         # 初始化槽位
#         self._init_slots()
        
#         # 已放置砖块集合 (brick_id)
#         self.placed_brick_ids: Set[int] = set()
        
#         self._print_init_info()
    
#     def _init_slots(self):
#         """从 layout 初始化槽位"""
#         self.slots: List[Slot] = []
        
#         if not hasattr(self.env, 'layout_targets'):
#             self.env._parse_layout()
        
#         layout_targets = self.env.layout_targets
#         yaw = self.env.cfg["goal"]["yaw"]
        
#         for idx, target in enumerate(layout_targets):
#             level = target["level"]
#             xy = target["xy"]
#             gz = self.ground_z + self.brick_H / 2 + level * self.brick_H
            
#             self.slots.append(Slot(
#                 slot_idx=idx,
#                 level=level,
#                 goal_pos=np.array([xy[0], xy[1], gz]),
#                 goal_orn=np.array([0.0, 0.0, yaw]),
#                 status=SlotStatus.EMPTY
#             ))
        
#         # 按 level 排序
#         self.slots.sort(key=lambda s: (s.level, s.slot_idx))
#         self.max_level = max(s.level for s in self.slots) if self.slots else 0
    
#     def _print_init_info(self):
#         """打印初始化信息"""
#         print(f"\n[QP] ═══════════════════════════════════════════════════")
#         print(f"[QP] 简化版调度器初始化 (距离优先 + Level顺序)")
#         print(f"[QP] 填充阈值: < {self.fill_threshold*100:.1f}cm")
#         print(f"[QP] 槽位信息:")
#         for level in range(self.max_level + 1):
#             level_slots = [s for s in self.slots if s.level == level]
#             print(f"     Level {level}: {len(level_slots)} 个槽位")
#         print(f"[QP] ═══════════════════════════════════════════════════\n")
    
#     # ================== TCP 位置 ==================
    
#     def get_current_tcp_position(self) -> np.ndarray:
#         """获取当前 TCP 位置"""
#         if hasattr(self.env, 'robot_model'):
#             rm = self.env.robot_model
#             tcp_state = p.getLinkState(rm.id, rm.ee_link)
#             return np.array(tcp_state[0])
#         return np.array([0.0, 0.0, 0.5])
    
#     # ================== 槽位状态管理 ==================
    
#     def update_slot_status(self):
#         """更新所有槽位的状态"""
#         # 重置
#         for slot in self.slots:
#             slot.status = SlotStatus.EMPTY
#             slot.filled_brick_id = None
        
#         # 检查每个砖块
#         for brick_id in self.env.brick_ids:
#             if brick_id in self.placed_brick_ids:
#                 continue  # 已标记为放置，跳过重复检查
            
#             try:
#                 pos, _ = p.getBasePositionAndOrientation(brick_id)
#                 pos = np.array(pos)
                
#                 # 找最匹配的槽位
#                 best_slot = None
#                 best_dist = float('inf')
                
#                 for slot in self.slots:
#                     if slot.filled_brick_id is not None:
#                         continue  # 已被其他砖块占用
                    
#                     # XY 距离
#                     xy_dist = np.linalg.norm(pos[:2] - slot.goal_pos[:2])
#                     # Z 高度差
#                     z_diff = abs(pos[2] - slot.goal_pos[2])
                    
#                     # Z 高度必须接近
#                     if z_diff > self.brick_H * 0.8:
#                         continue
                    
#                     if xy_dist < best_dist and xy_dist < self.fill_threshold:
#                         best_dist = xy_dist
#                         best_slot = slot
                
#                 if best_slot is not None:
#                     best_slot.status = SlotStatus.FILLED
#                     best_slot.filled_brick_id = brick_id
#                     self.placed_brick_ids.add(brick_id)
                    
#             except Exception as e:
#                 print(f"[QP] Error checking brick {brick_id}: {e}")
    
#     def get_empty_slots_in_level(self, level: int) -> List[Slot]:
#         """获取某层的空槽位"""
#         return [s for s in self.slots if s.level == level and s.status == SlotStatus.EMPTY]
    
#     def is_level_complete(self, level: int) -> bool:
#         """检查某层是否完成"""
#         level_slots = [s for s in self.slots if s.level == level]
#         return all(s.status == SlotStatus.FILLED for s in level_slots)
    
#     def get_current_working_level(self) -> int:
#         """获取当前应该工作的 Level"""
#         for level in range(self.max_level + 1):
#             if not self.is_level_complete(level):
#                 return level
#         return self.max_level
    
#     def all_slots_filled(self) -> bool:
#         """检查是否所有槽位都已填充"""
#         return all(s.status == SlotStatus.FILLED for s in self.slots)
    
#     # ================== 砖块选择 ==================
    
#     def get_available_bricks(self) -> List[Tuple[int, int, np.ndarray]]:
#         """
#         获取所有可用砖块（未被放置到槽位的）
        
#         Returns:
#             List of (brick_idx, brick_id, position)
#         """
#         available = []
        
#         for idx, brick_id in enumerate(self.env.brick_ids):
#             # 跳过已放置的
#             if brick_id in self.placed_brick_ids:
#                 continue
            
#             try:
#                 pos, _ = p.getBasePositionAndOrientation(brick_id)
#                 pos = np.array(pos)
#                 available.append((idx, brick_id, pos))
#             except:
#                 pass
        
#         return available
    
#     def find_nearest_brick(self, tcp_pos: np.ndarray) -> Optional[Tuple[int, int, np.ndarray]]:
#         """找到距离 TCP 最近的可用砖块"""
#         available = self.get_available_bricks()
        
#         if not available:
#             return None
        
#         # 按距离排序
#         available.sort(key=lambda b: np.linalg.norm(b[2] - tcp_pos))
        
#         return available[0]
    
#     def find_nearest_slot(self, empty_slots: List[Slot], brick_pos: np.ndarray) -> Slot:
#         """找到距离砖块最近的空槽位"""
#         return min(empty_slots, key=lambda s: np.linalg.norm(s.goal_pos[:2] - brick_pos[:2]))
    
#     # ================== 距离计算 ==================
    
#     def calculate_task_distance(self, brick_pos: np.ndarray, 
#                                  slot_pos: np.ndarray, 
#                                  tcp_pos: np.ndarray) -> float:
#         """计算任务总距离"""
#         d1 = np.linalg.norm(tcp_pos - brick_pos)  # TCP -> 砖块
#         d2 = np.linalg.norm(brick_pos - slot_pos)  # 砖块 -> 槽位
#         return d1 + d2
    
#     # ================== 主规划方法 ==================
    
#     def plan_next_task(self) -> Optional[TaskItem]:
#         """
#         规划下一个任务
        
#         策略：
#         1. 更新槽位状态
#         2. 找当前工作 Level 的空槽位
#         3. 选择距离 TCP 最近的可用砖块
#         4. 选择距离砖块最近的空槽位
#         """
#         # 更新状态
#         self.update_slot_status()
        
#         tcp_pos = self.get_current_tcp_position()
        
#         print(f"\n[QP] ═══════════════════════════════════════════════════")
#         print(f"[QP] 规划下一个任务")
#         print(f"[QP] TCP 位置: ({tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f})")
        
#         # 打印槽位状态
#         self._print_slot_status()
        
#         # 检查是否完成
#         if self.all_slots_filled():
#             print(f"[QP] ✅ 所有槽位已填充!")
#             print(f"[QP] ═══════════════════════════════════════════════════\n")
#             return None
        
#         # 获取当前工作 Level
#         current_level = self.get_current_working_level()
#         print(f"[QP] 当前工作 Level: {current_level}")
        
#         # 获取空槽位
#         empty_slots = self.get_empty_slots_in_level(current_level)
        
#         if not empty_slots:
#             print(f"[QP] Level {current_level} 没有空槽位")
#             print(f"[QP] ═══════════════════════════════════════════════════\n")
#             return None
        
#         print(f"[QP] Level {current_level} 有 {len(empty_slots)} 个空槽位")
        
#         # 找最近的可用砖块
#         nearest_brick = self.find_nearest_brick(tcp_pos)
        
#         if nearest_brick is None:
#             print(f"[QP] ⚠️ 没有可用砖块!")
#             print(f"[QP] ═══════════════════════════════════════════════════\n")
#             return None
        
#         brick_idx, brick_id, brick_pos = nearest_brick
#         print(f"[QP] 选择砖块: idx={brick_idx}, id={brick_id}")
#         print(f"[QP] 砖块位置: ({brick_pos[0]:.3f}, {brick_pos[1]:.3f}, {brick_pos[2]:.3f})")
        
#         # 找最近的空槽位
#         target_slot = self.find_nearest_slot(empty_slots, brick_pos)
#         print(f"[QP] 目标槽位: Level {target_slot.level}, Slot {target_slot.slot_idx}")
#         print(f"[QP] 目标位置: ({target_slot.goal_pos[0]:.3f}, {target_slot.goal_pos[1]:.3f}, {target_slot.goal_pos[2]:.3f})")
        
#         # 计算距离
#         total_dist = self.calculate_task_distance(brick_pos, target_slot.goal_pos, tcp_pos)
#         print(f"[QP] 预计距离: {total_dist:.2f}m")
        
#         task = TaskItem(
#             task_type=TaskType.NORMAL_PLACE,
#             brick_idx=brick_idx,
#             brick_id=brick_id,
#             source_pos=tuple(brick_pos),
#             target_pos=tuple(target_slot.goal_pos),
#             target_orn=tuple(target_slot.goal_orn),
#             level=target_slot.level,
#             slot_idx=target_slot.slot_idx,
#             reason=f"放置到 Level {target_slot.level} 槽位 {target_slot.slot_idx}",
#             estimated_cost=total_dist
#         )
        
#         print(f"[QP] ═══════════════════════════════════════════════════\n")
        
#         return task
    
#     def _print_slot_status(self):
#         """打印槽位状态"""
#         print(f"[QP] 槽位状态:")
#         for level in range(self.max_level + 1):
#             level_slots = [s for s in self.slots if s.level == level]
#             status_str = "  ".join([
#                 f"S{s.slot_idx}:{'✓' if s.status == SlotStatus.FILLED else '○'}"
#                 for s in level_slots
#             ])
#             print(f"     Level {level}:   {status_str}")
        
#         available = self.get_available_bricks()
#         print(f"[QP] 可用砖块: {len(available)} 个")
    
#     def mark_brick_placed(self, brick_id: int):
#         """标记砖块已放置"""
#         self.placed_brick_ids.add(brick_id)
#         print(f"[QP] 标记砖块 {brick_id} 为已放置")
    
#     def get_slot_status_string(self) -> str:
#         """获取槽位状态字符串（用于打印）"""
#         lines = ["[QP] 槽位状态:"]
#         for level in range(self.max_level + 1):
#             level_slots = [s for s in self.slots if s.level == level]
#             status_str = "  ".join([
#                 f"S{s.slot_idx}:{'✓' if s.status == SlotStatus.FILLED else '○'}"
#                 for s in level_slots
#             ])
#             lines.append(f"     Level {level}:   {status_str}")
#         return "\n".join(lines)