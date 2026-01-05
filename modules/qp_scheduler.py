# """
# QP 任务调度器 (带依赖约束)
# 功能: 检测已放置砖块的偏移，动态调整任务序列
# 关键: 使用 MILP 优化求解最优任务序列
# """

# import numpy as np
# import pybullet as p
# from typing import List, Dict, Set, Optional, Tuple
# from enum import Enum
# from dataclasses import dataclass

# try:
#     import cvxpy as cp
#     HAS_CVXPY = True
# except ImportError:
#     HAS_CVXPY = False
#     raise ImportError("cvxpy is required for QP optimization. Install with: pip install cvxpy")


# class TaskType(Enum):
#     """任务类型枚举"""
#     NORMAL_PLACE = "normal_place"      # 正常放置新砖块
#     REPAIR_PLACE = "repair_place"      # 修复已放置的砖块
#     TEMP_PLACE = "temp_place"          # 临时放置（移开碍事的砖块）


# class ActionType(Enum):
#     """原子动作类型"""
#     PRE_GRASP = "pre_grasp"
#     DESCEND = "descend"
#     CLOSE = "close"
#     LIFT = "lift"
#     PRE_PLACE = "pre_place"
#     DESCEND_PLACE = "descend_place"
#     RELEASE = "release"


# # 每个动作的估计时间成本（秒）
# ACTION_COSTS = {
#     ActionType.PRE_GRASP: 1.5,
#     ActionType.DESCEND: 1.0,
#     ActionType.CLOSE: 0.5,
#     ActionType.LIFT: 1.0,
#     ActionType.PRE_PLACE: 1.5,
#     ActionType.DESCEND_PLACE: 1.0,
#     ActionType.RELEASE: 0.5,
# }

# # 成本常量
# PLACE_ONLY_COST = (ACTION_COSTS[ActionType.PRE_PLACE] + 
#                    ACTION_COSTS[ActionType.DESCEND_PLACE] + 
#                    ACTION_COSTS[ActionType.RELEASE])  # ~3秒

# FULL_PICK_PLACE_COST = sum(ACTION_COSTS.values())  # ~7秒


# @dataclass
# class TaskItem:
#     """任务项"""
#     task_type: TaskType
#     brick_idx: int
#     brick_id: int
#     target_pos: Tuple[float, float, float]
#     target_orn: Tuple[float, float, float]
#     level: int
#     priority: int = 0
#     reason: str = ""
#     is_temp: bool = False
#     estimated_cost: float = 0.0
    
#     def to_goal_pose(self) -> Tuple[float, float, float, float, float, float]:
#         return (*self.target_pos, *self.target_orn)


# class QPTaskScheduler:
#     """基于 MILP 的动态任务调度器"""
    
#     def __init__(self, env, threshold_low=0.015, threshold_critical=0.03):
#         if not HAS_CVXPY:
#             raise ImportError("cvxpy is required. Install with: pip install cvxpy")
        
#         self.env = env
#         self.threshold_low = threshold_low
#         self.threshold_critical = threshold_critical
        
#         self.dependency_map = self._build_dependency_map()
#         self.placed_bricks_info: List[Dict] = []
#         self.temp_positions = self._generate_temp_positions()
#         self.used_temp_positions: Set[int] = set()
#         self.bricks_in_temp: Dict[int, Tuple[float, float, float]] = {}
        
#     def _build_dependency_map(self) -> Dict[int, List[int]]:
#         """构建依赖关系图"""
#         if hasattr(self.env, 'get_brick_dependencies'):
#             dep_map = self.env.get_brick_dependencies()
            
#             print(f"\n[QP] ═══════════════════════════════════════════════════")
#             print(f"[QP] Brick Dependency Map:")
#             for brick_idx in sorted(dep_map.keys()):
#                 deps = dep_map[brick_idx]
#                 if deps:
#                     print(f"     Brick {brick_idx} depends on: {deps}")
#                 else:
#                     print(f"     Brick {brick_idx} depends on: [] (base layer)")
#             print(f"[QP] ═══════════════════════════════════════════════════\n")
            
#             return dep_map
        
#         raise ValueError("[QP] Environment must provide get_brick_dependencies()")
    
#     def _generate_temp_positions(self) -> List[Tuple[float, float, float]]:
#         """生成临时放置位置列表"""
#         ground_z = 0.0
#         if hasattr(self.env, 'get_ground_top'):
#             ground_z = self.env.get_ground_top()
        
#         L, W, H = 0.20, 0.10, 0.035
#         if hasattr(self.env, 'cfg') and 'brick' in self.env.cfg:
#             L, W, H = self.env.cfg['brick']['size_LWH']
        
#         self.brick_L = L
#         self.brick_W = W
#         self.brick_H = H
#         self.ground_z = ground_z
#         self.temp_z = ground_z + H / 2
#         self.temp_offset_distance = L + 0.1
        
#         if not hasattr(self.env, 'layout_targets'):
#             self.env._parse_layout()
        
#         layout_targets = self.env.layout_targets
        
#         if not layout_targets:
#             self.stack_center_x = 0.0
#             self.stack_center_y = 0.0
#             return []
        
#         xs = [t['xy'][0] for t in layout_targets]
#         ys = [t['xy'][1] for t in layout_targets]
        
#         self.stack_center_x = (min(xs) + max(xs)) / 2
#         self.stack_center_y = (min(ys) + max(ys)) / 2
        
#         fallback_positions = []
#         for i in range(len(layout_targets)):
#             if i % 2 == 0:
#                 tx = min(xs) - self.temp_offset_distance - (i // 2) * (L + 0.05)
#             else:
#                 tx = max(xs) + self.temp_offset_distance + (i // 2) * (L + 0.05)
#             ty = self.stack_center_y
#             fallback_positions.append((tx, ty, self.temp_z))
        
#         return fallback_positions
    
#     # ================== 状态查询方法 ==================
    
#     def get_temp_position_for_brick(self, brick_idx: int) -> Tuple[float, float, float]:
#         """根据砖块期望位置计算临时位置（远离堆叠中心）"""
#         if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
#             target = self.env.layout_targets[brick_idx]
#             expected_x, expected_y = target['xy']
            
#             # 向远离中心的方向偏移
#             if expected_x >= self.stack_center_x:
#                 temp_x = expected_x + self.temp_offset_distance
#             else:
#                 temp_x = expected_x - self.temp_offset_distance
            
#             temp_y = expected_y
#             temp_z = self.temp_z
            
#             # 冲突检测
#             for other_idx, other_pos in self.bricks_in_temp.items():
#                 if other_idx != brick_idx:
#                     dist = np.sqrt((temp_x - other_pos[0])**2 + (temp_y - other_pos[1])**2)
#                     if dist < self.brick_L * 0.8:
#                         if expected_x >= self.stack_center_x:
#                             temp_x += self.brick_L + 0.05
#                         else:
#                             temp_x -= self.brick_L + 0.05
            
#             return (temp_x, temp_y, temp_z)
        
#         # 使用后备位置
#         for i, pos in enumerate(self.temp_positions):
#             if i not in self.used_temp_positions:
#                 self.used_temp_positions.add(i)
#                 return pos
        
#         offset = len(self.used_temp_positions) * 0.15
#         return (-0.4 - offset, 0.0, self.temp_z)
    
#     def release_temp_position(self, pos: Tuple[float, float, float]):
#         for i, temp_pos in enumerate(self.temp_positions):
#             if np.allclose(pos, temp_pos, atol=0.01):
#                 self.used_temp_positions.discard(i)
#                 break
    
#     def mark_brick_in_temp(self, brick_idx: int, temp_pos: Tuple[float, float, float]):
#         self.bricks_in_temp[brick_idx] = temp_pos
#         print(f"[QP] Marked brick {brick_idx} in temp position")
    
#     def unmark_brick_from_temp(self, brick_idx: int):
#         if brick_idx in self.bricks_in_temp:
#             temp_pos = self.bricks_in_temp.pop(brick_idx)
#             self.release_temp_position(temp_pos)
#             print(f"[QP] Unmarked brick {brick_idx} from temp position")
    
#     def is_brick_in_temp(self, brick_idx: int) -> bool:
#         return brick_idx in self.bricks_in_temp
    
#     def get_dependencies_for_brick(self, brick_idx: int) -> List[int]:
#         return self.dependency_map.get(brick_idx, [])
    
#     def get_all_ancestors(self, brick_idx: int) -> Set[int]:
#         """递归获取所有祖先依赖"""
#         ancestors = set()
#         direct_deps = self.get_dependencies_for_brick(brick_idx)
#         for dep in direct_deps:
#             ancestors.add(dep)
#             ancestors.update(self.get_all_ancestors(dep))
#         return ancestors
    
#     def get_all_dependents(self, brick_idx: int) -> Set[int]:
#         """获取所有后代依赖（压在这个砖块上面的）"""
#         dependents = set()
#         for idx, deps in self.dependency_map.items():
#             if brick_idx in deps:
#                 dependents.add(idx)
#                 dependents.update(self.get_all_dependents(idx))
#         return dependents
    
#     def check_brick_deviation(self, brick_id: int, expected_pos: np.ndarray) -> float:
#         current_pos, _ = p.getBasePositionAndOrientation(brick_id)
#         current_pos = np.array(current_pos)
#         return np.linalg.norm(current_pos[:2] - expected_pos[:2])
    
#     def check_all_placed_bricks(self) -> List[Dict]:
#         """检查所有已放置砖块的偏差"""
#         deviations = []
#         for brick_info in self.placed_bricks_info:
#             brick_id = brick_info["brick_id"]
#             expected_pos = np.array(brick_info["expected_pos"])
#             deviation = self.check_brick_deviation(brick_id, expected_pos)
#             brick_idx = brick_info.get("brick_idx")
#             is_in_temp = self.is_brick_in_temp(brick_idx)
            
#             deviations.append({
#                 "brick_id": brick_id,
#                 "brick_idx": brick_idx,
#                 "deviation": deviation,
#                 "expected_pos": expected_pos,
#                 "expected_orn": brick_info.get("expected_orn", (0.0, 0.0, 0.0)),
#                 "level": brick_info.get("level", 0),
#                 "needs_repair": deviation > self.threshold_low and not is_in_temp,
#                 "is_in_temp": is_in_temp
#             })
        
#         return deviations
    
#     def update_placed_bricks(self, placed_bricks_info: List[Dict]):
#         self.placed_bricks_info = placed_bricks_info
    
#     def get_bricks_needing_repair(self) -> List[Dict]:
#         deviations = self.check_all_placed_bricks()
#         return [d for d in deviations if d["needs_repair"]]
    
#     def _get_brick_level(self, brick_idx: int) -> int:
#         if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
#             return self.env.layout_targets[brick_idx]["level"]
#         return 0
    
#     def should_replan(self) -> bool:
#         return len(self.get_bricks_needing_repair()) > 0
    
#     def _solve_with_milp(self, 
#                          current_brick_idx: Optional[int],
#                          remaining_sequence: List[int],
#                          bricks_needing_repair: List[Dict],
#                          is_holding_brick: bool) -> List[TaskItem]:
#         """
#         使用 MILP 求解最优任务序列
#         """
        
#         print("\n[QP-MILP] ═══════════════════════════════════════════════")
#         print("[QP-MILP] Building MILP problem...")
        
#         # ========== Step 1: 分析当前状态 ==========
#         deviations = self.check_all_placed_bricks()
#         deviation_map = {d["brick_idx"]: d for d in deviations}
#         repair_set = {d["brick_idx"] for d in bricks_needing_repair}
        
#         # 已正确放置的砖块
#         placed_correctly = set()
#         for d in deviations:
#             if not d["needs_repair"] and not d["is_in_temp"]:
#                 placed_correctly.add(d["brick_idx"])
        
#         print(f"[QP-MILP] Placed correctly: {placed_correctly}")
#         print(f"[QP-MILP] Needs repair: {repair_set}")
#         print(f"[QP-MILP] In temp: {set(self.bricks_in_temp.keys())}")
#         print(f"[QP-MILP] Remaining to place: {remaining_sequence}")
#         print(f"[QP-MILP] Currently holding: {current_brick_idx if is_holding_brick else 'None'}")
        
#         # ========== Step 2: 确定需要处理的任务 ==========
#         tasks_to_schedule = []  # [(brick_idx, task_type, must_use_temp)]
#         scheduled_set = set()  # 用于避免重复: (brick_idx, task_type)
        
#         # 如果正在抓着砖块
#         held_brick = current_brick_idx if is_holding_brick else None
        
#         # 检查抓着的砖块的依赖是否满足
#         held_deps_ok = True
#         if held_brick is not None:
#             ancestors = self.get_all_ancestors(held_brick)
#             ancestors_needing_repair = ancestors & repair_set
#             ancestors_in_temp = ancestors & set(self.bricks_in_temp.keys())
#             held_deps_ok = len(ancestors_needing_repair) == 0 and len(ancestors_in_temp) == 0
            
#             if not held_deps_ok:
#                 tasks_to_schedule.append((held_brick, "TEMP", True))
#                 scheduled_set.add((held_brick, "TEMP"))
#                 tasks_to_schedule.append((held_brick, "RESTORE_HELD", False))
#                 scheduled_set.add((held_brick, "RESTORE_HELD"))
#                 print(f"[QP-MILP] Held brick {held_brick}: deps not OK, TEMP then RESTORE_HELD")
#             else:
#                 tasks_to_schedule.append((held_brick, "PLACE", False))
#                 scheduled_set.add((held_brick, "PLACE"))
#                 print(f"[QP-MILP] Held brick {held_brick}: deps OK, direct PLACE")
        
#         # 需要修复的砖块
#         for brick_idx in repair_set:
#             if brick_idx == held_brick:
#                 continue
            
#             dependents = self.get_all_dependents(brick_idx)
#             blocking = []
#             for d in deviations:
#                 if d["brick_idx"] in dependents and not d["is_in_temp"]:
#                     blocking.append(d["brick_idx"])
            
#             for blocker in blocking:
#                 if (blocker, "TEMP") not in scheduled_set:
#                     tasks_to_schedule.append((blocker, "TEMP", True))
#                     scheduled_set.add((blocker, "TEMP"))
#                     print(f"[QP-MILP] Brick {blocker}: blocking repair of {brick_idx}, TEMP")
            
#             if (brick_idx, "REPAIR") not in scheduled_set:
#                 tasks_to_schedule.append((brick_idx, "REPAIR", False))
#                 scheduled_set.add((brick_idx, "REPAIR"))
#                 print(f"[QP-MILP] Brick {brick_idx}: REPAIR task")
            
#             for blocker in blocking:
#                 if (blocker, "RESTORE") not in scheduled_set:
#                     tasks_to_schedule.append((blocker, "RESTORE", False))
#                     scheduled_set.add((blocker, "RESTORE"))
#                     print(f"[QP-MILP] Brick {blocker}: RESTORE after repair")
        
#         # 【关键修复】临时位置的砖块必须全部恢复，不管有没有后续依赖
#         for temp_brick in self.bricks_in_temp.keys():
#             if temp_brick == held_brick:
#                 continue
#             if (temp_brick, "RESTORE") in scheduled_set:
#                 continue
#             if (temp_brick, "RESTORE_HELD") in scheduled_set:
#                 continue
            
#             # 【修复】无条件添加 RESTORE 任务 - 临时位置的砖块必须恢复到正确位置
#             tasks_to_schedule.append((temp_brick, "RESTORE", False))
#             scheduled_set.add((temp_brick, "RESTORE"))
#             print(f"[QP-MILP] Brick {temp_brick}: RESTORE from temp (MANDATORY)")
        
#         # 正常放置任务
#         for brick_idx in remaining_sequence:
#             if brick_idx == held_brick:
#                 continue
#             if brick_idx in repair_set:
#                 continue
#             if brick_idx in self.bricks_in_temp:
#                 continue
#             if (brick_idx, "NORMAL") not in scheduled_set:
#                 tasks_to_schedule.append((brick_idx, "NORMAL", False))
#                 scheduled_set.add((brick_idx, "NORMAL"))
#                 print(f"[QP-MILP] Brick {brick_idx}: NORMAL place task")
        
#         n_tasks = len(tasks_to_schedule)
        
#         if n_tasks == 0:
#             print("[QP-MILP] No tasks to schedule")
#             return []
        
#         print(f"\n[QP-MILP] Total tasks to schedule: {n_tasks}")
#         for i, (brick_idx, task_type, must_temp) in enumerate(tasks_to_schedule):
#             print(f"     Task {i}: brick={brick_idx}, type={task_type}, must_temp={must_temp}")
        
#         # ========== Step 3: 构建 MILP 问题 ==========
        
#         # 决策变量
#         order = cp.Variable(n_tasks, integer=True)
        
#         # 目标函数: 最小化总执行时间
#         costs = []
#         for i, (brick_idx, task_type, must_temp) in enumerate(tasks_to_schedule):
#             # 第一个任务如果是放下手中砖块，只需要放置成本
#             if i == 0 and held_brick is not None and brick_idx == held_brick and task_type in ["PLACE", "TEMP"]:
#                 base_cost = PLACE_ONLY_COST
#             else:
#                 base_cost = FULL_PICK_PLACE_COST
#             costs.append(base_cost)
        
#         total_cost = sum(costs)
#         objective = cp.Minimize(total_cost)
        
#         # 约束
#         constraints = []
        
#         # 约束 1: 顺序范围
#         constraints.append(order >= 0)
#         constraints.append(order <= n_tasks - 1)
        
#         # 约束 2: AllDifferent (顺序互不相同)
#         for i in range(n_tasks):
#             for j in range(i + 1, n_tasks):
#                 z = cp.Variable(boolean=True)
#                 M = n_tasks
#                 constraints.append(order[i] - order[j] >= 1 - M * z)
#                 constraints.append(order[j] - order[i] >= 1 - M * (1 - z))
        
#         # 约束 3: 如果手持砖块，第一个任务必须是处理它 (PLACE 或 TEMP)
#         if held_brick is not None:
#             for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#                 if brick_idx == held_brick and task_type in ["PLACE", "TEMP"]:
#                     constraints.append(order[i] == 0)
#                     print(f"[QP-MILP] Constraint: Task {i} (held brick {task_type}) must be order=0")
#                     break
        
#         # 构建任务索引映射
#         task_indices = {}  # brick_idx -> [(task_idx, task_type)]
#         for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#             if brick_idx not in task_indices:
#                 task_indices[brick_idx] = []
#             task_indices[brick_idx].append((i, task_type))
        
#         # 约束 4: 放置依赖 (NORMAL, PLACE, REPAIR, RESTORE, RESTORE_HELD 都需要依赖满足)
#         for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#             if task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
#                 deps = self.get_dependencies_for_brick(brick_idx)
#                 for dep in deps:
#                     # 依赖砖块必须已经在正确位置
#                     if dep in placed_correctly:
#                         # 已经正确放置，无需约束
#                         continue
                    
#                     if dep in task_indices:
#                         # 找到依赖砖块的放置任务 (NORMAL, PLACE, REPAIR, RESTORE)
#                         for dep_task_idx, dep_task_type in task_indices[dep]:
#                             if dep_task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
#                                 constraints.append(order[dep_task_idx] <= order[i] - 1)
#                                 print(f"[QP-MILP] Dep constraint: Task {dep_task_idx} ({dep}.{dep_task_type}) "
#                                       f"before Task {i} ({brick_idx}.{task_type})")
        
#         # 约束 5: TEMP 必须在对应 REPAIR 之前，RESTORE 必须在 REPAIR 之后
#         for repair_brick in repair_set:
#             if repair_brick not in task_indices:
#                 continue
            
#             repair_task_idx = None
#             for idx, ttype in task_indices[repair_brick]:
#                 if ttype == "REPAIR":
#                     repair_task_idx = idx
#                     break
            
#             if repair_task_idx is None:
#                 continue
            
#             # 找出阻挡这个修复的砖块
#             dependents = self.get_all_dependents(repair_brick)
#             for blocker in dependents:
#                 if blocker in task_indices:
#                     for idx, ttype in task_indices[blocker]:
#                         if ttype == "TEMP":
#                             constraints.append(order[idx] <= order[repair_task_idx] - 1)
#                             print(f"[QP-MILP] Constraint: TEMP {idx} before REPAIR {repair_task_idx}")
#                         elif ttype == "RESTORE":
#                             constraints.append(order[idx] >= order[repair_task_idx] + 1)
#                             print(f"[QP-MILP] Constraint: RESTORE {idx} after REPAIR {repair_task_idx}")
        
#         # 约束 6: RESTORE_HELD 必须在所有阻挡它的 REPAIR 完成之后
#         if held_brick is not None and not held_deps_ok:
#             restore_held_idx = None
#             for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#                 if brick_idx == held_brick and task_type == "RESTORE_HELD":
#                     restore_held_idx = i
#                     break
            
#             if restore_held_idx is not None:
#                 ancestors = self.get_all_ancestors(held_brick)
#                 for ancestor in ancestors:
#                     if ancestor in repair_set and ancestor in task_indices:
#                         for idx, ttype in task_indices[ancestor]:
#                             if ttype == "REPAIR":
#                                 constraints.append(order[restore_held_idx] >= order[idx] + 1)
#                                 print(f"[QP-MILP] Constraint: RESTORE_HELD {restore_held_idx} "
#                                       f"after REPAIR {idx} (ancestor {ancestor})")
#                     # 也要在临时位置砖块恢复之后
#                     if ancestor in self.bricks_in_temp and ancestor in task_indices:
#                         for idx, ttype in task_indices[ancestor]:
#                             if ttype == "RESTORE":
#                                 constraints.append(order[restore_held_idx] >= order[idx] + 1)
#                                 print(f"[QP-MILP] Constraint: RESTORE_HELD {restore_held_idx} "
#                                       f"after RESTORE {idx} (ancestor {ancestor} in temp)")

#         # ========== 【新增】约束 7: 依赖临时位置砖块的任务必须在 RESTORE 之后 ==========
#         # 如果砖块 A 被移到临时位置，那么所有依赖 A 的砖块必须在 A 恢复之后才能放置
#         for temp_brick in self.bricks_in_temp.keys():
#             if temp_brick not in task_indices:
#                 continue
            
#             # 找到这个砖块的 RESTORE 任务
#             restore_task_idx = None
#             for idx, ttype in task_indices[temp_brick]:
#                 if ttype in ["RESTORE", "RESTORE_HELD"]:
#                     restore_task_idx = idx
#                     break
            
#             if restore_task_idx is None:
#                 continue
            
#             # 找出所有依赖这个临时砖块的任务
#             for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#                 if task_type in ["NORMAL", "PLACE", "REPAIR", "RESTORE", "RESTORE_HELD"]:
#                     # 检查这个任务的砖块是否依赖临时位置的砖块
#                     deps = self.get_dependencies_for_brick(brick_idx)
#                     if temp_brick in deps:
#                         # 这个任务依赖临时位置的砖块，必须在 RESTORE 之后
#                         if i != restore_task_idx:  # 不是自己
#                             constraints.append(order[i] >= order[restore_task_idx] + 1)
#                             print(f"[QP-MILP] Temp-dep constraint: Task {i} ({brick_idx}.{task_type}) "
#                                   f"must be after RESTORE {restore_task_idx} (temp brick {temp_brick})")
        
#         # ========== 【新增】约束 8: TEMP 任务中，依赖关系也要考虑 ==========
#         # 如果将要被 TEMP 的砖块被其他任务依赖，那些任务必须等 RESTORE 完成
#         for i, (brick_idx, task_type, _) in enumerate(tasks_to_schedule):
#             if task_type == "TEMP":
#                 # 找到对应的 RESTORE 任务
#                 restore_task_idx = None
#                 for idx, ttype in task_indices.get(brick_idx, []):
#                     if ttype in ["RESTORE", "RESTORE_HELD"]:
#                         restore_task_idx = idx
#                         break
                
#                 if restore_task_idx is None:
#                     continue
                
#                 # 找出所有依赖这个砖块的其他任务
#                 for j, (other_brick, other_type, _) in enumerate(tasks_to_schedule):
#                     if other_type in ["NORMAL", "PLACE"]:
#                         deps = self.get_dependencies_for_brick(other_brick)
#                         if brick_idx in deps:
#                             # 这个任务依赖将被 TEMP 的砖块
#                             constraints.append(order[j] >= order[restore_task_idx] + 1)
#                             print(f"[QP-MILP] Future-temp-dep: Task {j} ({other_brick}.{other_type}) "
#                                   f"must wait for RESTORE {restore_task_idx} of brick {brick_idx}")
                                    
#         # ========== Step 4: 求解 ==========
#         print(f"\n[QP-MILP] Solving MILP with {len(constraints)} constraints...")
        
#         prob = cp.Problem(objective, constraints)
        
#         solvers_to_try = [cp.GLPK_MI, cp.CBC, cp.SCIP, cp.ECOS_BB]
#         solved = False
        
#         for solver in solvers_to_try:
#             try:
#                 prob.solve(solver=solver, verbose=False)
#                 if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
#                     solved = True
#                     print(f"[QP-MILP] Solved with {solver}! Status: {prob.status}")
#                     break
#             except Exception as e:
#                 print(f"[QP-MILP] Solver {solver} failed: {e}")
#                 continue
        
#         if not solved:
#             raise RuntimeError(f"[QP-MILP] Failed to solve! Status: {prob.status}")
        
#         # ========== Step 5: 构建任务序列 ==========
#         # 【修复】对 order 值取整
#         order_values = [int(round(v)) for v in order.value]
#         print(f"\n[QP-MILP] Solution order: {order_values}")
        
#         # 按顺序排列任务
#         sorted_indices = sorted(range(n_tasks), key=lambda i: order_values[i])
        
#         task_sequence = []
        
# # 在 _solve_with_milp 方法的 Step 5 中，修改 REPAIR/RESTORE 部分

#         for task_idx in sorted_indices:
#             brick_idx, task_type, must_temp = tasks_to_schedule[task_idx]
#             goal = self.env.compute_goal_pose_from_layout(brick_idx)
#             level = self._get_brick_level(brick_idx)
            
#             if task_type == "TEMP":
#                 # 放到临时位置
#                 temp_pos = self.get_temp_position_for_brick(brick_idx)
#                 is_held_first = (brick_idx == held_brick and order_values[task_idx] == 0)
#                 task_sequence.append(TaskItem(
#                     task_type=TaskType.TEMP_PLACE,
#                     brick_idx=brick_idx,
#                     brick_id=self.env.brick_ids[brick_idx],
#                     target_pos=temp_pos,
#                     target_orn=(0.0, 0.0, 0.0),
#                     level=level,
#                     priority=len(task_sequence),
#                     reason=f"MILP: temp placement",
#                     is_temp=True,
#                     estimated_cost=PLACE_ONLY_COST if is_held_first else FULL_PICK_PLACE_COST
#                 ))
                
#             elif task_type in ["NORMAL", "PLACE"]:
#                 # 正常放置
#                 is_held_first = (brick_idx == held_brick and order_values[task_idx] == 0)
#                 task_sequence.append(TaskItem(
#                     task_type=TaskType.NORMAL_PLACE,
#                     brick_idx=brick_idx,
#                     brick_id=self.env.brick_ids[brick_idx],
#                     target_pos=goal[:3],
#                     target_orn=goal[3:],
#                     level=level,
#                     priority=len(task_sequence),
#                     reason=f"MILP: normal placement",
#                     estimated_cost=PLACE_ONLY_COST if is_held_first else FULL_PICK_PLACE_COST
#                 ))
                
#             elif task_type in ["REPAIR", "RESTORE", "RESTORE_HELD"]:
#                 # 【关键修复】区分 REPAIR 和 RESTORE
#                 if task_type == "REPAIR":
#                     # REPAIR: 砖块在原位但偏移了，使用 deviation_map 中的期望位置
#                     if brick_idx in deviation_map:
#                         d = deviation_map[brick_idx]
#                         # 【修复】检查 deviation_map 中的位置是否是临时位置
#                         # 如果是临时位置，应该使用 layout 中的正确位置
#                         if d.get("is_in_temp", False):
#                             target_pos = goal[:3]
#                             target_orn = goal[3:]
#                             reason = f"MILP: repair (from temp)"
#                         else:
#                             target_pos = tuple(d["expected_pos"])
#                             target_orn = d["expected_orn"]
#                             reason = f"MILP: repair (dev={d['deviation']*1000:.1f}mm)"
#                     else:
#                         target_pos = goal[:3]
#                         target_orn = goal[3:]
#                         reason = f"MILP: repair"
#                 else:
#                     # RESTORE / RESTORE_HELD: 砖块在临时位置，需要恢复到正确位置
#                     # 【关键】始终使用 layout 中定义的正确目标位置
#                     target_pos = goal[:3]
#                     target_orn = goal[3:]
#                     reason = f"MILP: {task_type.lower()} (to correct position)"
                
#                 task_sequence.append(TaskItem(
#                     task_type=TaskType.REPAIR_PLACE,
#                     brick_idx=brick_idx,
#                     brick_id=self.env.brick_ids[brick_idx],
#                     target_pos=target_pos,
#                     target_orn=target_orn,
#                     level=level,
#                     priority=len(task_sequence),
#                     reason=reason,
#                     estimated_cost=FULL_PICK_PLACE_COST
#                 ))
        
#         print(f"\n[QP-MILP] Generated {len(task_sequence)} tasks")
#         print("[QP-MILP] ═══════════════════════════════════════════════\n")
        
#         return task_sequence
#     # ================== 主入口 ==================
    
#     def plan_task_sequence(self, 
#                           current_brick_idx: int,
#                           remaining_sequence: List[int],
#                           is_holding_brick: bool = False) -> List[TaskItem]:
#         """
#         规划任务序列（主入口）
        
#         使用 MILP 优化求解最优任务序列，最小化执行时间
#         """
#         print(f"\n[QP] ═══════════════════════════════════════════════════")
#         print(f"[QP] Planning task sequence with MILP optimization...")
#         print(f"[QP] Current brick: {current_brick_idx}")
#         print(f"[QP] Remaining sequence: {remaining_sequence}")
#         print(f"[QP] Is holding brick: {is_holding_brick}")
#         print(f"[QP] Bricks in temp: {list(self.bricks_in_temp.keys())}")
        
#         # 获取需要修复的砖块
#         bricks_needing_repair = self.get_bricks_needing_repair()
        
#         # 打印当前状态
#         deviations = self.check_all_placed_bricks()
#         print(f"[QP] Checking {len(deviations)} placed bricks:")
#         for d in deviations:
#             if d["is_in_temp"]:
#                 status = "📦 IN TEMP"
#             elif d["needs_repair"]:
#                 status = "⚠️ NEED REPAIR"
#             else:
#                 status = "✓ OK"
#             print(f"     Brick {d['brick_idx']}: deviation={d['deviation']*1000:.2f}mm {status}")
        
#         # 使用 MILP 求解
#         task_sequence = self._solve_with_milp(
#             current_brick_idx, remaining_sequence,
#             bricks_needing_repair, is_holding_brick
#         )
        
#         # 计算总成本
#         total_cost = sum(t.estimated_cost for t in task_sequence)
        
#         # 打印结果
#         print(f"\n[QP] Planned {len(task_sequence)} tasks (est. time: {total_cost:.1f}s):")
#         for i, task in enumerate(task_sequence):
#             temp_marker = " [TEMP]" if task.is_temp else ""
#             print(f"     [{i}] {task.task_type.value}: brick={task.brick_idx}, "
#                   f"cost={task.estimated_cost:.1f}s{temp_marker}")
#             print(f"         reason: {task.reason}")
#         print(f"[QP] ═══════════════════════════════════════════════════\n")
        
#         return task_sequence

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