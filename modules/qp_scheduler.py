"""
QP 任务调度器 (带依赖约束)
功能: 检测已放置砖块的偏移，动态调整任务序列
关键: 输出调整后的任务队列，而非直接执行修复
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


class TaskType(Enum):
    """任务类型枚举"""
    NORMAL_PLACE = "normal_place"      # 正常放置新砖块
    REPAIR_PLACE = "repair_place"      # 修复已放置的砖块
    TEMP_PLACE = "temp_place"          # 临时放置（移开碍事的砖块）


@dataclass
class TaskItem:
    """任务项"""
    task_type: TaskType
    brick_idx: int              # 砖块索引
    brick_id: int               # 砖块 PyBullet ID
    target_pos: Tuple[float, float, float]  # 目标位置 (x, y, z)
    target_orn: Tuple[float, float, float]  # 目标姿态 (roll, pitch, yaw)
    level: int                  # 层级
    priority: int = 0           # 优先级 (数字越小优先级越高)
    reason: str = ""            # 任务原因说明
    is_temp: bool = False       # 是否为临时位置
    
    def to_goal_pose(self) -> Tuple[float, float, float, float, float, float]:
        """转换为 goal_pose 格式"""
        return (*self.target_pos, *self.target_orn)


class QPTaskScheduler:
    """基于 QP 的动态任务调度器 (输出任务序列)"""
    
    def __init__(self, env, threshold_low=0.015, threshold_critical=0.03):
        self.env = env
        self.threshold_low = threshold_low
        self.threshold_critical = threshold_critical
        
        self.w_deviation = 1.0
        self.w_dependency = 5.0
        self.w_interrupt = 0.5
        
        self.dependency_map = self._build_dependency_map()
        self.task_queue: List[TaskItem] = []
        self.placed_bricks_info: List[Dict] = []
        self.temp_positions = self._generate_temp_positions()
        self.used_temp_positions: Set[int] = set()
        self.bricks_in_temp: Dict[int, Tuple[float, float, float]] = {}
        
    def _build_dependency_map(self) -> Dict[int, List[int]]:
        if hasattr(self.env, 'get_brick_dependencies'):
            dep_map = self.env.get_brick_dependencies()
            
            print(f"\n[QP] ═══════════════════════════════════════════════════")
            print(f"[QP] Brick Dependency Map (from env.get_brick_dependencies()):")
            for brick_idx in sorted(dep_map.keys()):
                deps = dep_map[brick_idx]
                if deps:
                    print(f"     Brick {brick_idx} depends on: {deps}")
                else:
                    print(f"     Brick {brick_idx} depends on: [] (base layer)")
            
            print(f"\n[QP] Reverse Dependencies (who depends on whom):")
            for brick_idx in sorted(dep_map.keys()):
                dependents = [idx for idx, deps in dep_map.items() if brick_idx in deps]
                if dependents:
                    print(f"     Brick {brick_idx} is needed by: {dependents}")
            print(f"[QP] ═══════════════════════════════════════════════════\n")
            
            return dep_map
        
        print("[QP] WARNING: Using default dependency map!")
        return {
            0: [], 1: [],
            2: [0, 1],
            3: [2], 4: [2],
            5: [3, 4],
        }
    
    def _generate_temp_positions(self) -> List[Tuple[float, float, float]]:
        """生成临时放置位置列表（保留作为后备）"""
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
        
        # 临时位置偏移量（绝对值）
        self.temp_offset_distance = L + 0.2  # 偏移距离
        
        # 计算堆叠区域中心，用于决定偏移方向
        if not hasattr(self.env, 'layout_targets'):
            self.env._parse_layout()
        
        layout_targets = self.env.layout_targets
        
        if not layout_targets:
            print("[QP] Warning: No layout_targets found, using default temp positions")
            self.stack_center_x = 0.0
            return [
                (-0.3, 0.4, self.temp_z),
                (-0.3, 0.5, self.temp_z),
                (-0.3, 0.6, self.temp_z),
            ]
        
        xs = [t['xy'][0] for t in layout_targets]
        ys = [t['xy'][1] for t in layout_targets]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 保存堆叠区域中心
        self.stack_center_x = (min_x + max_x) / 2
        self.stack_center_y = (min_y + max_y) / 2
        
        print(f"[QP] Temp position config:")
        print(f"     Offset distance: {self.temp_offset_distance:.3f}m")
        print(f"     Stack center: ({self.stack_center_x:.3f}, {self.stack_center_y:.3f})")
        print(f"     Stack region: X=[{min_x:.3f}, {max_x:.3f}], Y=[{min_y:.3f}, {max_y:.3f}]")
        
        # 生成后备位置（在堆叠区域两侧）
        fallback_positions = []
        for i in range(len(layout_targets)):
            # 交替放在左右两侧
            if i % 2 == 0:
                tx = min_x - self.temp_offset_distance - (i // 2) * (L + 0.05)
            else:
                tx = max_x + self.temp_offset_distance + (i // 2) * (L + 0.05)
            ty = (min_y + max_y) / 2
            fallback_positions.append((tx, ty, self.temp_z))
        
        return fallback_positions
        
    def get_temp_position_for_brick(self, brick_idx: int) -> Tuple[float, float, float]:
        """
        根据砖块的期望位置计算临时位置
        临时位置 = 期望位置 + offset（向远离堆叠中心的方向偏移）
        """
        # 获取砖块的期望位置
        if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
            target = self.env.layout_targets[brick_idx]
            expected_x, expected_y = target['xy']
            
            # 根据期望位置相对于堆叠中心的位置，决定偏移方向
            # 如果期望位置在中心右边（x > center），向右偏移（+offset）
            # 如果期望位置在中心左边（x < center），向左偏移（-offset）
            if expected_x >= self.stack_center_x:
                temp_x = expected_x + self.temp_offset_distance
                direction = "right (+X)"
            else:
                temp_x = expected_x - self.temp_offset_distance
                direction = "left (-X)"
            
            temp_y = expected_y  # Y 方向保持不变
            temp_z = self.temp_z
            
            # 检查是否与其他临时位置冲突
            conflict_resolved = False
            for other_idx, other_pos in self.bricks_in_temp.items():
                if other_idx != brick_idx:
                    dist = np.sqrt((temp_x - other_pos[0])**2 + (temp_y - other_pos[1])**2)
                    if dist < self.brick_L * 0.8:  # 太近，需要额外偏移
                        # 继续向同一方向偏移
                        if expected_x >= self.stack_center_x:
                            temp_x += self.brick_L + 0.05
                        else:
                            temp_x -= self.brick_L + 0.05
                        conflict_resolved = True
                        print(f"[QP] Temp position conflict with brick {other_idx}, adding extra offset")
            
            print(f"[QP] Temp position for brick {brick_idx}:")
            print(f"     Expected pos: ({expected_x:.4f}, {expected_y:.4f})")
            print(f"     Stack center X: {self.stack_center_x:.4f}")
            print(f"     Offset direction: {direction}")
            print(f"     Temp pos: ({temp_x:.4f}, {temp_y:.4f}, {temp_z:.4f})")
            
            return (temp_x, temp_y, temp_z)
        
        # 如果无法获取期望位置，使用后备位置
        print(f"[QP] WARNING: Cannot get expected position for brick {brick_idx}, using fallback")
        return self.get_temp_position()
    
    def get_temp_position(self) -> Tuple[float, float, float]:
        """获取一个可用的后备临时位置"""
        for i, pos in enumerate(self.temp_positions):
            if i not in self.used_temp_positions:
                self.used_temp_positions.add(i)
                return pos
        
        # 所有后备位置都用完了，生成新的
        offset = len(self.used_temp_positions) * 0.15
        return (-0.4 - offset, 0.0, self.temp_z)
    
    def release_temp_position(self, pos: Tuple[float, float, float]):
        for i, temp_pos in enumerate(self.temp_positions):
            if np.allclose(pos, temp_pos, atol=0.01):
                self.used_temp_positions.discard(i)
                break
    
    def mark_brick_in_temp(self, brick_idx: int, temp_pos: Tuple[float, float, float]):
        self.bricks_in_temp[brick_idx] = temp_pos
        print(f"[QP] Marked brick {brick_idx} in temp position: {temp_pos}")
    
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
        """递归获取某砖块的所有祖先依赖（包括间接依赖）"""
        ancestors = set()
        direct_deps = self.get_dependencies_for_brick(brick_idx)
        for dep in direct_deps:
            ancestors.add(dep)
            ancestors.update(self.get_all_ancestors(dep))
        return ancestors
    
    def get_all_dependents(self, brick_idx: int) -> Set[int]:
        """获取依赖于某砖块的所有砖块（递归获取所有后代）"""
        dependents = set()
        for idx, deps in self.dependency_map.items():
            if brick_idx in deps:
                dependents.add(idx)
                dependents.update(self.get_all_dependents(idx))
        return dependents
    
    def check_brick_deviation(self, brick_id: int, expected_pos: np.ndarray) -> float:
        current_pos, _ = p.getBasePositionAndOrientation(brick_id)
        current_pos = np.array(current_pos)
        deviation = np.linalg.norm(current_pos[:2] - expected_pos[:2])
        return deviation
    
    def check_all_placed_bricks(self) -> List[Dict]:
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
    
    def _get_bricks_blocking_repair(self, brick_to_repair_idx: int, 
                                    all_placed_deviations: List[Dict]) -> List[Dict]:
        """获取阻挡修复的砖块列表（在上面的砖块）"""
        dependents = self.get_all_dependents(brick_to_repair_idx)
        
        blocking_bricks = []
        for d in all_placed_deviations:
            if d["brick_idx"] in dependents and not d["is_in_temp"]:
                blocking_bricks.append(d)
        
        # 按层级从高到低排序（先移走顶层）
        blocking_bricks.sort(key=lambda x: -x["level"])
        
        return blocking_bricks
    
    def _get_all_repairs_needed_for(self, target_brick_idx: int, 
                                     bricks_to_repair: List[Dict]) -> List[Dict]:
        """
        获取为了放置 target_brick_idx，需要修复的所有砖块
        包括：直接依赖需要修复的 + 间接依赖需要修复的
        按依赖顺序排序（底层先修复）
        """
        all_ancestors = self.get_all_ancestors(target_brick_idx)
        
        # 找出所有需要修复的祖先
        repairs_needed = []
        for d in bricks_to_repair:
            if d["brick_idx"] in all_ancestors:
                repairs_needed.append(d)
        
        # 按层级排序（底层先修复）
        repairs_needed.sort(key=lambda x: x["level"])
        
        return repairs_needed
    
    def _get_bricks_in_temp_needed_for(self, brick_idx: int) -> List[int]:
        """获取放置 brick_idx 之前需要先恢复的临时位置砖块"""
        needed = []
        all_ancestors = self.get_all_ancestors(brick_idx)
        
        for temp_brick_idx in self.bricks_in_temp:
            if temp_brick_idx in all_ancestors:
                needed.append(temp_brick_idx)
        
        needed.sort(key=lambda x: self._get_brick_level(x))
        return needed
    
    def _plan_repair_sequence(self, brick_to_repair: Dict, 
                               all_deviations: List[Dict],
                               bricks_to_repair: List[Dict],
                               scheduled_indices: Set[int]) -> List[TaskItem]:
        """
        规划修复单个砖块的完整任务序列
        包括：移走上面的砖块 -> 修复依赖 -> 修复目标 -> 放回上面的砖块
        """
        tasks = []
        brick_idx = brick_to_repair["brick_idx"]
        
        print(f"[QP] Planning repair for brick {brick_idx}...")
        
        # Step 1: 找出阻挡的砖块（在这个砖块上面的）
        blocking = self._get_bricks_blocking_repair(brick_idx, all_deviations)
        print(f"[QP]   Blocking bricks (on top): {[b['brick_idx'] for b in blocking]}")
        
        # Step 2: 找出这个砖块的依赖中需要修复的
        deps_needing_repair = []
        ancestors = self.get_all_ancestors(brick_idx)
        for d in bricks_to_repair:
            if d["brick_idx"] in ancestors and d["brick_idx"] != brick_idx:
                deps_needing_repair.append(d)
        deps_needing_repair.sort(key=lambda x: x["level"])
        print(f"[QP]   Dependencies needing repair: {[d['brick_idx'] for d in deps_needing_repair]}")
        
        # Step 3: 移走阻挡的砖块（使用基于期望位置的临时位置）
        for blocker in blocking:
            if blocker["brick_idx"] not in scheduled_indices:
                # 根据砖块的期望位置计算临时位置
                temp_pos = self.get_temp_position_for_brick(blocker["brick_idx"])
                tasks.append(TaskItem(
                    task_type=TaskType.TEMP_PLACE,
                    brick_idx=blocker["brick_idx"],
                    brick_id=blocker["brick_id"],
                    target_pos=temp_pos,
                    target_orn=(0.0, 0.0, 0.0),
                    level=blocker["level"],
                    priority=0,
                    reason=f"Move to temp (blocking repair of brick {brick_idx})",
                    is_temp=True
                ))
                scheduled_indices.add(blocker["brick_idx"])
        
        # Step 4: 递归修复依赖（从底层开始）
        for dep in deps_needing_repair:
            if dep["brick_idx"] not in scheduled_indices:
                dep_tasks = self._plan_repair_sequence(dep, all_deviations, bricks_to_repair, scheduled_indices)
                tasks.extend(dep_tasks)
        
        # Step 5: 修复当前砖块
        if brick_idx not in scheduled_indices:
            tasks.append(TaskItem(
                task_type=TaskType.REPAIR_PLACE,
                brick_idx=brick_idx,
                brick_id=brick_to_repair["brick_id"],
                target_pos=tuple(brick_to_repair["expected_pos"]),
                target_orn=brick_to_repair["expected_orn"],
                level=brick_to_repair["level"],
                priority=1,
                reason=f"Repair (deviation={brick_to_repair['deviation']*1000:.1f}mm)"
            ))
            scheduled_indices.add(brick_idx)
        
        # Step 6: 放回阻挡的砖块（从底层开始）
        blocking.sort(key=lambda x: x["level"])
        for blocker in blocking:
            blocker_goal = self.env.compute_goal_pose_from_layout(blocker["brick_idx"])
            # 检查是否已经安排了恢复任务
            already_scheduled_restore = any(
                t.brick_idx == blocker["brick_idx"] and t.task_type == TaskType.REPAIR_PLACE 
                for t in tasks
            )
            if not already_scheduled_restore:
                tasks.append(TaskItem(
                    task_type=TaskType.REPAIR_PLACE,
                    brick_idx=blocker["brick_idx"],
                    brick_id=blocker["brick_id"],
                    target_pos=blocker_goal[:3],
                    target_orn=blocker_goal[3:],
                    level=blocker["level"],
                    priority=2,
                    reason=f"Restore after repair of brick {brick_idx}"
                ))
        
        return tasks

    
    def plan_task_sequence(self, 
                          current_brick_idx: int,
                          remaining_sequence: List[int],
                          is_holding_brick: bool = False) -> List[TaskItem]:
        """
        规划任务序列
        
        核心逻辑：
        1. 检查所有需要修复的砖块
        2. 对于要放置的每个砖块，检查其所有祖先依赖是否需要修复
        3. 按正确的顺序修复（底层先修复，移走阻挡的砖块）
        """
        print(f"\n[QP] ═══════════════════════════════════════════════════")
        print(f"[QP] Planning task sequence...")
        print(f"[QP] Current brick: {current_brick_idx}")
        print(f"[QP] Remaining sequence: {remaining_sequence}")
        print(f"[QP] Is holding brick: {is_holding_brick}")
        print(f"[QP] Bricks in temp positions: {list(self.bricks_in_temp.keys())}")
        
        deviations = self.check_all_placed_bricks()
        bricks_to_repair = [d for d in deviations if d["needs_repair"]]
        
        print(f"[QP] Checking {len(deviations)} placed bricks:")
        for d in deviations:
            if d["is_in_temp"]:
                status = "📦 IN TEMP"
            elif d["needs_repair"]:
                status = "⚠️ NEED REPAIR"
            else:
                status = "✓ OK"
            print(f"     Brick idx={d['brick_idx']}: deviation={d['deviation']*1000:.2f}mm {status}")
        
        task_sequence = []
        scheduled_indices = set()
        
        # ========== 场景 1: 正在抓着砖块 ==========
        if is_holding_brick and current_brick_idx is not None:
            current_goal = self.env.compute_goal_pose_from_layout(current_brick_idx)
            current_level = self._get_brick_level(current_brick_idx)
            
            # 获取所有祖先依赖
            all_ancestors = self.get_all_ancestors(current_brick_idx)
            
            # 检查哪些祖先需要修复
            ancestors_needing_repair = [d for d in bricks_to_repair if d["brick_idx"] in all_ancestors]
            
            # 检查依赖是否在临时位置
            deps_in_temp = [idx for idx in all_ancestors if self.is_brick_in_temp(idx)]
            
            print(f"[QP] Brick {current_brick_idx} ancestors: {all_ancestors}")
            print(f"[QP] Ancestors needing repair: {[d['brick_idx'] for d in ancestors_needing_repair]}")
            print(f"[QP] Ancestors in temp: {deps_in_temp}")
            
            if ancestors_needing_repair or deps_in_temp:
                print(f"[QP] ⚠️ Ancestors need attention, placing current brick to temp first")
                
                # Step 1: 把当前砖块放到临时位置（基于期望位置计算）
                temp_pos = self.get_temp_position_for_brick(current_brick_idx)
                task_sequence.append(TaskItem(
                    task_type=TaskType.TEMP_PLACE,
                    brick_idx=current_brick_idx,
                    brick_id=self.env.brick_ids[current_brick_idx],
                    target_pos=temp_pos,
                    target_orn=(0.0, 0.0, 0.0),
                    level=current_level,
                    priority=0,
                    reason="Move to temp (ancestors need attention)",
                    is_temp=True
                ))
                scheduled_indices.add(current_brick_idx)
                
                # Step 2: 按层级从低到高修复祖先
                ancestors_needing_repair.sort(key=lambda x: x["level"])
                for ancestor in ancestors_needing_repair:
                    repair_tasks = self._plan_repair_sequence(
                        ancestor, deviations, bricks_to_repair, scheduled_indices
                    )
                    task_sequence.extend(repair_tasks)
                
                # Step 3: 恢复在临时位置的依赖砖块
                deps_in_temp.sort(key=lambda x: self._get_brick_level(x))
                for dep_idx in deps_in_temp:
                    if dep_idx not in scheduled_indices:
                        dep_goal = self.env.compute_goal_pose_from_layout(dep_idx)
                        task_sequence.append(TaskItem(
                            task_type=TaskType.REPAIR_PLACE,
                            brick_idx=dep_idx,
                            brick_id=self.env.brick_ids[dep_idx],
                            target_pos=dep_goal[:3],
                            target_orn=dep_goal[3:],
                            level=self._get_brick_level(dep_idx),
                            priority=4,
                            reason="Restore from temp (needed as ancestor)"
                        ))
                        scheduled_indices.add(dep_idx)
                
                # Step 4: 把当前砖块放回正确位置
                task_sequence.append(TaskItem(
                    task_type=TaskType.REPAIR_PLACE,
                    brick_idx=current_brick_idx,
                    brick_id=self.env.brick_ids[current_brick_idx],
                    target_pos=current_goal[:3],
                    target_orn=current_goal[3:],
                    level=current_level,
                    priority=5,
                    reason="Restore from temp after ancestors fixed"
                ))
                
            else:
                # 祖先都正常，正常放置
                task_sequence.append(TaskItem(
                    task_type=TaskType.NORMAL_PLACE,
                    brick_idx=current_brick_idx,
                    brick_id=self.env.brick_ids[current_brick_idx],
                    target_pos=current_goal[:3],
                    target_orn=current_goal[3:],
                    level=current_level,
                    priority=0,
                    reason="Normal placement"
                ))
                scheduled_indices.add(current_brick_idx)
        
        # ========== 场景 2: 没有抓着砖块 ==========
        else:
            # 处理当前砖块
            if current_brick_idx is not None:
                all_ancestors = self.get_all_ancestors(current_brick_idx)
                ancestors_needing_repair = [d for d in bricks_to_repair if d["brick_idx"] in all_ancestors]
                
                if ancestors_needing_repair:
                    print(f"[QP] ⚠️ Must repair ancestors first: {[d['brick_idx'] for d in ancestors_needing_repair]}")
                    
                    # 按层级从低到高修复
                    ancestors_needing_repair.sort(key=lambda x: x["level"])
                    for ancestor in ancestors_needing_repair:
                        if ancestor["brick_idx"] not in scheduled_indices:
                            repair_tasks = self._plan_repair_sequence(
                                ancestor, deviations, bricks_to_repair, scheduled_indices
                            )
                            task_sequence.extend(repair_tasks)
        
        # ========== 添加剩余的正常任务（带依赖检查）==========
        for brick_idx in remaining_sequence:
            if brick_idx in scheduled_indices:
                continue
            if brick_idx in self.bricks_in_temp:
                continue
            
            # 检查这个砖块的所有祖先是否需要修复或在临时位置
            all_ancestors = self.get_all_ancestors(brick_idx)
            
            # 先处理需要修复的祖先
            ancestors_needing_repair = [d for d in bricks_to_repair 
                                        if d["brick_idx"] in all_ancestors 
                                        and d["brick_idx"] not in scheduled_indices]
            ancestors_needing_repair.sort(key=lambda x: x["level"])
            
            for ancestor in ancestors_needing_repair:
                repair_tasks = self._plan_repair_sequence(
                    ancestor, deviations, bricks_to_repair, scheduled_indices
                )
                task_sequence.extend(repair_tasks)
            
            # 再处理在临时位置的祖先
            ancestors_in_temp = [idx for idx in all_ancestors 
                                if self.is_brick_in_temp(idx) 
                                and idx not in scheduled_indices]
            ancestors_in_temp.sort(key=lambda x: self._get_brick_level(x))
            
            for temp_idx in ancestors_in_temp:
                temp_goal = self.env.compute_goal_pose_from_layout(temp_idx)
                task_sequence.append(TaskItem(
                    task_type=TaskType.REPAIR_PLACE,
                    brick_idx=temp_idx,
                    brick_id=self.env.brick_ids[temp_idx],
                    target_pos=temp_goal[:3],
                    target_orn=temp_goal[3:],
                    level=self._get_brick_level(temp_idx),
                    priority=8,
                    reason=f"Restore from temp (needed for brick {brick_idx})"
                ))
                scheduled_indices.add(temp_idx)
            
            # 添加正常放置任务
            goal = self.env.compute_goal_pose_from_layout(brick_idx)
            level = self._get_brick_level(brick_idx)
            task_sequence.append(TaskItem(
                task_type=TaskType.NORMAL_PLACE,
                brick_idx=brick_idx,
                brick_id=self.env.brick_ids[brick_idx],
                target_pos=goal[:3],
                target_orn=goal[3:],
                level=level,
                priority=10 + brick_idx,
                reason="Normal sequence"
            ))
            scheduled_indices.add(brick_idx)
        
        # ========== 最后检查：临时位置的砖块 ==========
        for temp_brick_idx in list(self.bricks_in_temp.keys()):
            if temp_brick_idx not in scheduled_indices:
                # 检查是否有后续任务需要它
                has_dependents = any(
                    temp_brick_idx in self.get_all_ancestors(idx) 
                    for idx in remaining_sequence
                )
                
                if has_dependents:
                    temp_brick_goal = self.env.compute_goal_pose_from_layout(temp_brick_idx)
                    task_sequence.append(TaskItem(
                        task_type=TaskType.REPAIR_PLACE,
                        brick_idx=temp_brick_idx,
                        brick_id=self.env.brick_ids[temp_brick_idx],
                        target_pos=temp_brick_goal[:3],
                        target_orn=temp_brick_goal[3:],
                        level=self._get_brick_level(temp_brick_idx),
                        priority=9,
                        reason="Restore from temp (has dependents)"
                    ))
                    scheduled_indices.add(temp_brick_idx)
        
        # 打印最终任务序列
        print(f"\n[QP] Planned task sequence ({len(task_sequence)} tasks):")
        for i, task in enumerate(task_sequence):
            temp_marker = " [TEMP]" if task.is_temp else ""
            print(f"     [{i}] {task.task_type.value}: brick_idx={task.brick_idx}, "
                  f"level={task.level}{temp_marker}, reason={task.reason}")
        print(f"[QP] ═══════════════════════════════════════════════════\n")
        
        return task_sequence
    
    def _get_brick_level(self, brick_idx: int) -> int:
        if hasattr(self.env, 'layout_targets') and brick_idx < len(self.env.layout_targets):
            return self.env.layout_targets[brick_idx]["level"]
        return 0
    
    def should_replan(self) -> bool:
        bricks_to_repair = self.get_bricks_needing_repair()
        return len(bricks_to_repair) > 0