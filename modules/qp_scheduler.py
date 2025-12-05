"""
QP 任务调度器 (带依赖约束)
功能: 检测已放置砖块的偏移，决定是否中断当前任务去修复
关键: 确保前置砖块状态正确后才能继续当前任务
"""

import numpy as np
import pybullet as p
from typing import List, Dict, Set, Optional
import cvxpy as cp
HAS_CVXPY = True

class QPTaskScheduler:
    """基于 QP 的动态任务调度器 (带依赖约束)"""
    
    def __init__(self, env, threshold_low=0.015, threshold_critical=0.03):
        """
        Args:
            env: BulletEnv 环境
            threshold_low: 可接受偏移阈值 (米), 低于此值不修复
            threshold_critical: 严重偏移阈值 (米), 高于此值必须修复
        """
        self.env = env
        self.threshold_low = threshold_low
        self.threshold_critical = threshold_critical
        
        # 成本权重 (可调参)
        self.w_deviation = 1.0      # 偏移量权重
        self.w_dependency = 5.0     # 依赖关系权重 (提高权重)
        self.w_interrupt = 0.5      # 中断成本权重
        
        # 砖块依赖关系表 (brick_idx -> [依赖的 brick_idx 列表])
        # 这个需要根据你的具体堆叠结构来定义
        self.dependency_map = self._build_dependency_map()
        
    def _build_dependency_map(self) -> Dict[int, List[int]]:
        """
        构建砖块依赖关系图
        
        返回: {brick_idx: [依赖的 brick_idx 列表]}
        
        根据六砖块三层结构:
          Level 0: 砖块 0, 1 (地面)
          Level 1: 砖块 2 (放在 0,1 上)
          Level 2: 砖块 3, 4 (放在 2 上)
          Level 3: 砖块 5 (放在 3,4 上)
        """
        # 从 env 获取依赖关系，如果没有则使用默认
        if hasattr(self.env, 'get_brick_dependencies'):
            return self.env.get_brick_dependencies()
        
        # 默认依赖关系 (六砖块金字塔结构)
        return {
            0: [],           # 第一层，无依赖
            1: [],           # 第一层，无依赖
            2: [0, 1],       # 第二层，依赖 0 和 1
            3: [2],          # 第三层，依赖 2
            4: [2],          # 第三层，依赖 2
            5: [3, 4],       # 第四层，依赖 3 和 4
        }
    
    def get_dependencies_for_brick(self, brick_idx: int) -> List[int]:
        """获取某砖块的所有前置依赖"""
        return self.dependency_map.get(brick_idx, [])
    
    def get_all_ancestors(self, brick_idx: int) -> Set[int]:
        """递归获取某砖块的所有祖先依赖（包括间接依赖）"""
        ancestors = set()
        direct_deps = self.get_dependencies_for_brick(brick_idx)
        
        for dep in direct_deps:
            ancestors.add(dep)
            ancestors.update(self.get_all_ancestors(dep))
        
        return ancestors
        
    def check_brick_deviation(self, brick_id: int, expected_pos: np.ndarray) -> float:
        """检查单个砖块的位置偏移"""
        current_pos, _ = p.getBasePositionAndOrientation(brick_id)
        current_pos = np.array(current_pos)
        deviation = np.linalg.norm(current_pos[:2] - expected_pos[:2])
        return deviation
    
    def check_all_placed_bricks(self, placed_bricks: List[Dict]) -> List[Dict]:
        """检查所有已放置砖块的状态"""
        deviations = []
        for brick_info in placed_bricks:
            brick_id = brick_info["brick_id"]
            expected_pos = np.array(brick_info["expected_pos"])
            deviation = self.check_brick_deviation(brick_id, expected_pos)
            
            deviations.append({
                "brick_id": brick_id,
                "brick_idx": brick_info.get("brick_idx", None),
                "deviation": deviation,
                "expected_pos": expected_pos,
                "level": brick_info.get("level", 0),
                "needs_repair": deviation > self.threshold_low
            })
        
        return deviations
    
    def check_dependencies_satisfied(self, 
                                     current_brick_idx: int, 
                                     placed_bricks: List[Dict],
                                     deviations: List[Dict]) -> Dict:
        required_deps = self.get_dependencies_for_brick(current_brick_idx)
        
        # 构建 brick_idx -> deviation 的映射
        placed_idx_set = set()
        idx_to_deviation = {}
        for i, brick_info in enumerate(placed_bricks):
            brick_idx = brick_info.get("brick_idx")
            if brick_idx is not None:
                placed_idx_set.add(brick_idx)
                idx_to_deviation[brick_idx] = deviations[i]
        
        failed_deps = []
        missing_deps = []
        
        for dep_idx in required_deps:
            if dep_idx not in placed_idx_set:
                # 依赖砖块还没放置
                missing_deps.append(dep_idx)
            else:
                # 检查依赖砖块的偏移
                dep_deviation = idx_to_deviation.get(dep_idx)
                if dep_deviation and dep_deviation["needs_repair"]:
                    failed_deps.append(dep_idx)
        
        return {
            "satisfied": len(failed_deps) == 0 and len(missing_deps) == 0,
            "failed_dependencies": failed_deps,
            "missing_dependencies": missing_deps
        }
    
    def solve(self, 
              current_task_idx: int,
              current_brick_idx: int,
              placed_bricks: List[Dict],
              remaining_bricks: int) -> Dict:
        """
        求解决策：考虑依赖约束
        
        Args:
            current_task_idx: 当前任务在序列中的索引
            current_brick_idx: 当前要放置的砖块索引
            placed_bricks: 已放置砖块信息 [{"brick_id", "brick_idx", "expected_pos", "level"}, ...]
            remaining_bricks: 剩余砖块数
            
        Returns:
            决策结果
        """
        # 检查所有已放置砖块的偏移
        deviations = self.check_all_placed_bricks(placed_bricks)
        
        # 打印当前状态
        print(f"\n[QP] ═══════════════════════════════════════════════════")
        print(f"[QP] Checking {len(deviations)} placed bricks:")
        for d in deviations:
            status = "⚠️ NEED REPAIR" if d["needs_repair"] else "✓ OK"
            print(f"     Brick {d['brick_id']} (idx={d.get('brick_idx')}): "
                  f"deviation={d['deviation']*1000:.2f}mm {status}")
        
        # 检查当前砖块的依赖是否满足
        dep_check = self.check_dependencies_satisfied(
            current_brick_idx, placed_bricks, deviations
        )
        
        print(f"[QP] Current brick idx: {current_brick_idx}")
        print(f"[QP] Dependencies: {self.get_dependencies_for_brick(current_brick_idx)}")
        print(f"[QP] Dependency satisfied: {dep_check['satisfied']}")
        
        # ========== 核心逻辑: 依赖约束优先 ==========
        
        # 1. 如果有依赖砖块偏移过大，必须先修复
        if dep_check["failed_dependencies"]:
            # 按层级排序，先修复底层
            failed_deps = dep_check["failed_dependencies"]
            
            # 找到这些依赖砖块的偏移信息
            deps_to_repair = []
            for d in deviations:
                if d.get("brick_idx") in failed_deps:
                    deps_to_repair.append(d)
            
            if deps_to_repair:
                # 选择层级最低的先修复（底层优先）
                deps_to_repair.sort(key=lambda x: x["level"])
                target = deps_to_repair[0]
                
                print(f"[QP] ⚠️ DEPENDENCY CONSTRAINT: Must repair brick idx "
                      f"{target.get('brick_idx')} first!")
                
                return {
                    "action": "REPAIR_DEPENDENCY",
                    "repair_brick_id": target["brick_id"],
                    "repair_brick_idx": target.get("brick_idx"),
                    "repair_target": target["expected_pos"],
                    "deviation": target["deviation"],
                    "reason": f"Dependency constraint: brick {target.get('brick_idx')} "
                              f"must be fixed (dev={target['deviation']*1000:.1f}mm)"
                }
        
        # 2. 如果有缺失的依赖，返回错误（这种情况不应该发生）
        if dep_check["missing_dependencies"]:
            print(f"[QP] ❌ ERROR: Missing dependencies: {dep_check['missing_dependencies']}")
            return {
                "action": "ERROR",
                "reason": f"Missing dependencies: {dep_check['missing_dependencies']}"
            }
        
        # 3. 依赖满足后，检查是否有其他砖块需要修复
        bricks_to_repair = [d for d in deviations if d["needs_repair"]]
        
        if not bricks_to_repair:
            print(f"[QP] ✓ All OK, continue with current task")
            return {
                "action": "CONTINUE",
                "repair_brick_id": None,
                "reason": "All placed bricks within tolerance, dependencies satisfied"
            }
        
        # 4. 有非依赖砖块需要修复，使用 QP 决策
        # 过滤掉已经作为依赖处理过的
        non_dep_repairs = [d for d in bricks_to_repair 
                          if d.get("brick_idx") not in dep_check["failed_dependencies"]]
        
        if not non_dep_repairs:
            return {
                "action": "CONTINUE",
                "repair_brick_id": None,
                "reason": "Dependencies handled, continue"
            }
        
        # 检查是否有严重偏移
        critical_bricks = [d for d in non_dep_repairs 
                          if d["deviation"] > self.threshold_critical]
        if critical_bricks:
            worst = max(critical_bricks, key=lambda x: x["deviation"])
            return {
                "action": "REPAIR",
                "repair_brick_id": worst["brick_id"],
                "repair_brick_idx": worst.get("brick_idx"),
                "repair_target": worst["expected_pos"],
                "deviation": worst["deviation"],
                "reason": f"Critical deviation: {worst['deviation']*1000:.1f}mm"
            }
        
        # 使用 QP 决策
        if HAS_CVXPY:
            decision = self._solve_qp(non_dep_repairs, remaining_bricks)
        else:
            decision = self._solve_heuristic(non_dep_repairs, remaining_bricks)
        
        return decision
    
    def _solve_qp(self, bricks_to_repair: List[Dict], remaining_bricks: int) -> Dict:
        """
        使用 CVXPY 求解 QP
        
        目标函数: min c^T x
        约束: sum(x) = 1, x >= 0
        
        其中:
            x[0] = 继续当前任务的权重
            x[i] = 修复砖块 i 的权重
        """
        n = len(bricks_to_repair) + 1
        x = cp.Variable(n, nonneg=True)
        
        # 构建成本向量
        c = np.zeros(n)
        
        # 继续任务的成本
        c[0] = self.w_interrupt * (1.0 / max(remaining_bricks, 1))
        
        # 修复各砖块的成本（负值表示收益）
        for i, brick in enumerate(bricks_to_repair):
            level_weight = 1.0 / (brick["level"] + 1)
            c[i + 1] = -self.w_deviation * brick["deviation"] * level_weight
        
        # 目标函数
        objective = cp.Minimize(c @ x)
        
        # 约束
        constraints = [cp.sum(x) == 1]
        
        # 求解
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                x_val = x.value
                best_idx = np.argmax(x_val)
                
                if best_idx == 0:
                    return {
                        "action": "CONTINUE",
                        "repair_brick_id": None,
                        "reason": f"QP decided: continue (weight={x_val[0]:.3f})"
                    }
                else:
                    brick = bricks_to_repair[best_idx - 1]
                    return {
                        "action": "REPAIR",
                        "repair_brick_id": brick["brick_id"],
                        "repair_brick_idx": brick.get("brick_idx"),
                        "repair_target": brick["expected_pos"],
                        "deviation": brick["deviation"],
                        "reason": f"QP decided: repair (weight={x_val[best_idx]:.3f}, "
                                  f"dev={brick['deviation']*1000:.1f}mm)"
                    }
        except Exception as e:
            print(f"QP solve failed: {e}, falling back to heuristic")
        
        return self._solve_heuristic(bricks_to_repair, remaining_bricks)
    
    def _solve_heuristic(self, bricks_to_repair: List[Dict], remaining_bricks: int) -> Dict:
        """简单启发式备选方案"""
        if not bricks_to_repair:
            return {"action": "CONTINUE", "repair_brick_id": None, "reason": "No repair needed"}
        
        # 按层级排序（底层优先），同层按偏移量排序
        sorted_bricks = sorted(bricks_to_repair, 
                               key=lambda x: (x["level"], -x["deviation"]))
        worst = sorted_bricks[0]
        
        if worst["deviation"] > self.threshold_low * 1.5:
            return {
                "action": "REPAIR",
                "repair_brick_id": worst["brick_id"],
                "repair_brick_idx": worst.get("brick_idx"),
                "repair_target": worst["expected_pos"],
                "deviation": worst["deviation"],
                "reason": f"Heuristic: repair brick (level={worst['level']}, "
                          f"dev={worst['deviation']*1000:.1f}mm)"
            }
        
        return {
            "action": "CONTINUE", 
            "repair_brick_id": None,
            "reason": "Heuristic: deviations acceptable"
        }
    
    def get_repair_order(self, bricks_to_repair: List[Dict]) -> List[Dict]:
        # 按层级排序
        return sorted(bricks_to_repair, key=lambda x: (x["level"], -x["deviation"]))