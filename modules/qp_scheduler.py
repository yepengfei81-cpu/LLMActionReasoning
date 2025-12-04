"""
最小化 QP 任务调度器
功能: 检测已放置砖块的偏移，决定是否中断当前任务去修复
"""

import numpy as np
import pybullet as p
from typing import List, Dict

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: cvxpy not installed, using simple heuristic fallback")


class QPTaskScheduler:
    """基于 QP 的动态任务调度器"""
    
    def __init__(self, env, threshold_low=0.005, threshold_critical=0.015):
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
        self.w_dependency = 2.0     # 依赖关系权重
        self.w_interrupt = 0.5      # 中断成本权重
        
    def check_brick_deviation(self, brick_id: int, expected_pos: np.ndarray) -> float:
        """检查单个砖块的位置偏移"""
        # 直接用 PyBullet 获取位置，避免依赖 env.get_brick_state 的返回格式
        current_pos, _ = p.getBasePositionAndOrientation(brick_id)
        current_pos = np.array(current_pos)
        deviation = np.linalg.norm(current_pos[:2] - expected_pos[:2])  # 只看 XY 平面
        return deviation
    
    def check_all_placed_bricks(self, placed_bricks: List[Dict]) -> List[Dict]:
        """
        检查所有已放置砖块的状态
        
        Args:
            placed_bricks: [{"brick_id": int, "expected_pos": [x,y,z], "level": int}, ...]
        
        Returns:
            偏移信息列表: [{"brick_id": int, "deviation": float, "level": int}, ...]
        """
        deviations = []
        for brick_info in placed_bricks:
            brick_id = brick_info["brick_id"]
            expected_pos = np.array(brick_info["expected_pos"])
            deviation = self.check_brick_deviation(brick_id, expected_pos)
            
            deviations.append({
                "brick_id": brick_id,
                "deviation": deviation,
                "expected_pos": expected_pos,
                "level": brick_info.get("level", 0),
                "needs_repair": deviation > self.threshold_low
            })
        
        return deviations
    
    def solve(self, 
              current_task_idx: int,
              placed_bricks: List[Dict],
              remaining_bricks: int) -> Dict:
        """
        求解 QP 决定下一步行动
        
        Args:
            current_task_idx: 当前要执行的任务索引
            placed_bricks: 已放置砖块信息
            remaining_bricks: 剩余砖块数
            
        Returns:
            决策: {"action": "CONTINUE" | "REPAIR", "repair_brick_id": int | None, "reason": str}
        """
        # 检查所有已放置砖块
        deviations = self.check_all_placed_bricks(placed_bricks)
        
        # 打印当前状态
        print(f"[QP] Checking {len(deviations)} placed bricks:")
        for d in deviations:
            status = "⚠️ NEED REPAIR" if d["needs_repair"] else "✓ OK"
            print(f"     Brick {d['brick_id']}: deviation={d['deviation']*1000:.2f}mm {status}")
        
        # 筛选需要修复的砖块
        bricks_to_repair = [d for d in deviations if d["needs_repair"]]
        
        if not bricks_to_repair:
            return {
                "action": "CONTINUE",
                "repair_brick_id": None,
                "reason": "All placed bricks within tolerance"
            }
        
        # 检查是否有严重偏移 (必须立即修复)
        critical_bricks = [d for d in bricks_to_repair if d["deviation"] > self.threshold_critical]
        if critical_bricks:
            worst = max(critical_bricks, key=lambda x: x["deviation"])
            return {
                "action": "REPAIR",
                "repair_brick_id": worst["brick_id"],
                "repair_target": worst["expected_pos"],
                "deviation": worst["deviation"],
                "reason": f"Critical deviation: {worst['deviation']*1000:.1f}mm"
            }
        
        # 使用 QP 决策: 继续 vs 修复
        if HAS_CVXPY:
            decision = self._solve_qp(bricks_to_repair, remaining_bricks)
        else:
            decision = self._solve_heuristic(bricks_to_repair, remaining_bricks)
        
        return decision
    
    def _solve_qp(self, bricks_to_repair: List[Dict], remaining_bricks: int) -> Dict:
        """使用 CVXPY 求解 QP"""
        n = len(bricks_to_repair) + 1  # +1 for "CONTINUE" option
        
        # 决策变量: x[0] = 继续, x[1:] = 修复各砖块
        x = cp.Variable(n, nonneg=True)
        
        # 构建成本向量 c
        c = np.zeros(n)
        
        # 继续任务的成本 (考虑剩余任务数)
        c[0] = self.w_interrupt * (1.0 / max(remaining_bricks, 1))
        
        # 修复各砖块的成本 (偏移越大，成本越低 → 越应该修复)
        for i, brick in enumerate(bricks_to_repair):
            # 成本 = 负的偏移量 (偏移大 → 负成本大 → 优先选择)
            # 加入层级权重: 底层砖块更重要
            level_weight = 1.0 / (brick["level"] + 1)
            c[i + 1] = -self.w_deviation * brick["deviation"] * level_weight
        
        # 目标函数
        objective = cp.Minimize(c @ x)
        
        # 约束
        constraints = [
            cp.sum(x) == 1,  # 必须选择一个动作
        ]
        
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
                        "repair_target": brick["expected_pos"],
                        "deviation": brick["deviation"],
                        "reason": f"QP decided: repair (weight={x_val[best_idx]:.3f}, dev={brick['deviation']*1000:.1f}mm)"
                    }
        except Exception as e:
            print(f"QP solve failed: {e}, falling back to heuristic")
        
        return self._solve_heuristic(bricks_to_repair, remaining_bricks)
    
    def _solve_heuristic(self, bricks_to_repair: List[Dict], remaining_bricks: int) -> Dict:
        """简单启发式备选方案"""
        if not bricks_to_repair:
            return {"action": "CONTINUE", "repair_brick_id": None, "reason": "No repair needed"}
        
        # 选择偏移最大的砖块
        worst = max(bricks_to_repair, key=lambda x: x["deviation"])
        
        # 简单阈值判断
        if worst["deviation"] > self.threshold_low * 2:
            return {
                "action": "REPAIR",
                "repair_brick_id": worst["brick_id"],
                "repair_target": worst["expected_pos"],
                "deviation": worst["deviation"],
                "reason": f"Heuristic: repair worst brick (dev={worst['deviation']*1000:.1f}mm)"
            }
        
        return {
            "action": "CONTINUE", 
            "repair_brick_id": None,
            "reason": "Heuristic: deviations acceptable"
        }