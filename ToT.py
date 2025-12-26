from typing import List, Optional, Any, Callable, Set
from dataclasses import dataclass
import itertools

"""思维树节点 - 存储当前状态、路径及评估分数"""
@dataclass
class ThoughtNode:
    state: List[float]      # 当前剩余数字
    path: str = ""          # 推导路径
    score: float = 0.0      # 评估分数

"""Tree of Thoughts 核心搜索框架"""
class TreeOfThoughts:
    def __init__(self, thought_generator: Callable, state_evaluator: Callable, goal_checker: Callable):
        self.generate_thoughts = thought_generator
        self.evaluate_state = state_evaluator
        self.is_goal = goal_checker

    """执行广度优先搜索 (BFS)"""
    def search_bfs(self, initial_state: List[int]) -> Optional[ThoughtNode]:
        queue = [ThoughtNode(state=[float(x) for x in initial_state])]
        visited: Set[tuple] = set()

        while queue:
            current_node = queue.pop(0)
            state_key = tuple(sorted(current_node.state))
            if state_key in visited: continue
            visited.add(state_key)

            if self.is_goal(current_node):
                return current_node

            # 扩展节点并剪枝
            next_thoughts = self.generate_thoughts(current_node)
            for thought in next_thoughts:
                thought.score = self.evaluate_state(thought)
                if thought.score > 0:  # 仅保留有潜力的分支
                    queue.append(thought)
        return None

    """执行深度优先搜索 (DFS)"""
    def search_dfs(self, current_node: ThoughtNode, visited=None) -> Optional[ThoughtNode]:
        if visited is None: visited = set()
        
        state_key = tuple(sorted(current_node.state))
        if state_key in visited: return None
        visited.add(state_key)

        if self.is_goal(current_node):
            return current_node

        next_thoughts = self.generate_thoughts(current_node)
        # 启发式排序：优先搜索评分高的思维
        next_thoughts.sort(key=lambda x: self.evaluate_state(x), reverse=True)

        for thought in next_thoughts:
            if self.evaluate_state(thought) > 0:
                result = self.search_dfs(thought, visited)
                if result: return result
        return None

"""24点求解器 - 结合 ToT 框架实现"""
class Point24Solver:
    def __init__(self, strategy="BFS"):
        self.strategy = strategy
        self.tot = TreeOfThoughts(
            self._generate_thoughts,
            self._evaluate_state,
            self._goal_checker
        )

    """思维生成器：尝试两两组合当前数字进行四则运算"""
    def _generate_thoughts(self, node: ThoughtNode) -> List[ThoughtNode]:
        thoughts = []
        nums = node.state
        for i, j in itertools.combinations(range(len(nums)), 2):
            a, b = nums[i], nums[j]
            remain = [nums[k] for k in range(len(nums)) if k != i and k != j]
            
            # 定义可能的运算
            ops = {
                "+": a + b,
                "-": a - b,
                "r-": b - a,
                "*": a * b,
            }
            if abs(b) > 1e-6: ops["/"] = a / b
            if abs(a) > 1e-6: ops["r/"] = b / a

            for op_symbol, val in ops.items():
                op_path = f"({a}{op_symbol}{b}={val})"
                new_path = f"{node.path} -> {op_path}" if node.path else op_path
                thoughts.append(ThoughtNode(state=remain + [val], path=new_path))
        return thoughts

    """评估函数：判断当前数字状态是否有解"""
    def _evaluate_state(self, node: ThoughtNode) -> float:
        if not node.state: return 0.0
        # 目标检查：若已出现24，给满分
        for x in node.state:
            if abs(x - 24) < 1e-6: return 1.0
        # 剪枝逻辑：若中间状态已无法凑成整数或数值极端（示例简化，默认给0.5）
        return 0.5

    """目标检查：判断是否只剩下一个数且为24"""
    def _goal_checker(self, node: ThoughtNode) -> bool:
        return len(node.state) == 1 and abs(node.state[0] - 24) < 1e-6

    """调用 ToT 搜索求解"""
    def solve(self, numbers: List[int]) -> Optional[str]:
        if self.strategy == "BFS":
            result = self.tot.search_bfs(numbers)
        else:
            # DFS 初始节点
            initial_node = ThoughtNode(state=[float(x) for x in numbers])
            result = self.tot.search_dfs(initial_node)
        return result.path if result else None

"""测试用例"""
if __name__ == "__main__":
    solver = Point24Solver(strategy="BFS")
    test_cases = [
        [3, 3, 8, 8],  # 有解 (经典题)
        [1, 1, 1, 1],  # 无解
        [1, 2, 3, 4],  # 有解
    ]
    
    for nums in test_cases:
        result = solver.solve(nums)
        print(f"输入: {nums} -> 路径: {result if result else '无解'}")