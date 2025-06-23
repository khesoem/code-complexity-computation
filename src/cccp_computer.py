import ast
import textwrap
from pathlib import Path

class PlanTree:
    def __init__(self, node: ast.AST, depth: int, line: int, child_plans: list['PlanNode']):
        self.node = node
        self.depth = depth
        self.line = line
        self.child_plans = child_plans

def expression_depth(node: ast.expr) -> (int, set): # returns depth and a list of top binary operators
    if isinstance(node, ast.Constant):
        return 0, set()

    elif isinstance(node, ast.BinOp):
        left_depth, left_ops = expression_depth(node.left)
        right_depth, right_ops = expression_depth(node.right)
        child_ops = left_ops.union(right_ops)
        all_ops = child_ops.union({type(node.op).__name__})
        return max(left_depth, right_depth) + (1 if len(child_ops) != len(all_ops) else 0), all_ops

    elif isinstance(node, ast.Compare):
        left_depth, left_ops = expression_depth(node.left)
        right_depth, right_ops = expression_depth(node.comparators)
        child_ops = left_ops.union(right_ops)
        all_ops = child_ops.union({type(node.ops).__name__})
        return max(left_depth, right_depth) + (1 if len(child_ops) != len(all_ops) else 0), all_ops

    elif isinstance(node, ast.UnaryOp):
        return expression_depth(node.operand)[0] + 1, set()

    elif isinstance(node, ast.Call):
        if not node.args:
            return 1, set()
        return max(expression_depth(arg)[0] for arg in node.args) + 1, set()

    elif isinstance(node, ast.Name):
        return 1, set()

    elif isinstance(node, list):
        return 1, set()

    elif isinstance(node, ast.List)or isinstance(node, ast.Tuple):
        return max(expression_depth(elt)[0] for elt in node.elts) + 1, set()

    elif isinstance(node, ast.Subscript):
        return max(expression_depth(node.value)[0], expression_depth(node.slice)[0]) + 1, set()

    else:
        return 0, set()

def create_plan_tree(node: ast.AST, branch_depth: int) -> int:
    if isinstance(node, ast.Module):
        children_plans = [create_plan_tree(child, branch_depth) for child in ast.iter_child_nodes(node)]
        return PlanTree(node, 0, 0, children_plans)

    if isinstance(node, ast.Assign):
        depth = expression_depth(node.value)[0] + 1 + branch_depth
        return PlanTree(node, depth, node.lineno, [])

    if isinstance(node, ast.AugAssign):
        depth = expression_depth(node.value)[0] + 2 + branch_depth
        return PlanTree(node, depth, node.lineno, [])

    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if not node.value.args:
            return PlanTree(node, branch_depth + 1, node.lineno, [])
        return PlanTree(node, max(expression_depth(arg)[0] for arg in node.value.args) + branch_depth + 1, node.lineno, [])

    if isinstance(node, ast.For):
        child_plans = []

        iter_depth = expression_depth(node.iter)[0]
        for_depth = branch_depth + iter_depth + 1

        for child in node.body:
            child_plans.append(create_plan_tree(child, for_depth))

        return PlanTree(node, for_depth, node.lineno, child_plans)

    if isinstance(node, ast.While):
        child_plans = []

        test_depth = expression_depth(node.test)[0]
        while_depth = branch_depth + test_depth + 1

        for child in node.body:
            child_plans.append(create_plan_tree(child, while_depth))

        return PlanTree(node, while_depth, node.lineno, child_plans)

    if isinstance(node, ast.If):
        child_plans = []

        test_depth = expression_depth(node.test)[0]
        if_depth = branch_depth + test_depth + 1

        for child in node.body:
            child_plans.append(create_plan_tree(child, if_depth))

        for orelse in node.orelse:
            child_plans.append(create_plan_tree(orelse, if_depth))

        return PlanTree(node, if_depth, node.lineno, child_plans)

    if (isinstance(node, ast.Break) or isinstance(node, ast.Continue) or isinstance(node, ast.Pass)
            or isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)):
        return PlanTree(node, branch_depth + 1, node.lineno, [])

    raise NotImplementedError(f"Node type {type(node).__name__} is not supported")

def depth_from_plan_tree(plan_tree: PlanTree) -> int:
    if not plan_tree.child_plans:
        return plan_tree.depth
    return max(depth_from_plan_tree(child) for child in plan_tree.child_plans)

def depth_from_source(source: str) -> int:
    tree = ast.parse(textwrap.dedent(source))
    plan_tree = create_plan_tree(tree, 0)
    return depth_from_plan_tree(plan_tree)

def depth_from_file(path: Path) -> int:
    return depth_from_source(path.read_text(encoding="utf‑8"))

def compute_cccp(p):
    print(f'CCCP for {p} is {depth_from_file(p)}')
