import ast
import textwrap
from pathlib import Path

class CCTree:
    """
    A class representing a control flow tree node. The tree contains only certain types of AST nodes.
    Plan_depth is the plan depth of the node, according to the CCCP paper.
    Note that the plan depth for a node inside a for loop is larger than the for loop itself.
    Branch_depth is the depth of the node in the control flow tree, i.e. how many branches are there.
    mpi stands for max plan activity, per CCCP paper definition.
    """
    def __init__(self, node: ast.AST, plan_depth: int, line: int, branch_depth: int, mpi: int, subtrees: list):
        self.node = node
        self.plan_depth = plan_depth
        self.line = line
        self.branch_depth = branch_depth
        self.mpi = mpi
        self.subtrees = subtrees

    def get_depth_plan(self):
        if not self.subtrees:
            return self.plan_depth

        return max(child.get_depth_plan() for child in self.subtrees)

    def get_mpi(self):
        if not self.subtrees:
            return self.mpi

        return max(self.mpi, max(child.get_mpi() for child in self.subtrees))

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

def get_expression_identifiers(node: ast.expr) -> set:
    """
    Returns a set of identifiers used in the expression.
    """
    if isinstance(node, ast.Constant):
        return set()

    elif isinstance(node, ast.BinOp):
        return get_expression_identifiers(node.left).union(get_expression_identifiers(node.right))

    elif isinstance(node, ast.Compare):
        return get_expression_identifiers(node.left).union(get_expression_identifiers(node.comparators))

    elif isinstance(node, ast.UnaryOp):
        return get_expression_identifiers(node.operand)

    elif isinstance(node, ast.Call):
        identifiers = {node.func.id} if isinstance(node.func, ast.Name) else set()
        for arg in node.args:
            identifiers.update(get_expression_identifiers(arg))
        return identifiers

    elif isinstance(node, ast.Name):
        return {node.id}

    elif isinstance(node, list):
        return set()

    elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        identifiers = set()
        for elt in node.elts:
            identifiers.update(get_expression_identifiers(elt))
        return identifiers

    elif isinstance(node, ast.Subscript):
        return get_expression_identifiers(node.value).union(get_expression_identifiers(node.slice))

    else:
        return set()

def create_cctree(node: ast.AST, parent_plan_depth: int, branch_depth: int) -> CCTree:
    if isinstance(node, ast.Module):
        children_plans = [create_cctree(child, parent_plan_depth, branch_depth) for child in ast.iter_child_nodes(node)]
        return CCTree(node, parent_plan_depth, 0, branch_depth, 0, children_plans)

    if isinstance(node, ast.Assign):
        exp_depth = expression_depth(node.value)[0]
        plan_depth = exp_depth + 1 + parent_plan_depth
        identifiers = (get_expression_identifiers(node.value)
                       .union({target.id for target in node.targets if isinstance(target, ast.Name)}))

        """mpi is sum of branch depth, plan depth of the expression, and number of identifiers, 
            minus one if there are identifiers in the expression to avoid double-counting the identifier"""
        mpi = len(identifiers) + exp_depth + branch_depth + (1 if len(identifiers) == 0 else 0)
        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi, [])

    if isinstance(node, ast.AugAssign):
        exp_depth = expression_depth(node.value)[0]
        plan_depth = expression_depth(node.value)[0] + 2 + parent_plan_depth
        identifiers = (get_expression_identifiers(node.value)
                       .union({node.target.id if isinstance(node.target, ast.Name) else set()}))
        mpi = len(identifiers) + exp_depth + branch_depth + 1 + (1 if len(identifiers) == 0 else 0)

        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi,[])

    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if not node.value.args:
            mpi = branch_depth + 1
            return CCTree(node, parent_plan_depth + 1, node.lineno, branch_depth, mpi, [])

        exp_depth = max(expression_depth(arg)[0] for arg in node.value.args)
        plan_depth = exp_depth + 1 + parent_plan_depth
        identifiers = (get_expression_identifiers(node.value)
                       .union({node.value.func.id if isinstance(node.value.func, ast.Name) else set()}))
        mpi = len(identifiers) + exp_depth + branch_depth + (1 if len(identifiers) == 0 else 0)

        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi, [])

    if isinstance(node, ast.For):
        subtrees = []

        iter_depth = expression_depth(node.iter)[0]
        for_depth = parent_plan_depth + iter_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, for_depth, branch_depth + 1))

        identifiers = get_expression_identifiers(node.iter)
        mpi = len(identifiers) + iter_depth + branch_depth + (1 if len(identifiers) == 0 else 0)
        return CCTree(node, for_depth, node.lineno, branch_depth, mpi, subtrees)

    if isinstance(node, ast.While):
        subtrees = []

        test_depth = expression_depth(node.test)[0]
        while_depth = parent_plan_depth + test_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, while_depth, branch_depth + 1))

        identifiers = get_expression_identifiers(node.test)
        mpi = len(identifiers) + test_depth + branch_depth + (1 if len(identifiers) == 0 else 0)

        return CCTree(node, while_depth, node.lineno, branch_depth, mpi, subtrees)

    if isinstance(node, ast.If):
        subtrees = []

        test_depth = expression_depth(node.test)[0]
        if_depth = parent_plan_depth + test_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, if_depth, branch_depth + 1))

        for orelse in node.orelse:
            subtrees.append(create_cctree(orelse, if_depth, branch_depth + 1))

        identifiers = get_expression_identifiers(node.test)
        mpi = len(identifiers) + test_depth + branch_depth + (1 if len(identifiers) == 0 else 0)
        return CCTree(node, if_depth, node.lineno, branch_depth, mpi, subtrees)

    if (isinstance(node, ast.Break) or isinstance(node, ast.Continue) or isinstance(node, ast.Pass)
            or isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)):
        mpi = branch_depth + 1
        return CCTree(node, parent_plan_depth + 1, node.lineno, branch_depth, mpi,[])

    raise NotImplementedError(f"Node type {type(node).__name__} is not supported")

def compute_cccp(p: Path):
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(textwrap.dedent(src))
    cct = create_cctree(tree, 0, 0)
    print(f'CCCP-PD and CCCP-MPI for {p} are {cct.get_depth_plan()} and {cct.get_mpi()}')


