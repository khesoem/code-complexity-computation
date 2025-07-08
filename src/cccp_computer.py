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

    elif isinstance(node, ast.BoolOp):
        if not node.values:
            return 0, set()

        child_depths = [expression_depth(value) for value in node.values]
        max_depth = max(depth[0] for depth in child_depths)

        child_ops = set()
        for depth, ops in child_depths:
            child_ops = child_ops.union(ops)

        all_ops = child_ops.union(set(type(node.op).__name__))
        return max_depth + (1 if len(child_ops) != len(all_ops) else 0), all_ops

    elif isinstance(node, ast.Call):
        if not node.args:
            return 1, set()
        return max(expression_depth(arg)[0] for arg in node.args) + 1, set()

    elif isinstance(node, ast.Name):
        return 1, set()

    elif isinstance(node, list):
        if not node:
            return 1, set()
        return max(expression_depth(elt)[0] for elt in node), set()

    elif isinstance(node, ast.List)or isinstance(node, ast.Tuple):
        return max(expression_depth(elt)[0] for elt in node.elts) + 1, set()

    elif isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Subscript): # reading the (i+1)th dimension of an array is different from and child of reading the i-th dimension
            return expression_depth(node.value)[0] + expression_depth(node.slice)[0] + 1, set()
        else:
            return max(expression_depth(node.value)[0], expression_depth(node.slice)[0]) + 1, set()

    else:
        return 0, set()

def get_expression_identifiers(node: ast.expr, exclude_store_identifiers: bool=False) -> set:
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

    elif isinstance(node, ast.BoolOp):
        identifiers = set()
        for value in node.values:
            identifiers.update(get_expression_identifiers(value))
        return identifiers

    elif isinstance(node, ast.Call):
        identifiers = {node.func.id} if isinstance(node.func, ast.Name) else set()
        for arg in node.args:
            identifiers.update(get_expression_identifiers(arg))
        return identifiers

    elif isinstance(node, ast.Name):
        if exclude_store_identifiers and isinstance(node.ctx, ast.Store):
            # If the node is a Name in a tuple context, we do not count it as an identifier
            return set()
        return {node.id}

    elif isinstance(node, list):
        identifiers = set()
        for elt in node:
            identifiers.update(get_expression_identifiers(elt))
        return identifiers

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
        left_depth = max([expression_depth(target)[0] for target in node.targets])
        assignment_depth = max(exp_depth + 1, left_depth) # if left is just an identifier, its complexity is already counted in the assignment operation (+1)
        plan_depth = assignment_depth + parent_plan_depth
        exp_identifiers = get_expression_identifiers(node.value)
        loaded_identifiers = exp_identifiers
        for target in node.targets:
            loaded_identifiers = loaded_identifiers.union(get_expression_identifiers(target, exclude_store_identifiers=True))

        """mpi is sum of branch depth, plan depth of the expression, and number of identifiers minus one as the one identifier is already counted in the plan depth"""
        mpi = len(loaded_identifiers) + assignment_depth + branch_depth - 1
        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi, [])

    if isinstance(node, ast.AugAssign):
        exp_depth = expression_depth(node.value)[0]
        left_depth = expression_depth(node.target)[0]
        assignment_depth = max(exp_depth + 1, left_depth) + 1
        plan_depth = assignment_depth + parent_plan_depth
        exp_identifiers = get_expression_identifiers(node.value)
        loaded_identifiers = (exp_identifiers.union(get_expression_identifiers(node.target, exclude_store_identifiers=False)))
        mpi = len(loaded_identifiers) + assignment_depth + branch_depth - (1 if exp_depth > 0 else 0)

        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi,[])

    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if not node.value.args:
            mpi = branch_depth + 1
            return CCTree(node, parent_plan_depth + 1, node.lineno, branch_depth, mpi, [])

        exp_depth = max(expression_depth(arg)[0] for arg in node.value.args)
        plan_depth = exp_depth + 1 + parent_plan_depth
        exp_identifiers = get_expression_identifiers(node.value)
        loaded_identifiers = (exp_identifiers.union({node.value.func.id if isinstance(node.value.func, ast.Name) else set()}))
        mpi = len(loaded_identifiers) + exp_depth + branch_depth + (1 if len(loaded_identifiers) == 0 else 0)

        return CCTree(node, plan_depth, node.lineno, branch_depth, mpi, [])

    if isinstance(node, ast.For):
        subtrees = []

        iter_depth = expression_depth(node.iter)[0]
        for_depth = parent_plan_depth + iter_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, for_depth, branch_depth + 1))

        loaded_identifiers = get_expression_identifiers(node.iter)
        mpi = len(loaded_identifiers) + iter_depth + branch_depth + (1 if len(loaded_identifiers) == 0 else 0)
        return CCTree(node, for_depth, node.lineno, branch_depth, mpi, subtrees)

    if isinstance(node, ast.While):
        subtrees = []

        test_depth = expression_depth(node.test)[0]
        while_depth = parent_plan_depth + test_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, while_depth, branch_depth + 1))

        loaded_identifiers = get_expression_identifiers(node.test)
        mpi = len(loaded_identifiers) + test_depth + branch_depth + (1 if len(loaded_identifiers) == 0 else 0)

        return CCTree(node, while_depth, node.lineno, branch_depth, mpi, subtrees)

    if isinstance(node, ast.If):
        subtrees = []

        test_depth = expression_depth(node.test)[0]
        if_depth = parent_plan_depth + test_depth + 1

        for child in node.body:
            subtrees.append(create_cctree(child, if_depth, branch_depth + 1))

        for orelse in node.orelse:
            subtrees.append(create_cctree(orelse, if_depth, branch_depth + 1))

        loaded_identifiers = get_expression_identifiers(node.test)
        mpi = len(loaded_identifiers) + test_depth + branch_depth + (1 if len(loaded_identifiers) == 0 else 0)
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


