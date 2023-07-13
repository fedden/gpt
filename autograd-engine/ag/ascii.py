"""ASCII rendering of the computation graph."""
from typing import List

import yaml

from ag.tensor import Tensor
from ag.scalar import Scalar


def render_as_tree(node):
    """Render the tree as a nested tree of lists."""
    if isinstance(node, Tensor):
        assert node.size == 1, f"Expected a scalar, got {node.size}"
        node = node.data[0]
    assert isinstance(node, Scalar), f"Expected a scalar, got {type(node)}"
    tree = _build_dag(node)
    print(yaml.safe_dump(tree, sort_keys=False, indent=4))


def _build_dag(node):
    if node.is_leaf_node():
        return {
            "node": repr(node),
        }
    else:
        elements: List[str] = [f"{node.data:.4f}", f"grad={node.grad.data:.4f}"]
        if node.name is not None:
            elements.append(f"name='{node.name}'")
        body: str = ", ".join(elements)
        return {
            "node": f"{node._op_type.__name__}({body})",
            "inputs": [
                _build_dag(node) for node in node._child_nodes
            ],
        }
