"""ASCII rendering of the computation graph."""
from typing import List

import yaml


def render_as_tree(node):
    """Render the tree as a nested tree of lists."""
    tree = _build_dag(node)
    print(yaml.safe_dump(tree, sort_keys=False, indent=4))


def _build_dag(node):
    if node.is_leaf_node():
        return {
            "node": repr(node),
        }
    else:
        elements: List[str] = [f"{node.data:.4f}", f"grad={node.grad:.4f}"]
        if node.name is not None:
            elements.append(f"name='{node.name}'")
        body: str = ", ".join(elements)
        return {
            "node": f"{node._op_type.__name__}({body})",
            "inputs": [
                _build_dag(node) for node in node._child_nodes
            ],
        }
