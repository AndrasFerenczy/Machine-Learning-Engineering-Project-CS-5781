from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.
    Uses the central difference formula: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant for finite difference approximation

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_forward = list(vals)
    vals_backward = list(vals)
    vals_forward[arg] += epsilon
    vals_backward[arg] -= epsilon
    return (f(*vals_forward) - f(*vals_backward)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative for this variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Unique identifier for this variable."""
        pass

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf node."""
        pass

    def is_constant(self) -> bool:
        """Check if this variable is a constant."""
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return parent variables in the computation graph."""
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply chain rule to compute derivatives for parent variables."""
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    Hints:
        - Use depth-first search (DFS) to visit nodes
        - Track visited nodes to avoid cycles (use node.unique_id)
        - Return nodes in reverse order (dependencies first)

    """
    order = []
    visited = set()

    def visit(node: Variable) -> None:
        if node.unique_id in visited:
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            visit(parent)
        order.append(node)

    visit(variable)
    return order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:x
        - First get all nodes in topological order using topological_sort()
        - Create a dictionary to store derivatives for each node (keyed by unique_id)
        - Initialize the starting node's derivative to the input deriv
        - Process nodes in the topological order (which is already correct for backprop)
        - For leaf nodes: call node.accumulate_derivative(derivative)
        - For non-leaf nodes: call node.chain_rule(derivative) to get parent derivatives
        - Sum derivatives when the same parent appears multiple times

    """
    derivatives = {variable.unique_id: deriv}

    for node in topological_sort(variable):
        node_deriv = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(node_deriv)
        else:
            for parent, parent_deriv in node.chain_rule(node_deriv):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += parent_deriv
                else:
                    derivatives[parent.unique_id] = parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return saved tensors for backward pass."""
        return self.saved_values
