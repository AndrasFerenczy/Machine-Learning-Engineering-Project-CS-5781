"""Collection of the core mathematical operators used throughout the code base.

This module implements fundamental mathematical operations that serve as building blocks
for neural network computations in MiniTorch.

NOTE: The `task0_1` tests will not fully pass until you complete `task0_3`.
Some tests depend on higher-order functions implemented in the later task.
"""

import math
from typing import Callable, Iterable, List
# =============================================================================
# Task 0.1: Mathematical Operators
# =============================================================================


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function that returns its input.

    Args:
        x (float): Input number.

    Returns:
        float: The same value as x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x (float): Input number.

    Returns:
        float: The negative of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if one number is less than another.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x < y, otherwise False.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if x == y, otherwise False.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The greater of x and y.

    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close to each other within a tolerance of 1e-2.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        bool: True if |x - y| < 1e-2, otherwise False.

    """
    return abs(x - y) < pow(10, -2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of x, in range (0,1).

    """
    if x >= 0:
        return 1 / (1 + exp(-x))
    else:
        return exp(x) / (1 + exp(x))


def relu(x: float) -> float:
    """Compute the ReLU activation function.

    Args:
        x (float): Input value.

    Returns:
        float: x if x > 0, otherwise 0.

    """
    return max(0, x)


def log(x: float) -> float:
    """Compute the natural logarithm of a number.

    Args:
        x (float): Positive input value.

    Returns:
        float: Natural log of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential of a number.

    Args:
        x (float): Input value.

    Returns:
        float: e raised to the power of x.

    """
    e = 2.718281828459045
    return pow(e, x)


def inv(x: float) -> float:
    """Compute the reciprocal of a number.

    Args:
        x (float): Non-zero input value.

    Returns:
        float: 1 / x

    """
    return 1 / x


def log_back(x: float, d: float) -> float:
    """Compute the backward derivative of log(x) with respect to its input.

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: Derivative of log(x) multiplied by d.

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Compute the backward derivative of 1/x with respect to its input.

    Args:
        x (float): Input value.
        d (float): Upstream gradient.

    Returns:
        float: Derivative of (1/x) multiplied by d.

    """
    return -d / pow(x, 2)


def relu_back(x: float, d: float) -> float:
    """Compute the backward derivative of ReLU with respect to its input.

    Args:
        x (float): Input value before ReLU.
        d (float): Upstream gradient.

    Returns:
        float: d if x > 0, otherwise 0.

    """
    if x > 0.0:
        return d
    else:
        return 0.0


# =============================================================================
# Task 0.3: Higher-Order Functions
# =============================================================================


def map(fn: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of an iterable of floats.

    Args:
        fn (Callable[[float], float]): A function that takes a float and returns a float.
        iterable (Iterable[float]): An iterable of float values.

    Returns:
        Iterable[float]: A list containing the results of applying fn to each element of iterable.

    """
    ret = []
    for i in iterable:
        ret.append(fn(i))
    return ret

def zipWith(
    fn: Callable[[float, float], float], list1: List[float], list2: List[float]
) -> List[float]:
    """Apply a binary function element-wise to two lists of floats.

    Args:
        fn (Callable[[float, float], float]): Function that takes two floats and returns a float.
        list1 (list): First list of floats.
        list2 (list): Second list of floats.

    Returns:
        List[float]: List of results of applying fn to each pair of elements from list1 and list2.

    """
    ret = []
    for i in range(len(list1)):
        ret.append(fn(list1[i], list2[i]))
    return ret


def reduce(
    fn: Callable[[float, float], float], iterable: Iterable[float], initial_value: float
) -> float:
    """Reduce an iterable of floats to a single float by cumulatively applying a binary function.

    Args:
        fn (Callable[[float, float], float]): Function that takes two floats and returns a float.
        iterable (Iterable[float]): Iterable of float values to be reduced.
        initial_value (float): Initial value to start the reduction.

    Returns:
        float: The final reduced value after applying fn over the iterable.

    """
    a = initial_value
    for i in iterable:
        a = fn(a, i)


    a = initial_value
    for i in range(len(iterable)):
        a = fn(a, iterable[i])
    return a


def negList(lst: List[float]) -> List[float]:
    """Negate each element in a list of floats.

    Args:
        lst (List[float]): List of float values.

    Returns:
        List[float]: List of negated float values.

    """
    return list(map(neg, lst))


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add two lists of floats element-wise.

    Args:
        lst1 (List[float]): First list of floats.
        lst2 (List[float]): Second list of floats.

    Returns:
        List[float]: List of sums from element-wise addition of lst1 and lst2.

    """
    return zipWith(add, lst1, lst2)


def sum(lst1: List[float]) -> float:
    """Compute the sum of elements in a list of floats.

    Args:
        lst1 (List[float]): List of float values.

    Returns:
        float: Sum of all elements in lst1.

    """
    return reduce(add, lst1, 0)


def prod(lst1: List[float]) -> float:
    """Compute the product of elements in a list of floats.

    Args:
        lst1 (List[float]): List of float values.

    Returns:
        float: Product of all elements in lst1.

    """
    return reduce(mul, lst1, 1)
