"""2D Classification Datasets for MiniTorch Visualization

This module provides various 2D point classification datasets used for testing
and visualizing machine learning models in MiniTorch.

PYRIGHT STYLE REQUIREMENTS:
To pass the type checking tests, you need to:

1. ADD TYPE ANNOTATIONS to all function parameters and return values
   Example: def make_pts(N: int) -> List[Tuple[float, float]]:

2. ADD DOCSTRINGS to all functions
   Use the triple-quote format with a brief description of what the function does

3.ENSURE ALL IMPORTS are properly typed
   The required imports are already provided at the top
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random 2D points with coordinates in [0, 1].

    Args:
        N (int): Number of points to generate.

    Returns:
        List[Tuple[float, float]]: List of tuples representing the points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """Data class to store a graph representation with points and their labels.

    Attributes:
        N (int): Number of points.
        X (List[Tuple[float, float]]): List of 2D points.
        y (List[int]): List of labels associated with points.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Create a Graph where the label is 1 if the x-coordinate is less than 0.5, else 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph object with N points and binary labels based on a simple threshold.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Create a Graph where the label is 1 if the sum of x and y coordinates is less than 0.5, else 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph object with N points and binary labels based on the diagonal rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Create a Graph where the label is 1 if the x-coordinate is less than 0.2 or greater than 0.8, else 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph object with N points and binary labels based on split regions in the x dimension.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Create a Graph where the label is 1 if exclusively one coordinate is on opposite sides of 0.5, else 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph object with N points and labels based on XOR condition on coordinates.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Create a Graph where the label is 1 if the point lies outside a radius of sqrt(0.1) from center (0.5,0.5), else 0.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A Graph object with N points and labels based on distance from the center in circular fashion.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Create a Graph representing two spirals labeled 0 and 1.

    The function generates points along two interleaved spirals centered around (0.5, 0.5).

    Args:
        N (int): Number of points to generate. Should be even.

    Returns:
        Graph: A Graph object with N points on two spirals, half labeled 0, half labeled 1.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
