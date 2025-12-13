from __future__ import annotations

from typing import TYPE_CHECKING
from abc import abstractmethod

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        """Apply the function to the given inputs and create a computational graph node."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)

    @staticmethod
    @abstractmethod
    def forward(ctx: Context, *args: float) -> float:
        """Computes the forward pass of the function.

        Args:
            ctx (Context): Context object to save information for backward computation.
            *args (float): Input values.

        Returns:
            float: The result of the forward computation.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Forward method not implemented.")

    @staticmethod
    @abstractmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass (derivative) of the function.

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Derivative of the output with respect to some scalar.

        Returns:
            Tuple[float, ...]: The gradients with respect to each input.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Backward method not implemented.")


# Examples


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for addition: f(a, b) = a + b.

        Args:
            ctx (Context): Context object (not used for this function).
            a (float): First input value.
            b (float): Second input value.

        Returns:
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for addition using the chain rule.

        Given the derivative of the output (d_output), computes the gradients
        with respect to inputs a and b:
        - ∂f/∂a = 1 (derivative of (a + b) with respect to a)
        - ∂f/∂b = 1 (derivative of (a + b) with respect to b)

        Args:
            ctx (Context): Context object (not used for this function).
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing:
                - Gradient with respect to a (d_output * 1)
                - Gradient with respect to b (d_output * 1)

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for natural logarithm: f(a) = log(a).

        Saves the input value for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): Input value (must be positive).

        Returns:
            float: The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for natural logarithm using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = 1/a (derivative of log(a) with respect to a)

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output * 1/a)

        """
        (a,) = ctx.saved_tensors
        return (operators.log_back(a, d_output),)


### To implement for Task 1.2 and 1.4 ###
# Look at the above classes for examples on how to implement the forward and backward functions
# Use the operators.py file from Module 0


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for multiplication: f(a, b) = a * b.

        Saves the input values for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): First input value.
            b (float): Second input value.

        Returns:
            float: The product of a and b.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for multiplication using the chain rule.

        Given the derivative of the output (d_output), computes the gradients
        with respect to inputs a and b:
        - ∂f/∂a = b (derivative with respect to a)
        - ∂f/∂b = a (derivative with respect to b)

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, float]: A tuple containing:
                - Gradient with respect to a (d_output * b)
                - Gradient with respect to b (d_output * a)

        """
        (
            a,
            b,
        ) = ctx.saved_tensors
        return (d_output * b, d_output * a)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for inverse: f(a) = 1/a.

        Saves the input value for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): Input value.

        Returns:
            float: The inverse of a (1/a).

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for inverse using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = -1/a² (derivative of 1/a with respect to a)

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output * (-1/a²))

        """
        (a,) = ctx.saved_tensors
        return (d_output * (operators.neg(operators.inv(a) * operators.inv(a))),)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for negation: f(a) = -a.

        Args:
            ctx (Context): Context object (not used for this function).
            a (float): Input value.

        Returns:
            float: The negation of a.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for negation using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = -1 (derivative of -a with respect to a)

        Args:
            ctx (Context): Context object (not used for this function).
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output * -1)

        """
        return (operators.neg(d_output),)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for sigmoid: f(a) = 1 / (1 + e^(-a)).

        Saves the input value for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): Input value.

        Returns:
            float: The sigmoid of a.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for sigmoid using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = σ(a) * (1 - σ(a)) where σ is the sigmoid function

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output * σ(a) * (1 - σ(a)))

        """
        (a,) = ctx.saved_tensors
        return (d_output * operators.sigmoid(a) * (1.0 - operators.sigmoid(a)),)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for ReLU: f(a) = max(0, a).

        Saves the input value for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): Input value.

        Returns:
            float: The ReLU of a (a if a > 0, else 0).

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for ReLU using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = 1 if a > 0, else 0 (derivative is undefined at a = 0, we use 0)

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output if a > 0, else 0)

        """
        (a,) = ctx.saved_tensors
        if a < 0.0:
            return (0.0,)
        elif a > 0.0:
            return (d_output,)
        else:
            return (0.0,)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass for exponential: f(a) = e^a.

        Saves the input value for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): Input value.

        Returns:
            float: The exponential of a (e^a).

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass for exponential using the chain rule.

        Given the derivative of the output (d_output), computes the gradient
        with respect to input a:
        - ∂f/∂a = e^a (derivative of e^a with respect to a)

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, ...]: A tuple containing the gradient with respect to a
                (d_output * e^a)

        """
        (a,) = ctx.saved_tensors
        return (d_output * operators.exp(a),)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for less-than comparison: f(a, b) = 1.0 if a < b else 0.0.

        Saves the input values for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): First input value.
            b (float): Second input value.

        Returns:
            float: 1.0 if a < b, otherwise 0.0.

        """
        ctx.save_for_backward(a, b)
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for less-than comparison.

        The less-than function is a step function with zero gradient everywhere
        except at the discontinuity (a = b), where it is undefined. We return
        zero gradients for both inputs.

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, float]: A tuple of (0.0, 0.0) representing gradients
                with respect to a and b.

        """
        (
            a,
            b,
        ) = ctx.saved_tensors
        if a == b:
            raise ValueError("a and b are equal, therefore the gradient is undefined")
        else:
            return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass for equality comparison: f(a, b) = 1.0 if a == b else 0.0.

        Saves the input values for use in the backward pass.

        Args:
            ctx (Context): Context object to save values for backward computation.
            a (float): First input value.
            b (float): Second input value.

        Returns:
            float: 1.0 if a == b, otherwise 0.0.

        """
        ctx.save_for_backward(a, b)
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Computes the backward pass for equality comparison.

        The equality function is a step function with zero gradient everywhere
        except at the discontinuity (a = b), where it is undefined. We return
        zero gradients for both inputs.

        Args:
            ctx (Context): Context object containing saved values from forward pass.
            d_output (float): Gradient of the loss with respect to the output.

        Returns:
            Tuple[float, float]: A tuple of (0.0, 0.0) representing gradients
                with respect to a and b.

        """
        (
            a,
            b,
        ) = ctx.saved_tensors
        if a == b:
            raise ValueError("a and b are equal, therefore the gradient is undefined")
        else:
            return (0.0, 0.0)
