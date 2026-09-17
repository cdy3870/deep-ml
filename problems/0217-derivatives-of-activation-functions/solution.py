import math

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    """
    sigmoid_x = 1 / (1 + math.exp(-x))
    sigmoid_deriv = sigmoid_x * (1 - sigmoid_x)

    tanh_x = math.tanh(x)  # equivalent to (e^x - e^-x) / (e^x + e^-x)
    tanh_deriv = 1 - tanh_x ** 2

    relu_deriv = 1.0 if x > 0 else 0.0

    return {"sigmoid": sigmoid_deriv, "tanh": tanh_deriv, "relu": relu_deriv}