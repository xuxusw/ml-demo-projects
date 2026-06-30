from __future__ import annotations

import numpy as np
from .layers import Layer


class Activation(Layer):
    def __init__(self):
        self._cache = None

    def parameters(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}()" # объект / можно будет использовать eval


class Linear(Activation):
    """
    f(x) = x
    f'(x) = 1
    """
    
    def forward(self, x):
        self._cache = np.asarray(x)
        return self._cache

    def backward(self, grad):
        return grad


class Sigmoid(Activation):
    """
    f(x) = 1 / (1 + e^{-x})
    f'(x) = f(x) * (1 - f(x))
    """
    
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        out = np.clip(1 / (1 + np.exp(-x)), 1e-12, 1 - 1e-12)
        self._cache = out
        return out

    def backward(self, grad):
        s = self._cache
        return grad * s * (1 - s)


class Tanh(Activation):
    """
    f(x) = tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    f'(x) = 1 - tanh(x)^2
    """
    
    def forward(self, x):
        out = np.tanh(np.asarray(x, dtype=float))
        self._cache = out
        return out

    def backward(self, grad):
        return grad * (1 - self._cache ** 2)


class ReLU(Activation):
    """
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0
    """
    
    def forward(self, x):
        self._cache = np.asarray(x)
        return np.maximum(0, self._cache)

    def backward(self, grad):
        return grad * (self._cache > 0)


class Softmax(Activation):
    """
    f(x_i) = exp(x_i) / sum(exp(x_j))
    grad: ∂f_i/∂x_j = f_i * (δ_{ij} - f_j)
    """
    
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        out = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        self._cache = out
        return out

    def backward(self, grad):
        s = self._cache
        dot = np.sum(grad * s, axis=-1, keepdims=True)
        return s * (grad - dot)
