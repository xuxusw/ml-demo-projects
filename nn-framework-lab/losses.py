from __future__ import annotations

import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MSELoss(Loss):
    """
    L = (1/n) * Σ (y_pred - y_true)^2
    ∂L/∂y_pred = (2/n) * (y_pred - y_true)
    """
    
    def forward(self, y_pred, y_true):
        self.y_pred = np.asarray(y_pred, dtype=float)
        self.y_true = np.asarray(y_true, dtype=float)
        if self.y_pred.shape != self.y_true.shape:
            raise ValueError(f"Shape mismatch: {self.y_pred.shape} vs {self.y_true.shape}")
        return float(np.mean((self.y_pred - self.y_true) ** 2))

    def backward(self):
        n = self.y_true.shape[0]
        return (2.0 / n) * (self.y_pred - self.y_true)


class BinaryCrossEntropyLoss(Loss):
    """
    L = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    ∂L/∂y_pred = (y_pred - y_true) / (y_pred * (1 - y_pred) * n)
    """
    
    def forward(self, y_pred, y_true):
        eps = 1e-12
        self.y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
        self.y_true = np.asarray(y_true, dtype=float)
        loss = -(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return float(np.mean(loss))

    def backward(self):
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * n)


class CrossEntropyLoss(Loss):
    """
    L = -Σ y_true * log(y_pred)
    ∂L/∂y_pred = softmax(y_pred) - y_true
    """
    
    def forward(self, y_pred, y_true):
        eps = 1e-12
        self.y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
        self.y_true = np.asarray(y_true, dtype=float)
        loss = -np.sum(self.y_true * np.log(self.y_pred), axis=-1)
        return float(np.mean(loss))

    def backward(self):
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / n
