import numpy as np
from nn.losses import MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss


def test_mse_loss():
    loss = MSELoss()
    y_pred = np.array([[1.0], [2.0]])
    y_true = np.array([[1.0], [3.0]])
    value = loss.forward(y_pred, y_true)
    assert value >= 0
    grad = loss.backward()
    assert grad.shape == y_pred.shape


def test_bce_loss():
    loss = BinaryCrossEntropyLoss()
    y_pred = np.array([[0.9], [0.1]])
    y_true = np.array([[1.0], [0.0]])
    value = loss.forward(y_pred, y_true)
    assert value >= 0
    grad = loss.backward()
    assert grad.shape == y_pred.shape


def test_ce_loss():
    loss = CrossEntropyLoss()
    y_pred = np.array([[0.7, 0.2, 0.1]])
    y_true = np.array([[1.0, 0.0, 0.0]])
    value = loss.forward(y_pred, y_true)
    assert value >= 0
    grad = loss.backward()
    assert grad.shape == y_pred.shape
