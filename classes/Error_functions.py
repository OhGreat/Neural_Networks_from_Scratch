import numpy as np

class MSE:
    """ Mean squared error
    """
    def __call__(self, y_true, y_pred, ax=0):
        return ((y_true - y_pred)**2).mean(axis=ax)