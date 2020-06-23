import numpy as np

class SGD:
    def __init__(self, l=0.1):
        self._l = l
    
    def __call__(self, w, dw):
        return w - self._l * dw
