import numpy as np

def encode_state(board):
    # Example: log2 encoding
    return np.log2(board + 1) / 16.0