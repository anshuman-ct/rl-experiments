import numpy as np

def encode_state(grid):
    """
    Convert (4x4) grid with log2 tiles into (16,4,4) one-hot tensor
    """
    encoded = np.zeros((16, 4, 4), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            val = grid[i][j]
            if val > 0:
                encoded[val - 1][i][j] = 1.0

    return encoded