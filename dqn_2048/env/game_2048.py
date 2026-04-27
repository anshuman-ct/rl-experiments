import numpy as np

class Env2048:
    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self._score = 0
        self.add_tile()
        self.add_tile()

    def reset(self):
        self.__init__()
        return self.grid

    def add_tile(self):
        i, j = (self.grid == 0).nonzero()
        idx = np.random.choice(len(i))
        self.grid[i[idx], j[idx]] = np.random.choice([1, 2], p=[0.9, 0.1])

    def step(self, action):
        rotated = np.rot90(self.grid, action)
        next_state = np.zeros((4, 4), dtype=int)

        for col in range(4):
            tiles = rotated[col]
            result = np.zeros(4, dtype=int)
            j, prev = 0, None

            for i in range(4):
                if tiles[i] != 0:
                    if prev is None:
                        prev = tiles[i]
                    elif prev == tiles[i]:
                        result[j] = tiles[i] + 1
                        self._score += 1 << result[j]
                        j += 1
                        prev = None
                    else:
                        result[j] = prev
                        j += 1
                        prev = tiles[i]

            if prev is not None:
                result[j] = prev

            next_state[col] = result

        next_state = np.rot90(next_state, -action)

        moved = not (next_state == self.grid).all()

        if moved:
            self.grid = next_state
            self.add_tile()

        reward = self.empty_tiles()
        done = self.is_done()

        return self.grid, reward, done

    def empty_tiles(self):
        return (self.grid == 0).sum()

    def is_done(self):
        if self.empty_tiles() > 0:
            return False

        for i in range(4):
            for j in range(4):
                if i > 0 and self.grid[i][j] == self.grid[i-1][j]:
                    return False
                if j > 0 and self.grid[i][j] == self.grid[i][j-1]:
                    return False
        return True

    def score(self):
        return self._score