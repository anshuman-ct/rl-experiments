# expectimax.py

import math
import copy

GRID_SIZE = 4
DIRECTIONS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]

EVAL_CACHE = {}
EXPECTIMAX_CACHE = {}


# --- PUBLIC API ---
def find_best_move(grid):
    best_move = None
    best_score = float("-inf")

    empty_cells = sum(1 for row in grid for c in row if c is None)

    depth = 2 if empty_cells > 10 else (3 if empty_cells > 5 else 4)

    for direction in DIRECTIONS:
        new_grid, moved = simulate_move(grid, direction)

        if moved:
            score = expectimax(new_grid, depth, False, float("-inf"), float("inf"))

            if score > best_score:
                best_score = score
                best_move = direction

    return best_move


# --- EXPECTIMAX CORE ---
def expectimax(node_grid, depth, is_player_turn, alpha, beta):
    if depth == 0 or not can_move(node_grid):
        return evaluate_grid(node_grid)

    key = (_grid_to_hashable(node_grid), depth, is_player_turn)
    if key in EXPECTIMAX_CACHE:
        return EXPECTIMAX_CACHE[key]

    if is_player_turn:
        max_score = float("-inf")

        for direction in DIRECTIONS:
            new_grid, moved = simulate_move(node_grid, direction)

            if moved:
                score = expectimax(new_grid, depth - 1, False, alpha, beta)

                max_score = max(max_score, score)
                alpha = max(alpha, score)

                if beta <= alpha:
                    break

        EXPECTIMAX_CACHE[key] = max_score
        return max_score

    else:
        total_score = 0
        empty_cells = []

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if node_grid[r][c] is None:
                    empty_cells.append((r, c))

        if len(empty_cells) == 0:
            return evaluate_grid(node_grid)

        for r, c in empty_cells:
            # place 2
            grid2 = deep_copy(node_grid)
            grid2[r][c] = {"value": 2}
            total_score += 0.9 * expectimax(grid2, depth - 1, True, alpha, beta)

            # place 4
            grid4 = deep_copy(node_grid)
            grid4[r][c] = {"value": 4}
            total_score += 0.1 * expectimax(grid4, depth - 1, True, alpha, beta)

        result = total_score / len(empty_cells)
        EXPECTIMAX_CACHE[key] = result
        return result


# --- EVALUATION ---
def _grid_to_hashable(grid):
    return tuple(tuple(cell["value"] if cell else 0 for cell in row) for row in grid)


def evaluate_grid(grid):
    key = _grid_to_hashable(grid)
    if key in EVAL_CACHE:
        return EVAL_CACHE[key]

    orientations = [
        grid,
        flip_horizontal(grid),
        flip_vertical(grid),
        flip_vertical(flip_horizontal(grid)),
    ]

    max_score = float("-inf")

    for g in orientations:
        max_score = max(max_score, calculate_score_with_matrix(g))

    EVAL_CACHE[key] = max_score
    return max_score


def calculate_score_with_matrix(grid):
    weight_matrix = [
        [262144, 131072, 65536, 32768],
        [2048, 4096, 8192, 16384],
        [1024, 512, 256, 128],
        [8, 16, 32, 64],
    ]

    grid_score = 0
    empty_cells = 0
    smoothness = 0

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]

            if cell:
                value = cell["value"]
                grid_score += value * weight_matrix[r][c]

                if c < GRID_SIZE - 1 and grid[r][c + 1]:
                    smoothness -= abs(
                        math.log2(value) - math.log2(grid[r][c + 1]["value"])
                    )

                if r < GRID_SIZE - 1 and grid[r + 1][c]:
                    smoothness -= abs(
                        math.log2(value) - math.log2(grid[r + 1][c]["value"])
                    )
            else:
                empty_cells += 1

    empty_bonus = math.log(empty_cells + 1) * 20000

    return grid_score + empty_bonus + (smoothness * 10)


# --- SIMULATION ---
def simulate_move(sim_grid, direction):
    grid_copy = deep_copy(sim_grid)
    moved = False

    if direction in ["ArrowUp", "ArrowDown"]:
        working_grid = transpose(grid_copy)
    else:
        working_grid = grid_copy

    for i in range(GRID_SIZE):
        original = repr(working_grid[i])

        row = [cell for cell in working_grid[i] if cell is not None]

        if direction in ["ArrowRight", "ArrowDown"]:
            row.reverse()

        j = 0
        while j < len(row) - 1:
            if row[j]["value"] == row[j + 1]["value"]:
                row[j]["value"] *= 2
                row.pop(j + 1)
            j += 1

        while len(row) < GRID_SIZE:
            row.append(None)

        if direction in ["ArrowRight", "ArrowDown"]:
            row.reverse()

        working_grid[i] = row

        if repr(row) != original:
            moved = True

    if direction in ["ArrowUp", "ArrowDown"]:
        new_grid = transpose(working_grid)
    else:
        new_grid = working_grid

    return new_grid, moved


# --- HELPERS ---
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def flip_horizontal(grid):
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid):
    return list(reversed(grid))


def deep_copy(obj):
    return copy.deepcopy(obj)


def can_move(grid):
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] is None:
                return True

            if r < GRID_SIZE - 1:
                if grid[r + 1][c] and grid[r][c]["value"] == grid[r + 1][c]["value"]:
                    return True

            if c < GRID_SIZE - 1:
                if grid[r][c + 1] and grid[r][c]["value"] == grid[r][c + 1]["value"]:
                    return True

    return False
