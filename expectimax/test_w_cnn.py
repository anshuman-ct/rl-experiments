#test_w_cnn.py

import torch
import numpy as np
import random

from model import ExpectimaxCNN
from expectimax import simulate_move, can_move

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_MAP = {
    0: 'ArrowUp',
    1: 'ArrowDown',
    2: 'ArrowLeft',
    3: 'ArrowRight'
}

REVERSE_MAP = {v: k for k, v in ACTION_MAP.items()}


# --- ENCODING ---
def encode(grid):
    out = np.zeros((4, 4), dtype=np.float32)

    for r in range(4):
        for c in range(4):
            if grid[r][c] is not None:
                out[r][c] = np.log2(grid[r][c]["value"])

    return out


# --- GAME HELPERS ---
def add_random_tile(grid):
    empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] is None]
    if not empty:
        return

    r, c = random.choice(empty)
    grid[r][c] = {"value": 2 if random.random() < 0.9 else 4}


def init_grid():
    grid = [[None for _ in range(4)] for _ in range(4)]
    add_random_tile(grid)
    add_random_tile(grid)
    return grid


def compute_score(grid):
    score = 0
    max_tile = 0

    for row in grid:
        for cell in row:
            if cell:
                score += cell["value"]
                max_tile = max(max_tile, cell["value"])

    return score, max_tile


# --- LOAD MODEL ---
def load_model():
    model = ExpectimaxCNN().to(DEVICE)
    model.load_state_dict(torch.load("expectimax_cnn.pth"))
    model.eval()
    return model


# --- INFERENCE ---
def get_action(model, grid):
    state = encode(grid)
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(state)
        action = logits.argmax(dim=1).item()

    return ACTION_MAP[action]


# --- PLAY ONE GAME ---
def play_game(model, verbose=False):
    grid = init_grid()

    while can_move(grid):
        action = get_action(model, grid)

        new_grid, moved = simulate_move(grid, action)

        if not moved:
            # avoid getting stuck
            action = random.choice(list(ACTION_MAP.values()))
            new_grid, moved = simulate_move(grid, action)

        if moved:
            grid = new_grid
            add_random_tile(grid)
        else:
            break

    score, max_tile = compute_score(grid)

    if verbose:
        print("Final Grid:")
        for row in grid:
            print([c["value"] if c else 0 for c in row])

    return score, max_tile


# --- MAIN ---
def main():
    model = load_model()

    games = 20
    scores = []
    max_tiles = []

    for i in range(games):
        score, max_tile = play_game(model)

        scores.append(score)
        max_tiles.append(max_tile)

        print(f"Game {i+1}: Score={score}, MaxTile={max_tile}")

    print("\n--- Summary ---")
    print(f"Avg Score: {sum(scores)/len(scores):.2f}")
    print(f"Max Tile Distribution: {max_tiles}")


if __name__ == "__main__":
    main()