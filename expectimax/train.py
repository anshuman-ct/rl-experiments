# train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from model import ExpectimaxCNN
from expectimax import find_best_move, simulate_move, can_move

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_MAP = {
    'ArrowUp': 0,
    'ArrowDown': 1,
    'ArrowLeft': 2,
    'ArrowRight': 3
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


# --- DATASET GENERATION ---
def generate_data(num_games=2000):
    states = []
    actions = []

    for g in range(num_games):
        grid = init_grid()

        while can_move(grid):
            state = encode(grid)

            expert_action = find_best_move(grid)

            if expert_action is None:
                break

            # small noise to avoid overfitting
            if random.random() < 0.05:
                action = random.choice(list(ACTION_MAP.keys()))
            else:
                action = expert_action

            states.append(state)
            actions.append(ACTION_MAP[expert_action])  # label ALWAYS expert

            grid, moved = simulate_move(grid, action)
            if moved:
                add_random_tile(grid)

    return np.array(states), np.array(actions)


# --- TRAIN ---
def train():
    print("Generating dataset...")
    X, y = generate_data(1000)  
    X = torch.tensor(X).unsqueeze(1).to(DEVICE)
    y = torch.tensor(y).long().to(DEVICE)

    model = ExpectimaxCNN().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    epochs = 10

    for epoch in range(epochs):
        perm = torch.randperm(len(X))

        total_loss = 0

        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]

            xb = X[idx]
            yb = y[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "expectimax_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    train()