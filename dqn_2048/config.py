import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 200000

TARGET_UPDATE = 1000
MEMORY_SIZE = 100000

NUM_EPISODES = 10000