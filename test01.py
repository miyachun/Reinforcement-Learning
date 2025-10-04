import pygame
import torch
import imageio
import numpy as np
from PIL import Image

# === 參數設定（與訓練一致） ===
GRID_SIZE = 30
CELL_SIZE = 12
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
MODEL_PATH = "dqn_maze_large.pth"
GIF_PATH = "dqn_maze_test.gif"

# === 重新生成相同牆壁 ===
import random
random.seed(42)
WALLS = []
for _ in range(int(GRID_SIZE * GRID_SIZE * 0.07)):
    w = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    if w != START and w != GOAL:
        WALLS.append(w)
WALLS = list(set(WALLS))

# === CNN DQN 網路 ===
import torch.nn as nn
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * GRID_SIZE * GRID_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        x = x.view(-1, 1, GRID_SIZE, GRID_SIZE)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === 迷宮環境 ===
class MazeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent = list(START)
        return self._get_state()

    def step(self, action):
        r, c = self.agent
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        r = max(0, min(GRID_SIZE - 1, r))
        c = max(0, min(GRID_SIZE - 1, c))
        if (r, c) in WALLS:
            return self._get_state(), -0.5, False
        self.agent = [r, c]
        done = (r, c) == GOAL
        return self._get_state(), 1 if done else -0.01, done

    def _get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent[0], self.agent[1]] = 1
        state[GOAL[0], GOAL[1]] = 0.5
        for w in WALLS:
            state[w[0], w[1]] = -1
        return state

# === 初始化 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = 4
policy_net = QNetwork(action_size).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()

env = MazeEnv()

# === pygame 初始化 ===
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("DQN Maze - Test Run")

# === 選擇最佳動作 ===
def choose_best_action(state):
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = policy_net(state_t)
    return torch.argmax(q_values).item()

# === 開始走迷宮 ===
frames = []
state = env.reset()
done = False
steps = 0
MAX_STEPS = 100

while not done and steps < MAX_STEPS:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action = choose_best_action(state)
    state, reward, done = env.step(action)
    steps += 1

    # --- 畫面更新 ---
    screen.fill((255, 255, 255))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            if (r, c) == tuple(env.agent):
                pygame.draw.rect(screen, (255, 0, 0), rect)   # agent
            elif (r, c) == GOAL:
                pygame.draw.rect(screen, (0, 255, 0), rect)   # goal
            elif (r, c) in WALLS:
                pygame.draw.rect(screen, (0, 0, 0), rect)     # wall

    pygame.display.flip()

    # --- 擷取畫面到 GIF ---
    buffer = pygame.surfarray.array3d(screen)
    buffer = buffer.transpose([1, 0, 2])  # (w,h,3) → (h,w,3)
    img = Image.fromarray(buffer)
    frames.append(img.copy())

pygame.quit()

# === 儲存 GIF ===
if len(frames) > 1:
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f" 測試路徑已產生 GIF：{GIF_PATH}（總步數 {steps}）")
else:
    print(" 沒有產生任何移動畫面，可能模型尚未學會路徑。")
