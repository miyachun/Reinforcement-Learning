import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==== 迷宮設定 ====
GRID_SIZE = 30
CELL_SIZE = 12  # 避免太大視窗
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)

# 隨機生成一些牆壁（固定種子方便重現）
random.seed(42)
WALLS = []
for _ in range(int(GRID_SIZE * GRID_SIZE * 0.07)):  # 10% 牆壁
    w = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    if w != START and w != GOAL:
        WALLS.append(w)
WALLS = list(set(WALLS))


# ==== CNN DQN 網路 ====
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

# ==== 環境 ====
class MazeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent = list(START)
        return self._get_state()

    def step(self, action):
        r, c = self.agent
        old_distance = abs(r - GOAL[0]) + abs(c - GOAL[1])

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
        new_distance = abs(r - GOAL[0]) + abs(c - GOAL[1])

        # ✅ reward shaping
        shaping_reward = (old_distance - new_distance) * 0.1
        reward = 20 if done else shaping_reward - 0.01
        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent[0], self.agent[1]] = 1
        state[GOAL[0], GOAL[1]] = 0.5
        for w in WALLS:
            state[w[0], w[1]] = -1
        return state

# ==== DQN 參數 ====
EPISODES = 500
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 100

# ==== 初始化 ====
env = MazeEnv()
state_shape = (GRID_SIZE, GRID_SIZE)
action_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = QNetwork(action_size).to(device)
target_net = QNetwork(action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

# ==== Pygame 初始化 ====
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("🏁 DQN 迷宮 AI")

# ==== 選擇動作 ====
def choose_action(state):
    global EPSILON
    if random.random() < EPSILON:
        return random.randint(0, action_size - 1)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return torch.argmax(q_values).item()

# ==== 經驗回放 ====
def replay():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q = target_net(next_states).max(1)[0]
    target = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ==== 訓練 ====
for ep in range(EPISODES):
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < GRID_SIZE * GRID_SIZE * 4:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = choose_action(state)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        replay()
        state = next_state
        steps += 1

        # === 繪圖 ===
        screen.fill((255, 255, 255))
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)
                if (r, c) == tuple(env.agent):
                    pygame.draw.rect(screen, (255, 0, 0), rect)
                elif (r, c) == GOAL:
                    pygame.draw.rect(screen, (0, 255, 0), rect)
                elif (r, c) in WALLS:
                    pygame.draw.rect(screen, (0, 0, 0), rect)
        pygame.display.flip()

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    if ep % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f"Episode {ep+1}/{EPISODES} Steps={steps} Epsilon={EPSILON:.3f}")

MODEL_PATH = "dqn_maze_large.pth"
torch.save(policy_net.state_dict(), MODEL_PATH)
print(f"✅ 模型已儲存至 {MODEL_PATH}")

pygame.quit()
