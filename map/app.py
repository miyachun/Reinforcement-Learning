import os
import random
from collections import deque, namedtuple
import math
import numpy as np
import osmnx as ox
import networkx as nx
import folium
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 20000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 50  # episodes
MAX_STEPS_PER_EPISODE = 300
NUM_EPISODES = 800

# ---------------------------
# Utilities / Replay Buffer
# ---------------------------
Transition = namedtuple('Transition', ('s_c', 's_n', 's_goal', 'action_n_coord', 'reward', 'next_s_c', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Q-network: takes (current_coord, neighbor_coord, goal_coord) -> Q scalar
# coords normalized
# ---------------------------
class QNet(nn.Module):
    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # output scalar Q
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------
# Environment wrapper using osmnx Graph
# state representation: (current_lat, current_lon, neighbor_lat, neighbor_lon, goal_lat, goal_lon)
# reward: -1 per step, +100 for reaching goal
# ---------------------------
class GraphEnv:
    def __init__(self, G, start_node, goal_node):
        self.G = G
        self.start = start_node
        self.goal = goal_node
        self.current = start_node

    def reset(self):
        self.current = self.start
        return self.get_state(self.current)

    def get_state(self, node):
        # return current coord (lat, lon)
        node_data = self.G.nodes[node]
        return np.array([node_data['y'], node_data['x']])  # lat, lon

    def neighbors(self, node):
        return list(self.G.neighbors(node))

    def step(self, next_node):
        done = False
        reward = -1.0
        if next_node == self.goal:
            reward = 100.0
            done = True
            self.current = next_node
            return self.get_state(self.current), reward, done
        # safety if next_node not connected
        if next_node not in self.G:
            reward = -10.0
            done = True
            return self.get_state(self.current), reward, done

        self.current = next_node
        return self.get_state(self.current), reward, done


def compute_normalizer(G):
    lats = [data['y'] for _, data in G.nodes(data=True)]
    lons = [data['x'] for _, data in G.nodes(data=True)]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    def norm(lat, lon):
        # map to [-1, 1]
        la = ( (lat - lat_min) / (lat_max - lat_min + 1e-9) ) * 2 - 1
        lo = ( (lon - lon_min) / (lon_max - lon_min + 1e-9) ) * 2 - 1
        return la, lo
    return norm


def select_action(policy_net, state_c, candidate_neighbors, goal_coord, eps, norm_fn):
    # state_c: lat, lon
    # candidate_neighbors: list of node IDs
    # compute Q for each candidate
    if random.random() < eps:
        return random.choice(candidate_neighbors)
    # otherwise greedy: compute Q for each neighbor
    inputs = []
    for n in candidate_neighbors:
        ny, nx = G.nodes[n]['y'], G.nodes[n]['x']
        lc, lo = norm_fn(state_c[0], state_c[1])
        ln, lon = norm_fn(ny, nx)
        lg, lgx = norm_fn(goal_coord[0], goal_coord[1])
        # input: current(lat,lon), neighbor(lat,lon), goal(lat,lon) -> 6 dims
        inputs.append([lc, lo, ln, lon, lg, lgx])
    inputs = torch.tensor(inputs, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        qs = policy_net(inputs)  # shape (num_candidates,)
    idx = torch.argmax(qs).item()
    return candidate_neighbors[idx]

def optimize_model(policy_net, target_net, buffer, optimizer, norm_fn):
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    # build tensors
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    s_c = np.array([norm_fn(a[0], a[1]) for a in transitions.s_c])  # current coords
    s_n = np.array([norm_fn(a[0], a[1]) for a in transitions.s_n])  # neighbor coords (action)
    s_g = np.array([norm_fn(a[0], a[1]) for a in transitions.s_goal])  # goal coords
    # join -> [lc, lo, ln, lon, lg, lgx]
    input_x = np.hstack([s_c, s_n, s_g])
    input_x = to_tensor(input_x)

    rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(transitions.done, dtype=torch.float32, device=DEVICE)

    # next state's best Q: for each next_s_c we need to evaluate candidate neighbors; approximate by sampling one neighbor (simple) OR compute max over actual neighbors (slower).
    # We'll approximate by sampling up to k neighbors and using the max predicted by target_net.
    next_s_cs = transitions.next_s_c  # array of coords
    next_max_qs = []
    for ns in next_s_cs:
        node = coord_to_node[(ns[0], ns[1])]
        nbrs = list(G.neighbors(node))
        if len(nbrs) == 0:
            next_max_qs.append(0.0)
            continue
        inputs_next = []
        for n in nbrs:
            ny, nx = G.nodes[n]['y'], G.nodes[n]['x']
            lc, lo = norm_fn(ns[0], ns[1])  # next current
            ln, lon = norm_fn(ny, nx)
            lg, lgx = norm_fn(goal_coord_global[0], goal_coord_global[1])
            inputs_next.append([lc, lo, ln, lon, lg, lgx])
        with torch.no_grad():
            tq = target_net(torch.tensor(inputs_next, dtype=torch.float32, device=DEVICE))
            next_max_qs.append(float(torch.max(tq).cpu().numpy()))
    next_max_qs = torch.tensor(next_max_qs, dtype=torch.float32, device=DEVICE)

    expected_q = rewards + (1.0 - dones) * GAMMA * next_max_qs

    pred_q = policy_net(input_x)  # predicted q for chosen (s, a)

    loss = nn.MSELoss()(pred_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

#drive - 不包含辅路的机动车道
#drive_service - 包含辅路的机动车道
#walk - 可步行街道，不考虑单双向
#bike - 可骑行自行车道
#all - 除私有道路的所有道路
#all_private - 包含私有道路
if __name__ == "__main__":
    # 1) Download graph for an area
    place_name = "Zhongzheng District, Taipei, Taiwan"  # 可更換
    print("Downloading graph for:", place_name)
    G = ox.graph_from_place(place_name, network_type='drive_service')

    # simplify & convert to undirected for easy neighbor listing
    G = G.to_undirected()

    # 2) choose start and goal (coordinates) or replace with your lat/lon
    #start_latlon = (25.0478, 121.5170)  # Taipei Main Station
    #goal_latlon = (25.0330, 121.5654)   # Taipei 101
    start_latlon = (25.0478, 121.5170)
    goal_latlon = (25.0400, 121.5243)

    start_node = ox.distance.nearest_nodes(G, start_latlon[1], start_latlon[0])
    goal_node = ox.distance.nearest_nodes(G, goal_latlon[1], goal_latlon[0])
    print("start node", start_node, "goal node", goal_node)

    env = GraphEnv(G, start_node, goal_node)

    # precompute mapping from coord to node for quick lookup in buffer processing
    coord_to_node = {}
    for n, d in G.nodes(data=True):
        coord_to_node[(d['y'], d['x'])] = n

    # store a global goal for optimization access
    goal_coord_global = (G.nodes[goal_node]['y'], G.nodes[goal_node]['x'])

    norm_fn = compute_normalizer(G)

    policy_net = QNet().to(DEVICE)
    target_net = QNet().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer()

    eps = EPS_START
    losses = []
    print("Start training...")
    for ep in trange(NUM_EPISODES):
        s_c_coord = env.reset()  # lat, lon
        steps = 0
        done = False

        while not done and steps < MAX_STEPS_PER_EPISODE:
            cur_node = env.current
            nbrs = env.neighbors(cur_node)
            if len(nbrs) == 0:
                break
            # select action (neighbor node)
            action_node = select_action(policy_net, s_c_coord, nbrs, goal_coord_global, eps, norm_fn)
            # gather features for buffer: current coord, neighbor coord, goal coord
            s_n_coord = (G.nodes[action_node]['y'], G.nodes[action_node]['x'])
            next_s_coord, reward, done = env.step(action_node)
            # push to replay
            buffer.push(
                (s_c_coord[0], s_c_coord[1]),  # s_c
                (s_n_coord[0], s_n_coord[1]),  # s_n (action)
                (goal_coord_global[0], goal_coord_global[1]),  # s_goal
                (s_n_coord[0], s_n_coord[1]),  # action_n_coord (duplicate)
                reward,
                (next_s_coord[0], next_s_coord[1]),  # next_s_c
                done
            )
            s_c_coord = next_s_coord
            steps += 1

            # optimize
            l = optimize_model(policy_net, target_net, buffer, optimizer, norm_fn)
            if l is not None:
                losses.append(l)

        # decay eps
        eps = max(EPS_END, eps * EPS_DECAY)

        # update target
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Training finished.")


    print("Run greedy policy to get path...")
    env.current = start_node
    path = [start_node]
    visited = set([start_node])
    cur_coord = (G.nodes[start_node]['y'], G.nodes[start_node]['x'])
    for step in range(1000):
        if env.current == goal_node:
            break
        nbrs = env.neighbors(env.current)
        if len(nbrs) == 0:
            break
        nxt = select_action(policy_net, cur_coord, nbrs, goal_coord_global, eps=0.0, norm_fn=norm_fn)
        if nxt in visited:
            # fallback: shortest path step to goal to avoid loops
            try:
                sp = nx.shortest_path(G, env.current, goal_node, weight='length')
                if len(sp) > 1:
                    nxt = sp[1]
                else:
                    break
            except Exception:
                break
        path.append(nxt)
        visited.add(nxt)
        env.current = nxt
        cur_coord = (G.nodes[nxt]['y'], G.nodes[nxt]['x'])
        if len(path) > 500:
            break

    print("Path len:", len(path))


    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    m = folium.Map(location=[start_latlon[0], start_latlon[1]], zoom_start=14)
    folium.Marker([start_latlon[0], start_latlon[1]], popup="Start").add_to(m)
    folium.Marker([goal_latlon[0], goal_latlon[1]], popup="Goal").add_to(m)
    folium.PolyLine(route_coords, color="red", weight=5, opacity=0.8).add_to(m)
    out_file = "map.html"
    m.save(out_file)
    print("Saved map to", out_file)
