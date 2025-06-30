import os
import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from PIL import Image
from torchvision import transforms

# Hyperparameters
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=4)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10*10*128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return preprocess(frame).unsqueeze(0)

class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > minibatch_size:
            experiences = random.sample(self.memory, k=minibatch_size)
            self.learn(experiences)

    def act(self, state, epsilon=0.0):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(1).to(self.device)

        next_q = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        target_q = rewards + discount_factor * next_q * (1 - dones)

        predicted_q = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Environment
env = gym.make('ALE/MsPacman-v5', full_action_space=False)
number_actions = env.action_space.n
agent = Agent(number_actions)

# Training loop
episodes = 2000
max_timesteps = 10000
eps = 1.0
eps_end = 0.01
eps_decay = 0.995
scores = deque(maxlen=100)

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = agent.act(state, eps)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores.append(score)
    eps = max(eps_end, eps * eps_decay)
    print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}", end="")
    if episode % 100 == 0:
        print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}")
    if np.mean(scores) >= 500.0:
        print(f"\nEnvironment solved in {episode} episodes!")
        torch.save(agent.local_qnetwork.state_dict(), "checkpoint.pth")
        break
