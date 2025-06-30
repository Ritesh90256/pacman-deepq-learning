import gymnasium as gym
import imageio
import torch
from train import Agent, preprocess_frame

env_name = 'ALE/MsPacman-v5'
env = gym.make(env_name, render_mode='rgb_array')
state, _ = env.reset()
frames = []

agent = Agent(env.action_space.n)
agent.local_qnetwork.load_state_dict(torch.load("checkpoint.pth"))

done = False
while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action)

env.close()
imageio.mimsave("results/pacman_demo.mp4", frames, fps=30)
