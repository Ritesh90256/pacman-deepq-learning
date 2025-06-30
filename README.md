# Pac-Man AI Agent (Deep Convolutional Q-Learning)

This project implements a Deep Q-Learning (DQN) agent to play the classic Ms. Pac-Man game using the Atari environment from OpenAI Gymnasium.  
The agent learns how to play by maximizing cumulative rewards over time using a convolutional neural network and experience replay.

Built and trained entirely in Google Colab using PyTorch.

---

How It Works

- A Convolutional Neural Network (CNN) is used to process RGB frames and estimate Q-values.
- Frames are preprocessed and resized to 128x128 before being passed to the network.
- A Replay Buffer stores past experiences for stable learning.
- Target networks are updated to reduce instability during training.
- Trained with:
  - Frame size: 128×128×3
  - Action size: 9 (Ms. Pac-Man)
  - Discount factor γ = 0.99
  - Learning rate = 5e-4
  - Epsilon-greedy policy with decay for exploration

---

Libraries & Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Running the Code

- Since this was built and trained in Google Colab, to reproduce results:

1. Open src/train.py in Colab or locally

2. Run the training loop to train the agent and save a checkpoint (checkpoint.pth)

3. After training, run src/inference.py to render and save a video of the agent playing

4. The video will be saved in the /results/ folder

Results

The agent learns to survive longer and score higher over episodes.
A demo of the trained agent playing Ms. Pac-Man is saved as a video.

Example gameplay video can be found in /results/.

Project Structure :

Pacman/
├── src/ # Training and inference scripts
├── results/ # Gameplay video (.mp4)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

Note

This project was implemented line-by-line and trained in Colab using real-time learning and evaluation.

---

Author

Built with ❤️ by Ritesh
