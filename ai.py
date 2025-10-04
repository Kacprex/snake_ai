import os, random, numpy as np, matplotlib.pyplot as plt, json
import torch, torch.nn as nn, torch.optim as optim
from gameai_vec import SnakeVecEnv


# --- Dueling DQN Model ---
class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean(dim=1, keepdim=True)


# --- Prioritized Replay Buffer ---
class PrioritizedReplay:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    def __len__(self): return len(self.buffer)


# --- Greedy evaluation ---
def evaluate(model, env, device, episodes=5):
    model.eval()
    total_len, total_apples, total_reward = 0, 0, 0
    with torch.no_grad():
        for _ in range(episodes):
            states = torch.FloatTensor(env.reset()).to(device)
            done_flags = np.zeros(env.n_envs, dtype=bool)
            while not done_flags.any():
                q_values = model(states)
                actions = q_values.argmax(dim=1).cpu().numpy()
                next_states, rewards, dones, lengths = env.step(actions)
                states = torch.FloatTensor(next_states).to(device)
                total_reward += rewards.mean()
                total_len += lengths.mean()
                total_apples += (rewards > 0).mean()
                done_flags = dones
    model.train()
    return total_reward / episodes, total_len / episodes, total_apples / episodes


# --- Training ---
def train_dqn(episodes=2000, save_path="snake_ai.pth", num_apples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    env = SnakeVecEnv(n_envs=8, w=10, h=10, num_apples=num_apples)
    state_size = env.w * env.h

    model = DuelingDQN(state_size).to(device)
    target_model = DuelingDQN(state_size).to(device)
    target_model.load_state_dict(model.state_dict())

    if os.path.exists(save_path):
        try:
            state_dict = torch.load(save_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            target_model.load_state_dict(state_dict)
            print("Loaded model from", save_path)
        except RuntimeError:
            print("⚠️ Old model incompatible, starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    memory = PrioritizedReplay()
    gamma = 0.99
    epsilon, eps_min, eps_decay = 1.0, 0.01, 0.9995  # slower decay
    batch_size = 256
    beta_start, beta_frames = 0.4, 10000

    rewards_hist, length_hist, apples_hist, eval_hist = [], [], [], []

    for ep in range(episodes):
        states = torch.FloatTensor(env.reset()).to(device)
        total_rewards = np.zeros(env.n_envs)
        apples = np.zeros(env.n_envs)
        max_lengths = np.ones(env.n_envs)
        done_flags = np.zeros(env.n_envs, dtype=bool)

        while not done_flags.any():
            if random.random() < epsilon:
                actions = [random.randint(0, 3) for _ in range(env.n_envs)]
            else:
                with torch.no_grad():
                    q_values = model(states)
                    actions = q_values.argmax(dim=1).cpu().numpy()

            next_states, rewards, dones, lengths = env.step(actions)
            next_states = torch.FloatTensor(next_states).to(device)

            apples += (rewards > 0).astype(int)
            for i in range(env.n_envs):
                memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            total_rewards += rewards
            max_lengths = np.maximum(max_lengths, lengths)
            done_flags = dones

            if len(memory) > batch_size:
                beta = beta_start + (1 - beta_start) * ep / beta_frames
                states_b, actions_b, rewards_b, nss_b, dones_b, indices, weights = memory.sample(batch_size, beta)
                states_b = torch.stack(states_b).to(device)
                nss_b = torch.stack(nss_b).to(device)
                actions_b = torch.LongTensor(actions_b).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                dones_b = torch.FloatTensor(np.array(dones_b, dtype=np.float32)).to(device)
                weights = weights.to(device)
                q_values = model(states_b).gather(1, actions_b.unsqueeze(1)).squeeze()
                next_actions = model(nss_b).argmax(1)
                next_q = target_model(nss_b).gather(1, next_actions.unsqueeze(1)).squeeze()
                target = rewards_b + gamma * next_q * (1 - dones_b)
                loss = (q_values - target.detach()).pow(2) * weights
                prios = loss + 1e-5
                memory.update_priorities(indices, prios.data.cpu().numpy())
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        rewards_hist.append(total_rewards.mean())
        length_hist.append(max_lengths.mean())
        apples_hist.append(apples.mean())

        if ep % 50 == 0:
            target_model.load_state_dict(model.state_dict())
        if (ep + 1) % 50 == 0:
            eval_r, eval_len, eval_app = evaluate(model, env, device)
            eval_hist.append((ep+1, eval_r, eval_len, eval_app))
            print(f"Ep {ep+1}, TrainAvgReward {np.mean(rewards_hist[-50:]):.2f}, "
                  f"TrainAvgLen {np.mean(length_hist[-50:]):.2f}, "
                  f"EvalReward {eval_r:.2f}, EvalLen {eval_len:.2f}, "
                  f"EvalApples {eval_app:.2f}, Eps {epsilon:.3f}")

    torch.save(model.state_dict(), save_path)
    summary = {
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards_hist[-50:])),
        "avg_length": float(np.mean(length_hist[-50:])),
        "avg_apples": float(np.mean(apples_hist[-50:])),
        "epsilon": float(epsilon),
        "eval_last": eval_hist[-1] if eval_hist else None
    }
    with open("training_summary.json", "w") as f:
        json.dump(summary, f)

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_hist, label="Train Reward")
    plt.plot(length_hist, label="Train Length")
    plt.plot(apples_hist, label="Train Apples")
    if eval_hist:
        eval_x = [x[0] for x in eval_hist]
        eval_r = [x[1] for x in eval_hist]
        plt.plot(eval_x, eval_r, label="Eval Reward", linestyle="--")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_dqn(2000, num_apples=5)
