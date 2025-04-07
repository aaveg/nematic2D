import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RLAgent(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        super(RLAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        self.model = nn.Sequential(
            nn.Linear(state_size + 1, 24),  # +1 for setpoint
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, setpoint):
        if np.random.rand() < self.exploration_rate:
            return np.random.uniform(-1, 1, self.action_size)
        state = torch.FloatTensor(np.append(state, setpoint)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.numpy()[0]

    def learn(self, state, action, reward, next_state, setpoint):
        state = torch.FloatTensor(np.append(state, setpoint)).unsqueeze(0)
        next_state = torch.FloatTensor(np.append(next_state, setpoint)).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        reward = torch.FloatTensor([reward])

        target = reward + self.discount_factor * torch.max(self.model(next_state))
        current_q = self.model(state).gather(1, action.long().unsqueeze(0))

        loss = self.criterion(current_q, target.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

state_size = 10  # Example state size
action_size = 5  # Example action size

agent = RLAgent(state_size, action_size)

# Example training loop
for episode in range(100):
    state = np.random.randint(0, state_size)
    setpoint = np.random.randint(0, state_size)  # Random setpoint for each episode
    for t in range(50):
        action = agent.choose_action(state, setpoint)
        next_state = (state + action - (action_size // 2)) % state_size
        reward = -abs(next_state - setpoint)
        agent.learn(state, action, reward, next_state, setpoint)
        agent.update_exploration_rate()
        state = next_state