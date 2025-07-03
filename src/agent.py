from src.model import *
from src.buffer import ReplayBuffer, PrioritizedReplayBuffer, PERBufferSumTree
import torch
from torch.optim import RMSprop
import torch.nn as nn
import numpy as np
from typing import List
from torch.optim.lr_scheduler import CosineAnnealingLR

class GameAgent: 
    def __init__(self, 
                 frame_stack: int, 
                 ac_dim: int, 
                 lr: float = 1e-4, 
                 min_lr: float = 1e-5,
                 gamma: float = 0.995, 
                 max_memory: int = 100000, 
                 max_gradient: float = 1.0, 
                 action_mask: List[int] = [], 
                 buffer_type: str = "PER",
                 scheduler_max: int = 1000000,
                 beta_start: int = 0.5,
                 beta_frames: int = 10000, 
                 model_type: str = "EGM"):
        
        self.action_mask = action_mask
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if buffer_type == "REPLAY":
            self.buffer = ReplayBuffer(capacity=max_memory, device=self.device)
        elif buffer_type == "PER":
            self.buffer = PrioritizedReplayBuffer(capacity=max_memory, alpha=0.6, device=self.device)
        elif buffer_type == "PER_SUMTREE":
            self.buffer = PERBufferSumTree(max_len=max_memory, alpha=0.6)
        else:
            raise ValueError(f"Invalid buffer type: {buffer_type}. Expected 'replay' or 'prioritized'.")

        if model_type == "EGM":
            self.model = EfficientGameModel(frame_stack, ac_dim).to(self.device)
            self.target = EfficientGameModel(frame_stack, ac_dim).to(self.device)
        elif model_type == "GM":
            self.model = GameModel(frame_stack, ac_dim).to(self.device)
            self.target = GameModel(frame_stack, ac_dim).to(self.device)
        else: 
            raise ValueError("[ERROR] Invalid model type selected")
        
        self.opt = RMSprop(self.model.parameters(), lr=lr, alpha=0.95, eps=1e-2, centered=False)
        self.scheduler = CosineAnnealingLR(self.opt, scheduler_max, min_lr)
        
        self.criterion = nn.SmoothL1Loss(reduction="none")
        
        self.gamma = gamma
        self.action_dim = ac_dim
        self.max_grad = max_gradient
        self.beta_start = beta_start
        self.beta = self.beta_start
        self.beta_frames = beta_frames
        self.model_type = model_type
        self.training_step = 0

        self.update_target_network(True)
        
    def load_weights(self, path: str): 
        self.model.load_weights(path)
        self.target.load_weights(path)
    
    def save_weights(self, path: str):
        self.model.save_weights(path)
        
    def update_target_network(self, hard_update: bool = True, tau: float = 0.005):
        if hard_update: 
            self.target.load_state_dict(self.model.state_dict())
        else: 
            for target_param, param in zip(self.target.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
    def select_action(self, state: np.ndarray, epsilon: float = 0.01):
        is_batch = state.ndim == 4
        
        if not is_batch:
            state = np.expand_dims(state, axis=0)

        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        state_tensor = state_tensor / 255.0
        
        with torch.no_grad():
            q_values = self.model(state_tensor)

            if self.action_mask:
                mask = torch.full_like(q_values, 0.0)
                mask[:, self.action_mask] = float('-inf')
                q_values += mask

            batch_size = state.shape[0]
            if epsilon > 0.0:
                rand_mask = np.random.rand(batch_size) < epsilon
                random_actions = np.random.choice(
                    [i for i in range(self.action_dim) if i not in self.action_mask],
                    size=batch_size
                )
                greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
                actions = np.where(rand_mask, random_actions, greedy_actions)
            else:
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        return actions[0] if not is_batch else actions

            
    def update(self, batch_size: int, frame_step: int):
        if isinstance(self.buffer, PrioritizedReplayBuffer) or isinstance(self.buffer, PERBufferSumTree):
            self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (frame_step / self.beta_frames))
            states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(batch_size, beta=self.beta)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            weights = torch.ones_like(rewards, device=self.device)
            indices = None
        
        states = states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            max_next_q = self.target(next_states).gather(1, next_actions)           

            targets = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma * max_next_q
            targets = targets.squeeze(1).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        raw_loss = self.criterion(current_q_values, targets)
        weighted = weights * raw_loss
        loss = weighted.mean() 
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            td_errors = targets.detach() - current_q_values.detach()
            new_prios = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.buffer.update_priorities(indices, new_prios)
        
        q_value_mean = current_q_values.detach().cpu().numpy().mean() 
        q_value_std = current_q_values.detach().cpu().numpy().std()
        
        return loss.item(), q_value_mean, q_value_std
    
    def push(self, state, action, reward, next_state, done): 
        self.buffer.push(state, action, reward, next_state, done)