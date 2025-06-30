import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, ResizeObservation
from src.wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv
from tqdm import tqdm
from src.agent import GameAgent
import yaml
import logging
import matplotlib.pyplot as plt
from collections import deque
import ale_py

gym.register_envs(ale_py)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class GameEnv(): 
    def __init__(self, seed: int, env_id: str, num_envs: int, config: str, weights: str = None, verbose: bool = True):
        logger.info(f"Initializing Tetris environment with {num_envs} parallel environments")
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.seed = seed
        self.num_envs = num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer_type = self.config["buffer_type"]
        logger.info(f"Using device: {self.device}")

        self.set_seed(seed)
        
        self.id = env_id
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: self._make_env(env_id=env_id) for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        logger.info("Initializing TetrisAgent with configuration:")
        logger.info(f"- Environment: {env_id}")
        logger.info(f"- Frame stack: {self.config['frame_stack']}")
        logger.info(f"- Learning rate: {self.config['lr']}")
        logger.info(f"- Gamma: {self.config['gamma']}")
        logger.info(f"- Max memory: {self.config['max_memory']}")
        logger.info(f"- Action Space: {self.env.action_space[0].n}")
        
        self.agent = GameAgent(self.config["frame_stack"], 
                                 self.env.action_space[0].n, 
                                 self.config["lr"], 
                                 self.config["min_lr"],
                                 self.config["gamma"], 
                                 self.config["max_memory"], 
                                 self.config["max_gradient"], 
                                 self.config["action_mask"], 
                                 self.buffer_type,
                                 self.config["scheduler_max"],
                                 self.config["beta_start"], 
                                 self.config["beta_frames"],
                                 self.config["n_step"])
        
        if weights is not None: 
            logger.info(f"Loading pre-trained weights from: {weights}")
            self.agent.load_weights(weights)

        self.history = {
            "reward": deque(maxlen=self.config["window_size"] * 2),
            "loss": deque(maxlen=self.config["window_size"] * 2),
            "reward_history": [], 
            "loss_history": []
        }
        
        self.n_step_buffer = [deque(maxlen=self.config["n_step"]) for _ in range(self.num_envs)]
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.max_frames = self.config["max_frames"]
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        self.reward_window_size = self.config["window_size"]
        
        self.best_reward = float('-inf')
        self.save_freq = self.config["save_freq"]
        self.verbose = verbose
        
        logger.info(f"Environment initialized with seed: {seed}")
        
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _make_env(self, env_id: str, render_mode: str = None):
        env = gym.make(env_id, render_mode=render_mode, obs_type="grayscale")
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=int(self.config["frame_stack"]))
    
        return env
        
    def compute_n_step_transition(self, buffer):
        R = 0.0
        last_idx = 0
        for idx, (s, a, r, ns, done) in enumerate(buffer):
            R += (self.agent.gamma ** idx) * r
            last_idx = idx
            if done:
                break

        s0, a0, *_ = buffer[0]
        _,  _,  _, ns_last, done_last = buffer[last_idx]
        gamma_n = self.agent.gamma ** (last_idx + 1)
        return s0, a0, R, ns_last, done_last, gamma_n
        
    def train(self, path: str):
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)

        total_frames = 0
        episode_rewards = np.zeros(self.num_envs, dtype=float)

        state, _ = self.env.reset()  

        pbar = tqdm(total=self.max_frames, desc="Frames")

        while total_frames < self.max_frames:
            epsilon = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** (total_frames / 1000)))

            actions = []
            for i in range(self.num_envs):
                single_board = state[i]
                a_i = self.agent.select_action(single_board, epsilon)
                actions.append(int(a_i))
                    
            actions = np.array(actions, dtype=np.int32)

            next_state, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(self.num_envs):
                prev_board = torch.tensor(state[i]).float()
                next_board = torch.tensor(next_state[i]).float()
                
                r_i = float(rewards[i])
                d_i = bool(dones[i])
                
                self.n_step_buffer[i].append((prev_board, actions[i], r_i, next_board, d_i))
                episode_rewards[i] += r_i
                
                if len(self.n_step_buffer[i]) == self.agent.n_step or d_i :
                    n_step_transition = self.compute_n_step_transition(self.n_step_buffer[i])
                    self.agent.push(*n_step_transition)
                    
                    if d_i: 
                        self.n_step_buffer[i].clear()

                if d_i:
                    self.history["reward"].append(float(episode_rewards[i]))
                    episode_rewards[i] = 0.0

                total_frames += 1

                if total_frames % self.update_freq == 0:
                    self.agent.update_target_network(hard_update=True)
                    if self.verbose:
                        logger.info(f"Target network updated at frame {total_frames}")

                if total_frames % self.save_freq == 0:
                    checkpoint_path = os.path.join(path, f"checkpoint.pth")
                    self.plot_history(path)
                    self.agent.save_weights(checkpoint_path)
                    if self.verbose:
                        logger.info(f"Checkpoint saved at frame {total_frames}")

            pbar.update(self.num_envs)

            q_value_mean = 0.0
            q_value_std = 0.0
            
            if len(self.agent.buffer) > self.batch_size:
                loss, mean, std = self.agent.update(self.batch_size)
                q_value_mean = mean
                q_value_std = std
                self.history["loss"].append(loss)

            if len(self.history["reward"]) >= self.reward_window_size:
                recent_reward_avg = np.mean(self.history["reward"]) if len(self.history["reward"]) > 0 else 0.0
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    best_model_path = os.path.join(path, "best_model.pth")
                    self.agent.save_weights(best_model_path)

                    if self.verbose:
                        logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

            state = next_state

            pbar_rewards = np.mean(self.history['reward']) if len(self.history["reward"]) > 0 else 0.0
            pbar_loss = np.mean(self.history['loss']) if len(self.history["loss"]) > 0 else 0.0
            self.history["reward_history"].append(pbar_rewards)
            self.history["loss_history"].append(pbar_loss)
            
            pbar.set_postfix(
                reward=f"{pbar_rewards:.4f}", 
                loss=f"{pbar_loss:.4f}", 
                epsilon=f"{epsilon:.4f}",
                beta=f"{self.agent.beta:.4f}",
                q_values=f"{q_value_mean:.4f}",
                q_values_std=f"{q_value_std:.4f}",
            )

        pbar.close()
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final_model.pth"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final_model.pth')}")
        
        return np.mean(self.history['reward'])

    def test(self, path: str, num_episodes: int, random: bool = False):
        import cv2
        os.makedirs(path, exist_ok=True)

        env = self._make_env(self.id, render_mode="rgb_array")
        self.agent.model.eval()

        state, _ = env.reset()
        frame = env.render()
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(path, "record.mp4")
        video = cv2.VideoWriter(video_path, fourcc, 60, (width, height))

        total_rewards = 0
        total_steps = 0

        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            rewards = 0
            steps = 0

            while not done:
                frame = env.render()

                if random:
                    action = env.action_space.sample()
                else: 
                    action = self.agent.select_action(state, 0.0)
                
                video.write(frame)

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards += reward
                steps += 1

            if self.verbose:
                logger.info(f"Episode {i + 1} - Reward: {rewards:.2f} - Steps: {steps}")

            total_rewards += rewards
            total_steps += steps

        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes

        if self.verbose:
            logger.info(f"Average reward: {avg_reward:.2f} - Average steps: {avg_steps:.2f}")

        video.release()
        if self.verbose:
            logger.info(f"Video saved to: {video_path}")
        del env


    def close(self):
        self.env.close() 
        del self.agent
        torch.cuda.empty_cache()
    
    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.agent.save_weights(path)

    def plot_history(self, path: str):
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.history["reward_history"], label="Reward")
        plt.legend()
        plt.savefig(os.path.join(path, "reward_history.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history["loss_history"], label="Loss")        
        plt.legend()
        plt.savefig(os.path.join(path, "loss_history.png"))
        plt.close()