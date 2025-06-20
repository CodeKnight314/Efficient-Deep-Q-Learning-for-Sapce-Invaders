import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max  
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _, info = self.env.step(self.noop_action)
            if done: 
                obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

        obs_space = env.observation_space
        if isinstance(obs_space, Dict):
            obs_shape = obs_space["board"].shape
        elif isinstance(obs_space, Box):
            obs_shape = obs_space.shape
        else:
            raise TypeError("Unsupported observation space type")

        self._obs_buffer = np.zeros((2,) + obs_shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if isinstance(obs, dict):
                obs_frame = obs["board"]
            else:
                obs_frame = obs

            if i == self._skip - 2: self._obs_buffer[0] = obs_frame
            if i == self._skip - 1: self._obs_buffer[1] = obs_frame

            total_reward += reward
            terminated, truncated = term, trunc
            if terminated or truncated:
                break

        max_frame = self._obs_buffer.max(axis=0)
        if isinstance(obs, dict):
            obs["board"] = max_frame
            return obs, total_reward, terminated, truncated, info
        else:
            return max_frame, total_reward, terminated, truncated, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in action_meanings
        self.fire_action = action_meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)
