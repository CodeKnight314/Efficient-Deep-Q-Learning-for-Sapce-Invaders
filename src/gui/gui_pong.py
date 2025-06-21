from src.agent import GameModel
import pygame
import yaml
import torch
import gymnasium as gym
from src.wrappers import NoopResetEnv, MaxAndSkipEnv
from gymnasium.wrappers import ResizeObservation, FrameStackObservation
import numpy as np
import ale_py

gym.register_envs(ale_py)

class PongGUI:
    def __init__(self, config: str, weights: str = None, player_mode: str = "AI"):
        
        if player_mode not in ["human", "AI"]:
            raise ValueError(f"[ERROR] {player_mode} is not in available modes. Choose from {["human", "AI"]}")
        self.player_mode = player_mode
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.env = self._make_env("ALE/Pong-v5", render_mode="rgb_array")
        self.model = GameModel(4, self.env.action_space.n).to(self.device)  
        if weights is not None:
            try:
                self.model.load_weights(weights)   
            except Exception as e:
                print("[INFO] Attempted to load weights for Pong Agent but was not successful.")
                print("[INFO] Inititating random agent")
        
    def _make_env(self, env_id: str, render_mode: str = None):
        env = gym.make(env_id, render_mode=render_mode, obs_type="grayscale")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=self.config["skip_frame"])
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=int(self.config["frame_stack"]))
    
        return env
    
    def _get_bot_action(self, state: np.array):
        self.model.eval()
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state, normalize=True)
        
        action = torch.argmax(q_values, dim=1).item()
        return action
    
    def _get_human_action(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 2
        elif keys[pygame.K_DOWN]:
            return 5
        else:
            return 0

    def _get_action(self, state: np.array):
        if self.player_mode == "human":
            return self._get_human_action()
        elif self.player_mode == "AI":
            return self._get_bot_action(state)
    
    def run(self):
        print("[INFO] Starting GUI loop...")
        pygame.init()
        screen = pygame.display.set_mode((420, 420))
        clock = pygame.time.Clock()

        state, _ = self.env.reset()
        done = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    pygame.quit()
                    return

            action = self._get_action(state)
            state, reward, done, truncated, _ = self.env.step(action)
            frame = self.env.render()

            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (420, 420)), (0, 0))
            pygame.display.flip()

            if done or truncated:
                state, _ = self.env.reset()

            if self.player_mode == "human":
                clock.tick(10)
            else:
                clock.tick(30)
                    