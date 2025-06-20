from src.env import GameEnv
import argparse

MAPPING = {
    "pong": "ALE/Pong-v5", 
    "breakout": "ALE/Breakout-v5",
    "invaders": "ALE/SpaceInvaders-v5"
}

def main(args):
    try: 
        env_map = MAPPING[args.id]
        env = GameEnv(args.seed, env_map, args.num_envs, args.c, args.w, args.verbose)
        if args.mode == "train":
            env.train(args.o)
        else: 
            env.test(args.o, args.num_episodes)
    except KeyboardInterrupt as e: 
        env.save_weights(args.o)
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Render Tetris Simulations")
    parser.add_argument("--id", type=str, default="pong", choices=["pong", "breakout", "invaders"])
    parser.add_argument("--c", type=str, help="Config path for Tetris model")
    parser.add_argument("--o", type=str, help="Output path for Tetris model")
    parser.add_argument("--w", type=str, help="Path to available weights")
    parser.add_argument("--seed", type=int, default=1898, help="Seed for experiment reproduciability")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Mode to test/train Tetris Agent")
    parser.add_argument("--num_envs", type=int, help="Number of parallel environments to run at the same time")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run for testing")
    parser.add_argument("--verbose", action="store_true", help="Render the Tetris game")
    args = parser.parse_args()
    
    main(args)