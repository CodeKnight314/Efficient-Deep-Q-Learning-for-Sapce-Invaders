import argparse
from src.env import GameEnv
from src.gui import PongGUI, InvadersGUI, BreakoutGUI

MAPPING = {
    "pong": "ALE/Pong-v5",
    "breakout": "ALE/Breakout-v5",
    "invaders": "ALE/SpaceInvaders-v5",
    "assault": "ALE/Assault-v5", 
    "riverraid": "ALE/Riverraid-v5",
    "beamrider": "ALE/BeamRider-v5",
    "kaboom": "ALE/Kaboom-v5",
    "kungfu": "ALE/KungFuMaster-v5", 
    "seaquest": "ALE/Seaquest-v5"
}

def run_game(args):
    """Handles train/test for the Atari envs."""
    env_map = MAPPING[args.id]
    env = GameEnv(args.seed, env_map, args.nenvs, args.c, args.w, args.verbose)
    try:
        if args.mode == "train":
            env.train(args.o)
        else:
            env.test(args.o, args.neps, True if args.w == None else False)
    except KeyboardInterrupt:
        env.save_weights(args.o)
    finally:
        env.close()

def run_gui(args):
    """Launches the GUI for human or AI play."""
    if args.env == "pong":
        gui = PongGUI(MAPPING[args.env], args.c, args.w, args.mode)
    elif args.env == "breakout":
        gui = BreakoutGUI(MAPPING[args.env], args.c, args.w, args.mode)
    elif args.env == "kaboom":
        gui = InvadersGUI(MAPPING[args.env], args.c, args.w, args.mode)
    else: 
        raise ValueError(f"[ERROR] Unsupported GUI environment: {args.env}")
    
    gui.run()

def main():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for RL env (train/test) and GUI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_game = subparsers.add_parser(
        "game", help="Train or test an Atari RL agent"
    )
    p_game.add_argument(
        "--id", type=str, choices=list(MAPPING), default="pong",
        help="Which Atari game"
    )
    p_game.add_argument(
        "--c", required=True, help="Path to config file"
    )
    p_game.add_argument(
        "--w", help="Path to initial weights (optional)"
    )
    p_game.add_argument(
        "--o", required=True, help="Output path for saving model/weights"
    )
    p_game.add_argument(
        "--seed", type=int, default=1898,
        help="Random seed for reproducibility"
    )
    p_game.add_argument(
        "--nenvs", type=int, default=1,
        help="Number of parallel vectorized environments"
    )
    p_game.add_argument(
        "--mode", choices=["train", "test"], default="train",
        help="Whether to train or to run test episodes"
    )
    p_game.add_argument(
        "--neps", type=int, default=1,
        help="If testing: how many episodes to run"
    )
    p_game.add_argument(
        "--verbose", action="store_true",
        help="Enable rendering/log details"
    )
    p_game.set_defaults(func=run_game)

    p_gui = subparsers.add_parser(
        "gui", help="Launch GUI for human/AI play"
    )
    p_gui.add_argument(
        "--env", type=str, choices=list(MAPPING.keys()),
        default="pong", help="Which game GUI to launch"
    )
    p_gui.add_argument(
        "--c", required=True,
        help="Config path for GUI frame return"
    )
    p_gui.add_argument(
        "--w", help="Weights file for the model"
    )
    p_gui.add_argument(
        "--mode", choices=["AI", "human"], required=True,
        help="Control mode in GUI"
    )
    p_gui.set_defaults(func=run_gui)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
