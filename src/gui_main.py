from src.gui.gui_pong import PongGUI
import argparse

def main(args):
    if args.env == "pong":
        gui = PongGUI(args.c, args.w, args.mode)
        gui.run()
    else:
        raise ValueError("[ERROR] Either no environment id was given or the given id was incorrect")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Handle for GUI control/render for RL agents")
    parser.add_argument("--env", type=str, default="pong", choices=["pong","invaders","breakout"], help="Environment choices")
    parser.add_argument("--c", type=str, help="Environment Config for returning frames")
    parser.add_argument("--w", type=str, help="Weights for model")
    parser.add_argument("--mode", type=str, choices=["AI", "human"], help="Mode for GUI control")
    
    args = parser.parse_args()
    
    main(args)

