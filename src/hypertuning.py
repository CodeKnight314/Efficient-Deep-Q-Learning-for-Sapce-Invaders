import optuna
from src.env import GameEnv
import yaml
import os
import argparse

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

def train_agent_with_trial(trial, output_dir, env_id):
    learning_rate = trial.suggest_categorical("learning_rate", [1.6e-4, 3.2e-4, 6.4e-4])
    target_update = trial.suggest_int("target_update_freq", 1000, 8000, step=500)

    base_config_path = "src/config/base_config.yaml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    config["lr"] = learning_rate
    config["min_lr"] = learning_rate
    config["batch_size"] = 128
    config["update_freq"] = target_update
    config["max_frames"] = 750_000
    config["epsilon_decay"] = 0.9977

    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    trial_config_path = os.path.join(trial_dir, "config.yaml")
    with open(trial_config_path, "w") as f:
        yaml.dump(config, f)

    env = GameEnv(seed=1898, env_id=env_id, num_envs=128, config=trial_config_path, verbose=False)
    avg_reward = env.train(trial_dir)
    env.close()

    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument("--o", type=str, required=True, help="Path to store Optuna trial outputs")
    parser.add_argument("--n_trials", type=int, default=45, help="Number of trials to run")
    parser.add_argument("--game", type=str, required=True, choices=list(MAPPING.keys()), help="Game name to tune for")

    args = parser.parse_args()
    os.makedirs(args.o, exist_ok=True)

    env_id = MAPPING[args.game]

    def objective(trial):
        return train_agent_with_trial(trial, output_dir=args.o, env_id=env_id)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_path = os.path.join(args.o, "best_config.yaml")
    with open(best_path, "w") as f:
        yaml.dump(study.best_params, f)

    print("Best trial:")
    print("Reward:", study.best_value)
    print("Params:", study.best_params)

if __name__ == "__main__":
    main()
