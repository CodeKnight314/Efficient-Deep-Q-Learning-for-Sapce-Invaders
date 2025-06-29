import torch
import time
import matplotlib.pyplot as plt
from src.model import GameModel, EfficientGameModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_benchmark(model_class, model_name, mode, values, fixed_params):
    times = []

    for val in values:
        if mode == "batch":
            batch_size = val
            num_frames = fixed_params["num_frames"]
            res = fixed_params["resolution"]
        elif mode == "frames":
            batch_size = fixed_params["batch_size"]
            num_frames = val
            res = fixed_params["resolution"]
        elif mode == "resolution":
            batch_size = fixed_params["batch_size"]
            num_frames = fixed_params["num_frames"]
            res = (val, val)

        model = model_class(num_frames, action_space=6).to(device)
        model.eval()
        dummy_input = torch.randint(0, 256, (batch_size, num_frames, *res), dtype=torch.uint8).float().to(device)

        with torch.no_grad():
            for _ in range(10):
                model(dummy_input, normalize=True)

        start = time.time()
        with torch.no_grad():
            for _ in range(200):
                model(dummy_input, normalize=True)
        end = time.time()

        avg_time = (end - start) / 20
        times.append(avg_time)
        print(f"{model_name} | {mode.title()}: {val} | Avg Time: {avg_time:.5f} sec")

    return values, times

def plot_results(x_vals, y_vals_game, y_vals_eff, xlabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals_game, marker='o', label='GameModel')
    plt.plot(x_vals, y_vals_eff, marker='s', label='EfficientGameModel')
    plt.xlabel(xlabel)
    plt.ylabel("Average Inference Time (sec)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    batch_sizes = [16 * i for i in range(1, 64)]
    fixed = {"num_frames": 4, "resolution": (84, 84)}
    x_b, y_b_game = run_benchmark(GameModel, "GameModel", "batch", batch_sizes, fixed)
    _, y_b_eff = run_benchmark(EfficientGameModel, "EfficientGameModel", "batch", batch_sizes, fixed)
    plot_results(x_b, y_b_game, y_b_eff, "Batch Size", "Inference Time vs Batch Size", "inference_batch_size.png")

    num_frames_range = list(range(1, 33))
    fixed = {"batch_size": 64, "resolution": (84, 84)}
    x_f, y_f_game = run_benchmark(GameModel, "GameModel", "frames", num_frames_range, fixed)
    _, y_f_eff = run_benchmark(EfficientGameModel, "EfficientGameModel", "frames", num_frames_range, fixed)
    plot_results(x_f, y_f_game, y_f_eff, "Number of Stacked Frames", "Inference Time vs Number of Frames", "inference_num_frames.png")

    resolutions = list([(32 + i*16) for i in range(31)])
    fixed = {"batch_size": 64, "num_frames": 4}
    x_r, y_r_game = run_benchmark(GameModel, "GameModel", "resolution", resolutions, fixed)
    _, y_r_eff = run_benchmark(EfficientGameModel, "EfficientGameModel", "resolution", resolutions, fixed)
    plot_results(x_r, y_r_game, y_r_eff, "Input Image Resolution (HxW)", "Inference Time vs Resolution", "inference_resolution.png")
