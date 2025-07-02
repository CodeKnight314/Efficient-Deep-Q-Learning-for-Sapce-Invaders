import torch
import torch.nn as nn 

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

class EfficientGameModel(nn.Module):
    def __init__(self, num_frames: int, action_space: int):
        super().__init__()

        self.cnn = nn.Sequential(
            DepthwiseSeparableConv2d(num_frames, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),

            DepthwiseSeparableConv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            DepthwiseSeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )

        feature_size = 64 * 7 * 7

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

        elif isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels:
                nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                        nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        feats = self.cnn(x)
        value = self.value_stream(feats)
        advantage = self.advantage_stream(feats)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
    
    def reset(self): 
        self.value_stream.apply(self._init_weights)
        self.advantage_stream.apply(self._init_weights)

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

class GameModel(nn.Module):
    def __init__(self, num_frames: int, action_space: int):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4, padding=2), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        
        feature_size = 64 * 7 * 7
        
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512), 
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512), 
            nn.ReLU(), 
            nn.Linear(512, action_space)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        board_embeddings = self.cnn(x)
        value = self.value_stream(board_embeddings)
        advantage = self.advantage_stream(board_embeddings)
        
        Q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q_values
    
    def reset(self): 
        self.value_stream.apply(self._init_weights)
        self.advantage_stream.apply(self._init_weights)
    
    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cuda' if torch.cuda.is_available() else "cpu"))
        
    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)
        
if __name__ == "__main__":
    import time 
    from ptflops import get_model_complexity_info

    num_frames = 4
    ac_dim = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_a = GameModel(num_frames, ac_dim).to(DEVICE)
    model_b = EfficientGameModel(num_frames, ac_dim).to(DEVICE)
    
    dummy_input = torch.randn(512, 4, 84, 84).to(DEVICE)

    for _ in range(10):
        _ = model_a(dummy_input, normalize=True)
        _ = model_b(dummy_input, normalize=True)

    def benchmark(model, inputs, runs=250):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(inputs, normalize=True)
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) / runs

    time_a = benchmark(model_a, dummy_input)
    time_b = benchmark(model_b, dummy_input)

    print(f"Average inference time per forward pass (512 samples):")
    
    macs_a, params_a = get_model_complexity_info(
        GameModel(4, 10), (4, 84, 84),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )


    macs_b, params_b = get_model_complexity_info(
        EfficientGameModel(4, 10), (4, 84, 84),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"  GameModel:            {time_a*1000:.3f} ms")
    print(f"  GameModel FLOPs: {macs_a * 2 / 1e6:.2f} MFLOPs, Params: {params_a / 1e6:.2f} M")
    print("")
    print(f"  EfficientGameModel:   {time_b*1000:.3f} ms")
    print(f"  EfficientModel FLOPs: {macs_b * 2 / 1e6:.2f} MFLOPs, Params: {params_b / 1e6:.2f} M")