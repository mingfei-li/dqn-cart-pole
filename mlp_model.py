import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=out_features),
        )
    
    def forward(self, x):
        return self.model(x)