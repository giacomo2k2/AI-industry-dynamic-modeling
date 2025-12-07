import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.main(x))

class TrigLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: shape (B, 4) -> (B, 6)
        x_trig = torch.zeros((x.shape[0], 6), device=x.device)
        x_trig[:, 0] = torch.cos(x[:, 0])
        x_trig[:, 1] = torch.sin(x[:, 0])
        x_trig[:, 2] = x[:, 1]
        x_trig[:, 3] = torch.cos(x[:, 2])
        x_trig[:, 4] = torch.sin(x[:, 2])
        x_trig[:, 5] = x[:, 3]
        return x_trig


class BaselineFNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        hidden_dim = params['hidden_dim']

        self.input = nn.Linear(6, hidden_dim)
        self.trig = TrigLayer()
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        # x: shape (B, 4)
        x = self.trig(x)
        x = self.input(x)           # B, C
        x = self.activation(x)      # B, C
        x = self.linear(x)         # B, C
        x = self.activation(x)      # B, C    
        x = self.output(x)          # B, C
        return x



class DeepFNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        hidden_dim = params['hidden_dim']
        n_hidden = params['n_hidden']

        self.input = nn.Linear(6, hidden_dim)
        self.trig = TrigLayer()

        self.activation = nn.ReLU()
        
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_hidden)]
        )
    
        self.output = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = self.trig(x)
        x = self.input(x)
        x = self.activation(x)
        x = self.blocks(x)      
        x = self.output(x)

        return x
    


