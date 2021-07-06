import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, out_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(out_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, out):
        validity = self.model(out)
        return validity