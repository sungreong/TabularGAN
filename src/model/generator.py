import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, out_shape, scaler):
        super().__init__()
        self.out_shape = out_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.out_shape),
            nn.Tanh()
        )
        self.scaler = scaler 

    def forward(self, z):
        x = self.model(z)
        # img = img.view(img.size(0), *self.out_shape)
        return x

    def inference(self,z) :
        x = self.model(z).detach().numpy()
        x = self.scaler.inverse_transform(x)
        return x