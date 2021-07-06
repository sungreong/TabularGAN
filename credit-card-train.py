import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt

data = pd.read_csv("./dataset/creditcard.csv")
data.drop(["Time", "Class"],axis=1, inplace= True)
cuda = True if torch.cuda.is_available() else False

from sklearn.preprocessing import MinMaxScaler as mms
num_scaler = mms(feature_range=(-1,1))
columns = data.columns.tolist()
data[columns] = num_scaler.fit_transform(data[columns])
data_np = data.values

class TabularDataModule(pl.LightningDataModule) :
    def __init__(self, data , batch_size:int = 32 , num_workers:int=3) :
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = self.data.shape[1]
    
    def prepare_data(self,) :
        pass

    def setup(self,stage=None) :
        if stage == "fit" or state is None :
            train_length = int(len(self.data)*0.8)
            lengths = [train_length, int(len(self.data)-train_length)]
            self.train, self.val = random_split(self.data, lengths)
        
        if stage == "test" or stage is None :
            self.test = self.data   


    def train_dataloader(self):
        return DataLoader(self.train , batch_size= self.batch_size , num_workers=self.num_workers )

    def valid_dataloader(self):
        return DataLoader(self.val , batch_size= self.batch_size , num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test , batch_size= self.batch_size , num_workers=self.num_workers)

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
    
class GAN(pl.LightningModule): 
    def __init__(self,
                input_dim = None,
                scaler = None,
                latent_dim =100 , 
                lr: float = 0.0002,
                b1: float = 0.5,
                b2: float = 0.999,
                batch_size: int = 64,
                **kwargs) :
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim,
        out_shape=self.hparams.input_dim,scaler=scaler)

        self.discriminator = Discriminator(out_shape=self.hparams.input_dim)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)
    
    def inference(self,z) :
        return self.generator.inference(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim)
        z = z.type_as(x)
        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(x)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(x.size(0), 1)
            valid = valid.type_as(x)

            real_loss = self.adversarial_loss(self.discriminator(x), valid)

            # how well can it label as fake?
            fake = torch.zeros(x.size(0), 1)
            fake = fake.type_as(x)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

dm = TabularDataModule(data_np.astype(np.float32))
model = GAN(dm.size(),num_scaler)
trainer = pl.Trainer(gpus=0, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
model.eval()
z = torch.randn(1, model.hparams.latent_dim)
model.inference(z)