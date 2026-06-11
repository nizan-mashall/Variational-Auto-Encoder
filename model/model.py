import torch
from torch import nn
from torch.nn import functional as F
import wandb

# input image -> linear to hidden_dim -> mean and variation -> para trick -> Decoder -> output image
class VAE(nn.Module):
  def __init__(self, input_dim = 784, hidden_dim = 200, latent_dim = 20):
    super().__init__()
    #encoder
    self.image_2hid = nn.Linear(input_dim, hidden_dim)
    self.hid_mu = nn.Linear(hidden_dim, latent_dim)
    self.hid_sigma = nn.Linear(hidden_dim, latent_dim)
    #decoder
    self.latent_2hid = nn.Linear(latent_dim, hidden_dim)
    self.hid_2image = nn.Linear(hidden_dim, input_dim)

  def encode(self,x): #p(z|x)
    h = F.relu(self.image_2hid(x))
    mu, sigma = self.hid_mu(h), self.hid_sigma(h)
    return mu, sigma

  def decode(self,z): #p(x|z)
    h= F.relu(self.latent_2hid(z))
    h = self.hid_2image(h)
    return torch.sigmoid(h)

  def forward(self,x):
    mu,sigma = self.encode(x)
    epsilon = torch.randn_like(sigma)
    z_reparameterized = mu + sigma*epsilon
    x_reconstructed = self.decode(z_reparameterized)
    return x_reconstructed, mu, sigma

# if __name__ == '__main__':
#   x = torch.randn(4, 784)
#
#   x_reconstructed, mu, sigma = model(x)
#   print(x_reconstructed.shape)
#   print(mu.shape)
#   print(sigma.shape)

