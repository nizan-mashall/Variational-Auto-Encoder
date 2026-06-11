import torch
import wandb
from model.model import VAE
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# Configuration
INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4 # karpathy constant

if __name__ == "__main__":

    wandb.init(
        project="VAE-project",
        config={
            "learning_rate": 0.003,
            "epochs": 10,
        }
    )

    dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    wandb.watch(model, log="all", log_freq=100)   # allows to look inside the model wieghts
    loss_fn = nn.BCELoss(reduction='sum')

    for epoch in range(NUM_EPOCHS):
        for i,(x,y) in enumerate(data_loader):
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)
            recon_loss = loss_fn(x_reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({
                "total_loss": loss.item(),
                "reconstruction_loss": recon_loss.item(),
                "kl_loss": kl_loss.item()
            })
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), 'vae_model.pth')
    wandb.finish()