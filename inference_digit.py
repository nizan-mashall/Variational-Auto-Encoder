from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets 
from model import VAE
from train import INPUT_DIM, HIDDEN_DIM, LATENT_DIM, DEVICE
from torch.utils.data import DataLoader 

trained_model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
trained_model.load_state_dict(torch.load('C:\\Users\\nizan\\Desktop\\Work\\SAIL_Lab\\VAE\\vae_model.pth'))
trained_model.eval()

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

def inference(digit: int, num_samples: int = 1):
  for idx in range(len(test_dataset)):
    img, label = test_dataset[idx]
    if label == digit:
      x = img.to(DEVICE).view(1, INPUT_DIM)
      break

  with torch.no_grad():
    x_reconstructed, mu, sigma = trained_model(x)

    epsilon = torch.randn_like(sigma)
    z_reparameterized = mu + sigma*epsilon
    out = trained_model.decode(z_reparameterized)
    out = out.view(28,28)
    return out.cpu().detach().numpy()


three = inference(3)
plt.imshow(three, cmap='gray')
plt.axis('off')
plt.show()

