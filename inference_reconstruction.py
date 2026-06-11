import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from model.model import VAE
from train import INPUT_DIM, HIDDEN_DIM, LATENT_DIM, DEVICE
trained_model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
trained_model.load_state_dict(torch.load('vae_model.pth'))
trained_model.eval()

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

def compare_inference(digit: int):
    # Find the image
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        if label == digit:
            x = img.to(DEVICE).view(1, INPUT_DIM)
            original_img = img.squeeze().numpy() # Keep the original 28x28 array
            break

    # Run through the VAE
    with torch.no_grad():
        x_reconstructed, _, _ = trained_model(x)
    reconstructed_img = x_reconstructed.view(28, 28).cpu().numpy()

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title("VAE Reconstruction")
    axes[1].axis('off')

    plt.show()

compare_inference(3)