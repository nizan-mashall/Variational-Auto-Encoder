import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from model.model import VAE
import numpy as np
from train import INPUT_DIM, HIDDEN_DIM, LATENT_DIM, DEVICE

trained_model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
trained_model.load_state_dict(torch.load('vae_model.pth'))
trained_model.eval()

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

def interpolate_digits(digit_A: int, digit_B: int, num_steps: int = 8):
    img_A, img_B = None, None

    # 1. Find one example of digit_A and one of digit_B in the dataset
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        if label == digit_A and img_A is None:
            img_A = img.to(DEVICE).view(1, INPUT_DIM)
        if label == digit_B and img_B is None:
            img_B = img.to(DEVICE).view(1, INPUT_DIM)
        if img_A is not None and img_B is not None:
            break

    # 2. Extract their specific coordinates in the latent space
    with torch.no_grad():
        _, mu_A, _ = trained_model(img_A)
        _, mu_B, _ = trained_model(img_B)

        # Create an array of weights from 0.0 (fully Digit A) to 1.0 (fully Digit B)
        alphas = np.linspace(0, 1, num_steps)

        # Setup the matplotlib row grid
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))

        # 3. Blend them step-by-step
        for i, alpha in enumerate(alphas):
            # Linear Interpolation formula: (1 - alpha) * A + alpha * B
            z_interp = (1 - alpha) * mu_A + alpha * mu_B

            # 4. Decode the blended vector into a brand new image
            out = trained_model.decode(z_interp)
            out = out.view(28, 28).cpu().numpy()

            # 5. Display the result
            axes[i].imshow(out, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"{int((1-alpha)*100)}% {digit_A}\n{int(alpha*100)}% {digit_B}", fontsize=9)

        plt.show()

# Run the function to see the morphing steps between 3 and 4
interpolate_digits(3, 8, num_steps=8)