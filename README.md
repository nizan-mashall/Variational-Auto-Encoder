# Variational-Auto-Encoder (VAE)

A modular PyTorch implementation of a Variational Autoencoder (VAE). This repository contains the core components for training and evaluating a VAE model, including standard loss formulations, data input pipelines, and latent space visualization tools.

## Project Overview

Unlike standard autoencoders that map inputs to fixed bottleneck vectors, a VAE models the underlying latent distribution of the input data. By optimizing a probabilistic framework, the network learns a continuous, structured latent space suitable for data sampling and generation.

## Architecture & Mathematics

The model optimizes the **Evidence Lower Bound (ELBO)** to maximize the marginal log-likelihood of the data:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))$$

1. **Encoder ($q_\phi$):** Parameterizes the variational posterior distribution, mapping inputs to mean ($\mu$) and log-variance ($\log \sigma^2$) vectors.
2. **Latent Sampling ($z$):** Samples from the latent space using $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
3. **Decoder ($p_\theta$):** Reconstructs the input data from the latent representation $z$.

## Installation & Setup

1. **Clone the Repository:**

```bash
git clone https://github.com/nizan-mashall/Variational-Auto-Encoder.git
cd Variational-Auto-Encoder 
   ```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv\Scripts\activate
```

3. **Install the dependencies:**
```bash
pip install -r requirements.txt
```