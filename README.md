# Variational-Auto-Encoder (VAE)

A modular PyTorch implementation of a Variational Autoencoder (VAE). This repository contains the core components for training and evaluating a VAE model, including standard loss formulations, data input pipelines, and latent space visualization tools.

## Project Overview

Unlike standard autoencoders that map inputs to fixed bottleneck vectors, a VAE models the underlying latent distribution of the input data. By optimizing a probabilistic framework, the network learns a continuous, structured latent space suitable for data sampling and generation.

## Mathematical Motivation

In Bayesian modeling, we want to find the true posterior distribution of the latent characteristics $z$ given our input data $x$:

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

Where the denominator $p(x) = \int p(x|z)p(z)dz$ is the **evidence** (marginal likelihood). 

* In simpler discrete classification problems (like Naive Bayes), computing this denominator can become combinatorially intractable for large datasets, usually requiring a naive independence assumption ($\prod p(x_i|c)$).
* In complex generative tasks with continuous latent spaces, computing this integral directly is completely impossible because we cannot evaluate all infinite configurations of $z$.

**The VAE Approach:** Instead of calculating $p(x)$ directly, VAEs introduce an inference network $q_\phi(z|x)$ to approximate the true intractable posterior. We then maximize the **Evidence Lower Bound (ELBO)**, which acts as a mathematically tractable proxy for maximizing the true data evidence.

## Architecture

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
python -m venv venv
venv\Scripts\activate
```

3. **Install the dependencies:**
```bash
pip install -r requirements.txt
```