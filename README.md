# Beta-VAE Disentanglement Study: Latent Dimension Capacity and Rate-Distortion Analysis

This repository implements a study on the capacity of latent dimensions in beta-Variational Autoencoders (beta-VAE) for disentangled representation learning. We investigate whether the Rate-Distortion (R-D) curve saturates when latent dimensions exceed the intrinsic dimensionality of the data, and explore the theoretical limits imposed by Gaussian priors/channels.

## Background

### Beta-VAE for Disentanglement

Beta-VAE is a variant of Variational Autoencoders (VAEs) that encourages disentangled latent representations by scaling the KL divergence term in the loss function:

```
L = E[log p(x|z)] + β * KL(q(z|x) || p(z))
```

Where:
- `E[log p(x|z)]` is the reconstruction loss (distortion)
- `KL(q(z|x) || p(z))` is the KL divergence between the posterior and prior (rate)
- `β > 1` encourages the model to learn more disentangled representations

### Rate-Distortion Theory

In information theory, the Rate-Distortion function R(D) represents the minimum rate (bits) needed to represent a source with distortion D. For VAEs, we can interpret:
- **Rate (R)**: Average KL divergence between learned posterior and prior
- **Distortion (D)**: Reconstruction error

The R-D curve shows the trade-off between compression (rate) and fidelity (distortion).

## Experiment: Latent Dimension Capacity

We investigate how the R-D curve behaves when varying the latent dimension size beyond the intrinsic dimensionality of the data. The dSprites dataset has an intrinsic dimension of approximately 5 (shape, scale, orientation, position x, position y), with color being constant.

### Key Findings

1. **Diminishing Returns**: Beyond a certain latent dimension threshold, increasing the number of latent dimensions yields diminishing returns in terms of reconstruction quality.

2. **Saturation**: The distortion (reconstruction loss) stabilizes and no longer improves significantly when latent dimensions exceed the intrinsic dimension.

3. **Rate Stagnation**: The rate (KL divergence) also stagnates, indicating that excess latent dimensions are not being utilized effectively.

### Theoretical Explanation: Finite Capacity of Gaussian Channels

Each latent dimension in a VAE can be viewed as a Gaussian communication channel. In information theory, a Gaussian channel with noise variance σ² has finite capacity:

```
C = (1/2) log₂(1 + SNR)
```

Where SNR is the signal-to-noise ratio.

In beta-VAE:
- The prior p(z) is a standard Gaussian N(0, I)
- The posterior q(z|x) is learned as N(μ(x), Σ(x))
- The "channel" is the mapping from prior samples to data reconstructions

Since each dimension is independent and Gaussian, each has finite information capacity. When latent dimensions exceed the intrinsic dimension of the data, the excess dimensions cannot carry meaningful information about the data distribution, leading to:
- Unused capacity (KL divergence doesn't increase)
- No improvement in reconstruction (distortion doesn't decrease)

This explains why we observe saturation in both rate and distortion metrics.

## Implementation Details

### Model Architecture

- **Encoder**: Convolutional neural network that maps images to latent distributions
- **Decoder**: Transposed convolutional network for reconstruction
- **Latent Space**: Diagonal Gaussian with reparameterization trick

### Losses

1. **Reconstruction Loss**: Binary Cross-Entropy with logits
   ```
   L_recon = BCEWithLogitsLoss(x_hat, x)
   ```

2. **KL Divergence Loss**: Standard VAE KL term
   ```
   L_KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
   ```

3. **Total VAE Loss**:
   ```
   L_total = L_recon + β * L_KL
   ```

### Dataset

We use the dSprites dataset, which consists of 737,280 images of 2D shapes with 6 ground truth generative factors:
- Color: 1 value (constant)
- Shape: 3 values (square, ellipse, heart)
- Scale: 6 values
- Orientation: 40 values
- Position X: 32 values
- Position Y: 32 values

Effective intrinsic dimension: 5 (excluding constant color factor).

## Usage

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib pandas tqdm
```

### Running the Experiment

```bash
python main.py
```

This will:
1. Load the dSprites dataset
2. Perform a sweep over latent dimensions (2-10) and beta values (1-19)
3. Train beta-VAE models for each combination
4. Generate Rate-Distortion curves

### Key Parameters

- `latent_dim_values`: Range of latent dimensions to test
- `beta_sweep_values`: Range of beta values for disentanglement control
- `epochs`: Training epochs per model (default: 100)

## Results Interpretation

The generated plots show:
- **Rate-Distortion Curves**: Trade-off between KL divergence (rate) and reconstruction loss (distortion) for different latent dimensions
- **Training Curves**: Loss convergence for each model configuration
- **Reconstructions**: Visual comparison of original vs reconstructed images

Expected behavior:
- For latent dims ≤ intrinsic dim (5): R-D curve improves with more dimensions
- For latent dims > intrinsic dim: R-D curve saturates, showing no significant improvement

## Files Overview

- `main.py`: Main experiment script
- `models.py`: VAE architecture definition
- `losses.py`: Loss functions
- `helpers.py`: Training utilities and hyperparameter sweeps
- `plotters.py`: Visualization functions
- `datasets.py`: Data loading utilities
- `optuna_helpers.py`: Hyperparameter optimization (if used)

## Citation

If you use this code in your research, please cite the relevant beta-VAE papers and information theory references on Rate-Distortion theory.