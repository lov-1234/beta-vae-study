import torch
import torch.nn as nn

recon_loss = nn.BCEWithLogitsLoss(reduction='sum')


def kl_div_loss(mu, logvar): return -0.5 * \
    torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(x, x_hat, mu, logvar, beta=4):
    recon = recon_loss(x_hat, x)
    kl_div = kl_div_loss(mu, logvar)
    # Normalize by batch size
    return (recon + beta * kl_div) / x.size(0), recon / x.size(0), kl_div / x.size(0)


def capacity_schedule(step, c_max=25.0, c_stop_iter=100_000):
    return min(c_max, c_max * step / c_stop_iter)


def constrained_capacity_loss(x, x_hat, mu, logvar, C=25.0, gamma=1000.0):
    batch_size = x.size(0)

    recon = recon_loss(x_hat, x) / batch_size
    kl_div = kl_div_loss(mu, logvar) / batch_size

    loss = recon + gamma * torch.abs(kl_div - C)
    return loss, recon, kl_div
