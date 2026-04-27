import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from typing import overload

def save_training_curves(history, save_dir, prefix="best_model"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Train / Val Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_total_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_recon"], label="Train")
    plt.plot(epochs, history["val_recon"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("Train / Val Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_recon_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_kl"], label="Train")
    plt.plot(epochs, history["val_kl"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.title("Train / Val KL Divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_kl_loss.png"), dpi=200)
    plt.close()


def save_kl_per_dim_plot(history, save_dir, split="train", prefix="best_model"):
    key = "train_kl_per_dim" if split == "train" else "val_kl_per_dim"
    kl_hist = torch.stack(history[key])  # (epochs, latent_dim)

    plt.figure(figsize=(8, 5))
    for d in range(kl_hist.shape[1]):
        plt.plot(range(1, kl_hist.shape[0] + 1),
                 kl_hist[:, d].numpy(), label=f"z{d}")

    plt.xlabel("Epoch")
    plt.ylabel("Average KL (nats)")
    plt.title(f"{split.capitalize()} KL per latent dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, f"{prefix}_{split}_kl_per_dim.png"), dpi=200)
    plt.close()


def save_reconstructions(model, dataloader, device, save_path, n=10):
    model.eval()
    x = next(iter(dataloader))
    x = x[:n].to(device)

    with torch.no_grad():
        x_hat, _, _ = model(x)
        x_hat = torch.sigmoid(x_hat)

    x = x.cpu()
    x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Orig" if i == 0 else "")

        axes[1, i].imshow(x_hat[i, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Recon" if i == 0 else "")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def latent_traversal(model, sample, device, traversal_range=(-3, 3), steps=9):
    model.eval()

    if sample.dim() == 3:
        sample = sample.unsqueeze(0)

    sample = sample.to(device)

    with torch.no_grad():
        _, mu, _ = model(sample)
        z = mu.clone()

        latent_dim = z.shape[1]
        values = torch.linspace(
            traversal_range[0], traversal_range[1], steps, device=device)

        rows = []
        for d in range(latent_dim):
            row = []
            for val in values:
                z_mod = z.clone()
                z_mod[0, d] = val
                logits = model.decoder(z_mod)
                recon = torch.sigmoid(logits)
                row.append(recon[0, 0].cpu())
            rows.append(row)

    return rows


def save_latent_traversal(rows, save_path):
    n_rows = len(rows)
    n_cols = len(rows[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].imshow(rows[i][j], cmap="gray")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

@overload
def make_rate_distortion_curve(datapath: str, save_path=None):
    if isinstance(datapath, str):
        parquet_files = [f for f in os.listdir(
            datapath) if f.endswith('.parquet')]

    df = pd.DataFrame()
    for file in parquet_files:
        file_path = os.path.join(datapath, file)
        temp_df = pd.read_parquet(file_path)
        df = pd.concat([df, temp_df])

    plt.figure(figsize=(10, 6))
    for rd in df.groupby('latent_dims'):
        plt.plot(rd[1]['rate'], rd[1]['distortion'],
                label=rd[0], linestyle="--")
    plt.legend(title='Latent\nDimension\nSize', fancybox=True)
    plt.grid()
    plt.title('Rate-Distortion Curve')
    plt.xlabel('Rate')
    plt.ylabel('Distortion')
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

@overload
def make_rate_distortion_curve(df: pd.DataFrame, save_path=None):
    plt.figure(figsize=(10, 6))
    for rd in df.groupby('latent_dims'):
        plt.plot(rd[1]['rate'], rd[1]['distortion'],
                label=rd[0], linestyle="--")
    plt.legend(title='Latent\nDimension\nSize', fancybox=True)
    plt.grid()
    plt.title('Rate-Distortion Curve')
    plt.xlabel('Rate')
    plt.ylabel('Distortion')
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()