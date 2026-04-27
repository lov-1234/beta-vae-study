import os
import random
import numpy as np
import torch

from datasets import DSpritesDataset
from helpers import beta_latent_sweep, make_splits_and_loaders
from losses import vae_loss
from models import VAE
from plotters import make_rate_distortion_curve

PLOT_PATH = os.path.join(os.getcwd(), 'plots')
BASE_DIR = os.path.join(PLOT_PATH, 'beta_disentanglement')

DIRS = {
    "root": BASE_DIR,
    "search": os.path.join(BASE_DIR, "hyperparam_search"),
    "checkpoints": os.path.join(BASE_DIR, "checkpoints"),
    "curves": os.path.join(BASE_DIR, "training_curves"),
    "recon": os.path.join(BASE_DIR, "reconstructions"),
    "traversal": os.path.join(BASE_DIR, "latent_traversals"),
    "kl": os.path.join(BASE_DIR, "kl_analysis"),
    "test": os.path.join(BASE_DIR, "test_results"),
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)

print("Created:", BASE_DIR)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# D-Sprites Dataset

dsprites_data_path = os.path.join(
    os.getcwd(), 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
dsprites_data = np.load(dsprites_data_path)
print("D-Sprites dataset loaded from:", dsprites_data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_kwargs = {
    "data_path": dsprites_data_path,
    "device": device
}
train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = make_splits_and_loaders(
    dataset_cls=DSpritesDataset,
    dataset_kwargs=dataset_kwargs,
    batch_size=512,
    seed=42,
    num_workers=1,
    pin_memory=False
)

train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)
test_dataset_size = len(test_dataset)

beta_sweep_values = np.arange(1, 20, 1)
latent_dim_values = np.arange(2, 11, 1)

sweep_df = beta_latent_sweep(
    model_cls=VAE,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device=device,
    epochs=100,
    loss_fn=vae_loss,
    latent_dims=latent_dim_values,
    beta_values=beta_sweep_values,
    save_folder=os.path.join(DIRS["curves"], "beta_latent_sweep"),
)

make_rate_distortion_curve(sweep_df, os.path.join(DIRS["kl"], "rd.png"))