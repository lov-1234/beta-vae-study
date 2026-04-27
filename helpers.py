import copy
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from losses import vae_loss, constrained_capacity_loss, capacity_schedule
import pandas as pd
import numpy as np
import os
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False


def train_one_epoch(model, dataloader, optimizer, device, loss_fn=vae_loss, beta=4, scheduler=None, global_step=0, total_ramp_steps=100_000):
    model.train()
    total_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    total_kl_per_dim = None
    samples = 0
    batch_step = 0

    for x in tqdm(dataloader, desc="Training"):
        x = x.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)

        if loss_fn is vae_loss:
            loss, recon, kl_div = loss_fn(x, x_hat, mu, logvar, beta=beta)
        elif loss_fn is constrained_capacity_loss:
            C = capacity_schedule(global_step, c_max=25.0,
                                  c_stop_iter=total_ramp_steps)
            loss, recon, kl_div = constrained_capacity_loss(
                x, x_hat, mu, logvar, C=C, gamma=1000.0)
        else:
            raise ValueError("Unsupported loss function")

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_size
        total_kl_loss += kl_div.item() * batch_size
        total_recon_loss += recon.item() * batch_size

        kl_div_per_dim_batch = -0.5 * \
            (1 + logvar - mu.pow(2) - logvar.exp())   # (B, D)
        kl_div_per_dim_batch = kl_div_per_dim_batch.mean(
            dim=0)                  # (D,)

        if total_kl_per_dim is None:
            total_kl_per_dim = kl_div_per_dim_batch.detach() * batch_size
        else:
            total_kl_per_dim += kl_div_per_dim_batch.detach() * batch_size

        samples += batch_size
        batch_step += 1

    avg_loss = total_loss / samples
    avg_kl_loss = total_kl_loss / samples
    avg_recon_loss = total_recon_loss / samples
    avg_kl_per_dim = total_kl_per_dim / samples

    if loss_fn is vae_loss:
        return avg_loss, avg_kl_loss, avg_recon_loss, avg_kl_per_dim.cpu()
    return avg_loss, avg_kl_loss, avg_recon_loss, avg_kl_per_dim.cpu(), batch_step


def validate(model, dataloader, device, loss_fn=vae_loss, beta=4, desc='Validation', global_step=0, total_ramp_steps=100_000):
    model.eval()
    total_loss = 0.0
    total_kl_loss = 0.0
    total_recon_loss = 0.0
    total_kl_per_dim = None
    samples = 0

    with torch.no_grad():
        for x in tqdm(dataloader, desc=desc):
            x = x.to(device)
            batch_size = x.size(0)
            x_hat, mu, logvar = model(x)

            if loss_fn is vae_loss:
                loss, recon, kl_div = loss_fn(x, x_hat, mu, logvar, beta=beta)
            elif loss_fn is constrained_capacity_loss:
                C = capacity_schedule(
                    global_step, c_max=25.0, c_stop_iter=total_ramp_steps)
                loss, recon, kl_div = loss_fn(
                    x, x_hat, mu, logvar, C=C, gamma=1000.0)
            else:
                raise ValueError("Unsupported loss function")

            total_loss += loss.item() * batch_size
            total_kl_loss += kl_div.item() * batch_size
            total_recon_loss += recon.item() * batch_size

            kl_div_per_dim_batch = -0.5 * \
                (1 + logvar - mu.pow(2) - logvar.exp())
            kl_div_per_dim_batch = kl_div_per_dim_batch.mean(dim=0)

            if total_kl_per_dim is None:
                total_kl_per_dim = kl_div_per_dim_batch.detach() * batch_size
            else:
                total_kl_per_dim += kl_div_per_dim_batch.detach() * batch_size

            samples += batch_size

    avg_loss = total_loss / samples
    avg_kl_loss = total_kl_loss / samples
    avg_recon_loss = total_recon_loss / samples
    avg_kl_per_dim = total_kl_per_dim / samples

    return avg_loss, avg_kl_loss, avg_recon_loss, avg_kl_per_dim.cpu()


def train_pipeline(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    loss_fn=vae_loss,
    beta=4,
    scheduler=None,
    early_stopping=None,
    scheduler_step_per_batch=True,
    total_ramp_steps=100_000
):
    train_loss_history = []
    train_kl_div_loss_history = []
    train_recon_loss_history = []
    train_kl_div_loss_history_per_dim = []

    val_loss_history = []
    val_kl_div_loss_history = []
    val_recon_loss_history = []
    val_kl_div_loss_history_per_dim = []
    global_step = 0

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")

        if loss_fn is vae_loss:
            epoch_loss, epoch_kl_loss, epoch_recon_loss, epoch_kl_loss_per_dim = train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                loss_fn=loss_fn,
                beta=beta,
                scheduler=scheduler if scheduler_step_per_batch else None
            )
        else:
            epoch_loss, epoch_kl_loss, epoch_recon_loss, epoch_kl_loss_per_dim, batch_step = train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                loss_fn=loss_fn,
                beta=beta,
                scheduler=scheduler if scheduler_step_per_batch else None,
                global_step=global_step,
                total_ramp_steps=total_ramp_steps
            )
            global_step += batch_step

        train_loss_history.append(epoch_loss)
        train_kl_div_loss_history.append(epoch_kl_loss)
        train_recon_loss_history.append(epoch_recon_loss)
        train_kl_div_loss_history_per_dim.append(epoch_kl_loss_per_dim)

        val_epoch_loss, val_epoch_kl_loss, val_epoch_recon_loss, val_epoch_kl_loss_per_dim = validate(
            model,
            val_dataloader,
            device,
            loss_fn=loss_fn,
            beta=beta,
            global_step=global_step,
            total_ramp_steps=total_ramp_steps
        )

        val_loss_history.append(val_epoch_loss)
        val_kl_div_loss_history.append(val_epoch_kl_loss)
        val_recon_loss_history.append(val_epoch_recon_loss)
        val_kl_div_loss_history_per_dim.append(val_epoch_kl_loss_per_dim)

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        print(
            f"Train - Loss: {epoch_loss:.4f}, Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.6f}\n"
            f"Val   - Loss: {val_epoch_loss:.4f}, Recon: {val_epoch_recon_loss:.4f}, KL: {val_epoch_kl_loss:.6f}"
        )

        if early_stopping is not None:
            early_stopping.step(val_epoch_loss, model)
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if early_stopping is not None and early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    return {
        "train_loss": train_loss_history,
        "train_kl": train_kl_div_loss_history,
        "train_recon": train_recon_loss_history,
        "train_kl_per_dim": train_kl_div_loss_history_per_dim,
        "val_loss": val_loss_history,
        "val_kl": val_kl_div_loss_history,
        "val_recon": val_recon_loss_history,
        "val_kl_per_dim": val_kl_div_loss_history_per_dim,
    }


def make_splits_and_loaders(dataset_cls, dataset_kwargs, batch_size=256, seed=42, num_workers=4, pin_memory=True):
    full_dataset = dataset_cls(**dataset_kwargs)

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def train_one_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs,
    loss_fn=vae_loss,
    beta=4,
    scheduler=None,
    early_stopping=None,
    scheduler_step_per_batch=True,
    total_ramp_steps=100_000
):
    history = train_pipeline(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        loss_fn=loss_fn,
        beta=beta,
        scheduler=scheduler,
        early_stopping=early_stopping,
        scheduler_step_per_batch=scheduler_step_per_batch,
        total_ramp_steps=total_ramp_steps
    )

    return history

def test_model(model, test_dataloader, device, loss_fn=vae_loss, beta=4):
    test_loss, test_kl_loss, test_recon_loss, test_kl_per_dim = validate(
        model,
        test_dataloader,
        device,
        loss_fn=loss_fn,
        beta=beta
    )
    return {
        "test_loss": test_loss,
        "test_kl": test_kl_loss,
        "test_recon": test_recon_loss,
        "test_kl_per_dim": test_kl_per_dim
    }


def beta_latent_sweep(model_cls,
                      train_dataloader,
                      val_dataloader,
                      device,
                      epochs,
                      loss_fn=vae_loss,
                      scheduler=None,
                      early_stopping=None,
                      scheduler_step_per_batch=True,
                      total_ramp_steps=100_000, 
                      beta_values=[1, 4, 10, 25],
                      latent_dims=[5, 10, 20],
                      save_folder=os.getcwd()):
    distortion_list = []
    rate_list = []
    latent_dim_list = []
    beta_list = []
    for latent_dim in latent_dims:
        for beta in beta_values:
            print(f"Training with latent_dim={latent_dim}, beta={beta}")
            model = model_cls(input_channels=1, latent_dim=latent_dim, kernel_size=4, output_padding=0).to(device)
            optimizer = optim.Adam(model.parameters(), lr=5e-4)

            history = train_pipeline(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                device=device,
                epochs=epochs,
                loss_fn=loss_fn,
                beta=beta,
                scheduler=scheduler,
                early_stopping=early_stopping,
                scheduler_step_per_batch=scheduler_step_per_batch,
                total_ramp_steps=total_ramp_steps
            )

            lowest_elbo_idx = np.argmin(history["val_loss"])
            best_recon = history["val_recon"][lowest_elbo_idx]
            best_kl = history["val_kl"][lowest_elbo_idx]    
            distortion_list.append(best_recon)
            rate_list.append(best_kl)
            latent_dim_list.append(latent_dim)
            beta_list.append(beta)
        
        if latent_dim % 2 == 0:
            print(f"Finished beta sweep for latent_dim={latent_dim}")
            print('Saving intermediate results...')
            df = pd.DataFrame({
                "latent_dim": latent_dim_list,
                "beta": beta_list,
                "rate": rate_list,
                "distortion": distortion_list
            })
            df.to_csv(os.path.join(save_folder, f"beta_sweep_results_latent_dim_{latent_dim}.csv"), index=False)
            print(f"Saved intermediate results for latent_dim={latent_dim}")
    return pd.DataFrame({
        "latent_dim": latent_dim_list,
        "beta": beta_list,
        "rate": rate_list,
        "distortion": distortion_list
    })
            
