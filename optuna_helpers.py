import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models import VAE
from helpers import EarlyStopping, make_splits_and_loaders, train_pipeline
from losses import vae_loss
import numpy as np
import optuna
from main import set_seed

def get_search_space(trial):
    space = {
        "beta": trial.suggest_float("beta", 1.0, 50.0),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True),
        "scheduler": trial.suggest_categorical("scheduler", ["cosinewarm"]),
    }

    if space["scheduler"] == "cosinewarm":
        space["T_0"] = trial.suggest_int("T_0", 3, 10)
        space["T_mult"] = trial.suggest_int("T_mult", 1, 3)
        space["eta_min"] = trial.suggest_float("eta_min", 1e-6, 1e-4, log=True)

    return space


def make_objective(dataset_cls, dataset_kwargs, device, search_epochs=20, seed=42, num_workers=4):
    def objective(trial):
        torch.cuda.empty_cache()
        set_seed(seed)

        cfg = get_search_space(trial)

        _, _, _, train_loader, val_loader, _ = make_splits_and_loaders(
            dataset_cls=dataset_cls,
            dataset_kwargs=dataset_kwargs,
            batch_size=256,
            seed=seed,
            num_workers=num_workers
        )

        model = VAE(input_channels=1, latent_dim=5).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )

        scheduler = None
        scheduler_step_per_batch = True

        if cfg["scheduler"] == "cosinewarm":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg["T_0"],
                T_mult=cfg["T_mult"],
                eta_min=cfg["eta_min"]
            )

        early_stopping = EarlyStopping(
            patience=8,
            min_delta=1e-4,
            mode="min"
        )

        history = train_pipeline(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=search_epochs,
            loss_fn=vae_loss,
            beta=cfg["beta"],
            scheduler=scheduler,
            early_stopping=early_stopping,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )

        best_val_loss = min(history["val_loss"])
        best_epoch = int(np.argmin(history["val_loss"]) + 1)

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_val_kl", float(
            history["val_kl"][best_epoch - 1]))
        trial.set_user_attr("best_val_recon", float(
            history["val_recon"][best_epoch - 1]))

        return best_val_loss

    return objective

def run_optuna_study(dataset_cls, dataset_kwargs, device, n_trials=20, search_epochs=20, seed=42, num_workers=4):
    study = optuna.create_study(direction="minimize")
    objective = make_objective(dataset_cls, dataset_kwargs, device, search_epochs, seed, num_workers)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study

def get_best_hyperparams_from_study(study):
    best_trial = study.best_trial
    best_params = best_trial.params
    best_epoch = best_trial.user_attrs["best_epoch"]
    best_val_kl = best_trial.user_attrs["best_val_kl"]
    best_val_recon = best_trial.user_attrs["best_val_recon"]

    return {
        "best_params": best_params,
        "best_epoch": best_epoch,
        "best_val_kl": best_val_kl,
        "best_val_recon": best_val_recon
    }