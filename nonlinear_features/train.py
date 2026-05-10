"""Training loop for TopK SAEs."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .sae import TopKSAE


def train_sae(
    data: torch.Tensor,
    d: int = 128,
    c: int = 512,
    k: int = 4,
    lr: float = 3e-3,
    batch_size: int = 1024,
    epochs: int = 10,
    device: str = "cpu",
    log_every: int = 100,
    loss_fn: str = "l1",
    aux_weight: float = 0.05,
    aux_warmup_epochs: int = 2,
) -> tuple[TopKSAE, dict]:
    """Train a TopK SAE on the given data.

    Args:
        loss_fn: "l1" for L1 reconstruction loss (as in paper), "mse" for MSE.
        aux_weight: weight for dead neuron reanimation auxiliary loss.
        aux_warmup_epochs: epochs over which aux weight ramps from 4x to 1x base.

    Returns the trained SAE and a dict of training metrics.
    """
    model = TopKSAE(d, c, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    metrics = {"loss": [], "recon_loss": [], "dead_neurons": [], "variance_explained": []}
    step = 0

    # EMA tracker for firing rates — more stable dead neuron detection
    ema_firing = torch.zeros(c, device=device)
    ema_decay = 0.999  # ~1000 batch half-life

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} (k={k})")
        for (batch,) in pbar:
            optimizer.zero_grad()

            z, pre_acts = model.encode_with_pre_acts(batch)
            x_hat = model.decode(z)

            if loss_fn == "l1":
                recon_loss = (batch - x_hat).abs().sum(dim=-1).mean()
            else:
                recon_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # Update EMA firing rates
            with torch.no_grad():
                batch_firing = (z.abs() > 0).float().mean(dim=0)
                ema_firing = ema_decay * ema_firing + (1 - ema_decay) * batch_firing

            # Dead = EMA firing rate below threshold (more stable than per-batch)
            dead_mask = (ema_firing < 1e-4) if step > 100 else (z.abs().sum(dim=0) == 0)
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and aux_weight > 0:
                # For dead atoms, maximize their pre-activations relative to the
                # TopK threshold (the k-th largest pre-activation per sample)
                threshold = pre_acts.topk(k, dim=-1).values[:, -1:]  # (batch, 1)
                dead_pre_acts = pre_acts[:, dead_mask]  # (batch, n_dead)
                # Loss: how far below threshold are the dead atoms?
                # Encourage them to approach the threshold from below.
                aux_loss = (threshold - dead_pre_acts).clamp(min=0).mean()
                # Warmup: higher aux weight early to wake up dead neurons
                warmup_steps = len(loader) * aux_warmup_epochs
                if step < warmup_steps:
                    progress = step / warmup_steps
                    aw = aux_weight * (4.0 - 3.0 * progress)  # 4x -> 1x
                else:
                    aw = aux_weight
                loss = recon_loss + aw * aux_loss
            else:
                loss = recon_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Normalize decoder columns
            model.normalize_decoder()

            # Periodic hard reset of persistently dead neurons (every 2000 steps)
            if step > 0 and step % 2000 == 0:
                with torch.no_grad():
                    persistent_dead = (ema_firing < 1e-5)
                    n_persistent_dead = persistent_dead.sum().item()
                    if n_persistent_dead > 0:
                        model.reanimate_dead_neurons(z, batch, x_hat)

            epoch_loss += recon_loss.item()
            n_batches += 1
            step += 1

            if step % log_every == 0:
                with torch.no_grad():
                    var_total = batch.var(dim=0).sum().item()
                    var_residual = (batch - x_hat).var(dim=0).sum().item()
                    ve = 1 - var_residual / max(var_total, 1e-10)

                metrics["loss"].append(recon_loss.item())
                metrics["recon_loss"].append(recon_loss.item())
                metrics["dead_neurons"].append(n_dead)
                metrics["variance_explained"].append(ve)

                pbar.set_postfix(
                    loss=f"{recon_loss.item():.4f}",
                    VE=f"{ve:.4f}",
                    dead=n_dead,
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.6f}")

    return model, metrics


def train_sae_sweep(
    data: torch.Tensor,
    k_values: list[int] = [3, 4, 6, 8, 10, 14, 16, 20, 25],
    d: int = 128,
    c: int = 512,
    lr: float = 3e-3,
    batch_size: int = 1024,
    epochs: int = 10,
    device: str = "cpu",
    save_dir: str | None = None,
) -> dict[int, tuple[TopKSAE, dict]]:
    """Train SAEs across a sweep of sparsity budgets k."""
    results = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Training SAE with k={k}")
        print(f"{'='*60}")

        model, metrics = train_sae(
            data, d=d, c=c, k=k, lr=lr,
            batch_size=batch_size, epochs=epochs, device=device,
        )

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/sae_k{k}.pt")

        results[k] = (model, metrics)

    return results
