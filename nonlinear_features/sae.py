"""TopK Sparse Autoencoder."""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    x_hat = W_dec @ TopK(W_enc @ x)

    W_dec columns are constrained to unit norm.
    """

    def __init__(self, d: int, c: int, k: int):
        super().__init__()
        self.d = d  # input dimension
        self.c = c  # dictionary size
        self.k = k  # sparsity (number of active atoms)

        self.W_enc = nn.Linear(d, c, bias=True)
        self.W_dec = nn.Linear(c, d, bias=False)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.weight.data = nn.functional.normalize(
                self.W_dec.weight.data, dim=0
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and apply TopK sparsity."""
        pre_acts = self.W_enc(x)  # (batch, c)
        # TopK: keep only top-k activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        sparse_code = torch.zeros_like(pre_acts)
        sparse_code.scatter_(1, topk_idx, topk_vals)
        return sparse_code

    def encode_with_pre_acts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode returning both sparse code and raw pre-activations."""
        pre_acts = self.W_enc(x)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        sparse_code = torch.zeros_like(pre_acts)
        sparse_code.scatter_(1, topk_idx, topk_vals)
        return sparse_code, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.W_dec(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, sparse_code)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder columns back to unit norm."""
        self.W_dec.weight.data = nn.functional.normalize(
            self.W_dec.weight.data, dim=0
        )

    @torch.no_grad()
    def reanimate_dead_neurons(self, z: torch.Tensor, x: torch.Tensor, x_hat: torch.Tensor):
        """Reanimate dead neurons by pointing them toward high-error inputs.

        Dead neurons are those with zero activation in the current batch.
        """
        # Find dead neurons (no activation in batch)
        dead_mask = (z.abs().sum(dim=0) == 0)  # (c,)
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Find inputs with highest reconstruction error
        errors = (x - x_hat).pow(2).sum(dim=-1)  # (batch,)
        _, high_error_idx = errors.topk(min(n_dead, len(errors)))
        high_error_inputs = x[high_error_idx]  # (n_dead, d)

        # Point dead decoder columns toward high-error inputs
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_reassign = min(n_dead, len(high_error_inputs))
        directions = high_error_inputs[:n_reassign] - x_hat[high_error_idx[:n_reassign]]
        directions = nn.functional.normalize(directions, dim=-1)

        self.W_dec.weight.data[:, dead_indices[:n_reassign]] = directions.T

        # Update encoder to match
        self.W_enc.weight.data[dead_indices[:n_reassign]] = directions
        self.W_enc.bias.data[dead_indices[:n_reassign]] = 0.0

        return n_dead
