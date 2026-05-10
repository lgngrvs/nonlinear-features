"""JumpReLU Sparse Autoencoder — compatible with GemmaScope 2 checkpoints.

GemmaScope uses JumpReLU activation:
    f(x) = ReLU(z) * (z > threshold)   where z = x @ W_enc + b_enc
    x_hat = f @ W_dec + b_dec

Checkpoint format (safetensors):
    W_enc: (d_in, d_sae)
    W_dec: (d_sae, d_in)
    b_enc: (d_sae,)
    b_dec: (d_in,)
    threshold: (d_sae,)
"""

import torch
import torch.nn as nn
from pathlib import Path


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE matching GemmaScope 2 architecture."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with JumpReLU activation: ReLU(z) * (z > threshold)."""
        z = x @ self.W_enc + self.b_enc  # (batch, d_sae)
        return torch.relu(z) * (z > self.threshold).float()

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, sparse_code)."""
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "JumpReLUSAE":
        """Load from a GemmaScope safetensors checkpoint.

        Args:
            path: Path to directory containing params.safetensors (and config.json),
                  or direct path to a .safetensors file.
        """
        from safetensors.torch import load_file

        path = Path(path)
        if path.is_dir():
            weights_path = path / "params.safetensors"
            if not weights_path.exists():
                weights_path = path / "model.safetensors"
        else:
            weights_path = path

        state_dict = load_file(str(weights_path))

        d_in = state_dict["W_enc"].shape[0]
        d_sae = state_dict["W_enc"].shape[1]

        model = cls(d_in, d_sae)
        model.load_state_dict(state_dict)
        model = model.to(device=device, dtype=dtype)
        model.eval()
        return model

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = "google/gemma-scope-2-12b-it",
        layer: int = 24,
        site: str = "resid_post",
        width: str = "16k",
        l0: str = "medium",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "JumpReLUSAE":
        """Download and load a GemmaScope SAE from HuggingFace.

        Args:
            repo_id: HuggingFace repo (e.g. "google/gemma-scope-2-12b-it")
            layer: Transformer layer number
            site: Activation site ("resid_post", "attn_out", "mlp_out")
            width: SAE width ("16k", "65k", "262k", "1m")
            l0: Sparsity level ("small", "medium", "big")
            device: Target device
            dtype: Target dtype
        """
        from huggingface_hub import hf_hub_download

        subfolder = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="params.safetensors",
            subfolder=subfolder,
        )

        return cls.from_pretrained(weights_path, device=device, dtype=dtype)

    def sparsity_stats(self, x: torch.Tensor) -> dict:
        """Compute sparsity statistics for a batch."""
        with torch.no_grad():
            f = self.encode(x)
            l0 = (f > 0).float().sum(dim=-1)
            frac_alive = (f.sum(dim=0) > 0).float().mean()
        return {
            "l0_mean": l0.mean().item(),
            "l0_std": l0.std().item(),
            "frac_alive_latents": frac_alive.item(),
        }
