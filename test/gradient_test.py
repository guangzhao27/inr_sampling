"""Utility: compute sampled gradients from an INR checkpoint.

Main entry point
----------------
    compute_sampled_gradients(checkpoint_path, sampler_overrides, ...)

Given a .pt checkpoint (produced by inr_sample/single_image_inr.py) and a dict
of sampler overrides, this function reconstructs the model and graph, samples
coordinates using the specified sampler, performs a forward+backward pass, and
returns one gradient vector per requested repeat.

Typical usage
-------------
    from utils.compute_sampled_gradients import compute_sampled_gradients

    grads = compute_sampled_gradients(
        checkpoint_path="checkpoints/my_run_best.pt",
        sampler_overrides={"type": "random", "rate": 0.01},
        n_repeats=4,
        device="cuda",
    )
    # grads: list of torch.Tensor, each shape [n_params]

    # Single repeat → single tensor
    grad = compute_sampled_gradients("checkpoints/my_run_best.pt", n_repeats=1)[0]
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_utility_sampling.SamplerWrapper import create_inr_sampler
from utils.load_inr import create_inr_instance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(path), weights_only=False)


def _build_graph(ckpt: Dict, device: torch.device) -> Data:
    """Reconstruct a Data object from the graph_data dict stored in the checkpoint."""
    graph_data = ckpt.get("graph_data")
    if graph_data is None:
        raise KeyError(
            "Checkpoint does not contain 'graph_data'. Only checkpoints saved by "
            "inr_sample/single_image_inr.py (eval_*.pt, *_best.pt, *_final.pt) "
            "include this field."
        )
    required = ["cor", "space_emb", "feat", "time"]
    missing = [k for k in required if k not in graph_data]
    if missing:
        raise KeyError(f"graph_data is missing required keys: {missing}")

    kwargs: Dict = {k: graph_data[k] for k in required}
    if "T" in graph_data:
        kwargs["T"] = graph_data["T"]
    return Data(**kwargs).to(device)


def _apply_sampler_overrides(cfg, overrides: Optional[Dict]) -> object:
    """Return a deep-copied cfg with sampling fields replaced by *overrides*."""
    cfg_eval = copy.deepcopy(cfg)
    if overrides:
        for key, value in overrides.items():
            setattr(cfg_eval.sampling, key, value)
    return cfg_eval


def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    err_sq = (pred - target).pow(2).reshape(pred.shape[0], -1).mean(dim=1)
    if weight is None:
        return err_sq.mean()
    w = weight.to(err_sq.device, dtype=err_sq.dtype).reshape(-1)
    if w.numel() != err_sq.numel():
        return err_sq.mean()
    w_sum = w.sum()
    if not torch.isfinite(w_sum) or w_sum.item() <= 0.0:
        return err_sq.mean()
    return (w * err_sq).sum() / w_sum


def _call_sampler(sampler, graph: Data, step: int) -> Data:
    if sampler is None:
        return graph
    if sampler.__class__.__name__ == "EVOSSampler":
        return sampler.sample(graph=graph, epoch=step, inner_step=step, save_image=False)
    return sampler.sample(inner_step=step, graph=graph, save_image=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sampled_gradients(
    checkpoint_path: Union[str, Path],
    sampler_overrides: Optional[Dict] = None,
    n_repeats: int = 1,
    seed: int = 0,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """Compute parameter gradients of the sampled loss from an INR checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a .pt checkpoint file produced by inr_sample/single_image_inr.py.
        Must contain ``graph_data``, ``cfg``, ``inr``, ``input_dim``,
        ``output_dim``.
    sampler_overrides:
        Dict of sampling config overrides applied on top of the checkpoint's
        ``cfg.sampling``.  Keys match OmegaConf field names, e.g.::

            {"type": "random", "rate": 0.01}
            {"type": "2d_grid_adaptive", "adaptive_mode": "loss"}
            {"type": "NMT", "rate": 0.005}

        Pass ``None`` or an empty dict to keep the checkpoint's original sampler.
    n_repeats:
        Number of independent sampling+gradient evaluations to perform.  Each
        repeat uses a different random seed (``seed + repeat_index``), so the
        returned list contains ``n_repeats`` gradient vectors.
    seed:
        Base random seed.  Repeat *i* uses ``seed + i``.
    device:
        Torch device string (``"cuda"``, ``"cpu"``, …).  Defaults to
        ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.

    Returns
    -------
    List[torch.Tensor]
        A list of length ``n_repeats``.  Each element is a 1-D CPU tensor of
        shape ``[n_params]`` — the flattened gradient of the sampled MSE loss
        w.r.t. all model parameters.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    KeyError
        If the checkpoint is missing required fields (``graph_data``,
        ``cfg``, ``inr``, ``input_dim``, ``output_dim``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(checkpoint_path)

    cfg = ckpt["cfg"]
    input_dim: int = int(ckpt.get("input_dim", 2))
    output_dim: int = int(ckpt.get("output_dim", 1))

    # ------------------------------------------------------------------
    # 2. Reconstruct model
    # ------------------------------------------------------------------
    inr = create_inr_instance(cfg, input_dim, output_dim, dev)
    inr.load_state_dict(ckpt["inr"])
    inr.to(dev)

    # ------------------------------------------------------------------
    # 3. Reconstruct full graph
    # ------------------------------------------------------------------
    graph = _build_graph(ckpt, dev)

    # ------------------------------------------------------------------
    # 4. Build sampler (with overrides applied to cfg.sampling)
    # ------------------------------------------------------------------
    cfg_eval = _apply_sampler_overrides(cfg, sampler_overrides)
    image_width: int = int(graph.cor.max().item()) + 1
    current_date_str = ""
    run_name = "grad_eval"

    sampler = create_inr_sampler(
        cfg_eval, inr, graph, current_date_str, run_name, device=device
    )

    checkpoint_epoch: int = int(ckpt.get("epoch", 0))

    # ------------------------------------------------------------------
    # 5. Repeated sampling + gradient computation
    # ------------------------------------------------------------------
    gradients: List[torch.Tensor] = []

    for repeat_idx in range(n_repeats):
        _set_all_seeds(seed + repeat_idx)

        inr.train()
        inr.zero_grad(set_to_none=True)

        sampled_graph = _call_sampler(sampler, graph, step=checkpoint_epoch)

        pred = inr(sampled_graph.space_emb)
        weight = getattr(sampled_graph, "weight", None)
        loss = _weighted_mse(pred, sampled_graph.feat, weight)
        loss.backward()

        grad_vec = _flatten_gradients(inr).cpu()
        gradients.append(grad_vec)

    return gradients
