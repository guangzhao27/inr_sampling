from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import copy

import torch


def _is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def _clone_value(v: Any) -> Any:
    # Deep clone tensors; deep-copy python containers
    if torch.is_tensor(v):
        return v.clone()
    if isinstance(v, (dict, list, tuple, set)):
        return copy.deepcopy(v)
    return copy.copy(v)


class Data:
    """
    Minimal PyG-like Data container.
    Stores arbitrary fields as attributes.
    """
    def __init__(self, **kwargs: Any):
        object.__setattr__(self, "_store", {})  # avoid recursion in __setattr__
        for k, v in kwargs.items():
            self._store[k] = v

    def __getattr__(self, name: str) -> Any:
        store = object.__getattribute__(self, "_store")
        if name in store:
            return store[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_store":
            object.__setattr__(self, name, value)
        else:
            self._store[name] = value

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def clone(self) -> "Data":
        return Data(**{k: _clone_value(v) for k, v in self._store.items()})

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "Data":
        for k, v in self._store.items():
            if torch.is_tensor(v):
                self._store[k] = v.to(device, non_blocking=non_blocking)
        return self

    def cpu(self) -> "Data":
        """Move all tensor attributes to CPU."""
        return self.to('cpu')

    def cuda(self, device: Optional[int] = None, non_blocking: bool = False) -> "Data":
        """Move all tensor attributes to CUDA."""
        if device is None:
            return self.to('cuda', non_blocking=non_blocking)
        return self.to(f'cuda:{device}', non_blocking=non_blocking)

    @property
    def num_nodes(self) -> int:
        # Heuristic: prefer feat, then cor, then time, then space_emb
        if "feat" in self._store and torch.is_tensor(self._store["feat"]):
            return int(self._store["feat"].size(0))
        if "cor" in self._store and torch.is_tensor(self._store["cor"]):
            return int(self._store["cor"].size(0))
        if "time" in self._store and torch.is_tensor(self._store["time"]):
            return int(self._store["time"].size(0))
        if "space_emb" in self._store and torch.is_tensor(self._store["space_emb"]):
            return int(self._store["space_emb"].size(0))
        raise ValueError("Cannot infer num_nodes: no node-level tensor field found.")


class Batch(Data):
    """
    Minimal PyG-like Batch container.
    Provides `from_data_list` to concatenate fields across graphs.
    Creates:
      - batch: (total_nodes,) graph id for each node
      - ptr: (num_graphs+1,) node prefix sums
      - num_graphs: int
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_graphs: int = int(self._store.get("num_graphs", 0))

    @staticmethod
    def from_data_list(data_list: List[Data]) -> "Batch":
        if len(data_list) == 0:
            raise ValueError("from_data_list got an empty list.")

        # Gather all keys present
        all_keys = set()
        for d in data_list:
            all_keys.update(d.keys())

        # Node prefix sums / batch vector
        num_nodes_list = [d.num_nodes for d in data_list]
        ptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes_list), dim=0).tolist()),
                           dtype=torch.long, device=_infer_device(data_list))
        batch_vec = torch.repeat_interleave(
            torch.arange(len(data_list), device=ptr.device, dtype=torch.long),
            torch.tensor(num_nodes_list, device=ptr.device, dtype=torch.long),
        )

        out: Dict[str, Any] = {
            "batch": batch_vec,
            "ptr": ptr,
            "num_graphs": len(data_list),
        }

        # Helper: decide concat vs stack vs list
        def _cat_if_possible(vals: List[Any], key: str) -> Any:
            # If any value is not a tensor: keep as python list
            if not all(_is_tensor(v) for v in vals):
                return vals

            # Special-case latent_vector: concat along time dimension
            if key == "latent_vector":
                return torch.cat(vals, dim=0)

            # Special-case node-level fields: concat along dim 0 if they match num_nodes
            # (Your cor/time/feat/space_emb match this.)
            node_like = True
            for d, v in zip(data_list, vals):
                if v.dim() == 0:
                    node_like = False
                    break
                if v.size(0) != d.num_nodes:
                    node_like = False
                    break
            if node_like:
                return torch.cat(vals, dim=0)

            # Scalar tensors (e.g., T if stored as scalar tensor)
            if all(v.dim() == 0 or (v.dim() == 1 and v.numel() == 1) for v in vals):
                return torch.stack([v.reshape(()) for v in vals], dim=0)

            # If shapes match exactly, stack along new batch dimension
            shapes = [tuple(v.shape) for v in vals]
            if all(s == shapes[0] for s in shapes):
                return torch.stack(vals, dim=0)

            # Otherwise keep list (ragged tensors)
            return vals

        for key in sorted(all_keys):
            if key in ("batch", "ptr", "num_graphs"):
                continue
            vals = [d.get(key) for d in data_list]
            out[key] = _cat_if_possible(vals, key)

        return Batch(**out)


def _infer_device(data_list: List[Data]) -> torch.device:
    # Find first tensor and use its device; otherwise CPU
    for d in data_list:
        for k in d.keys():
            v = d.get(k)
            if torch.is_tensor(v):
                return v.device
    return torch.device("cpu")
