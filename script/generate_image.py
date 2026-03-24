"""
generate_image.py  --  Load a saved INR checkpoint and visualize the results.

Usage:
    python generate_image.py <checkpoint_path> [output_path]

Arguments:
    checkpoint_path : path to a .pt checkpoint saved by single_image_inr.py
                      (must contain keys: inr, cfg, input_dim, output_dim, graph_data)
    output_path     : (optional) where to save the figure; defaults to
                      ./results/visualization/<checkpoint_stem>.png

The script produces a 3-panel figure:
    [Ground Truth] | [Prediction] | [Absolute Difference]
and saves it to disk.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on the path so local modules are importable
# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parents[1]))
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.load_inr import create_inr_instance


# ---------------------------------------------------------------------------
# Helper: scatter flat node values onto a 2-D image grid
# ---------------------------------------------------------------------------

def nodes_to_image(cor: torch.Tensor, values: np.ndarray) -> np.ndarray:
    """
    Scatter per-node scalar values onto a 2-D pixel grid.

    Args:
        cor    : integer pixel coordinates, shape (N, 2), columns are (row, col)
        values : flat array of N scalar values

    Returns:
        img : numpy array of shape (H, W)
    """
    H = int(cor[:, 0].max().item()) + 1
    W = int(cor[:, 1].max().item()) + 1
    img = np.zeros((H, W), dtype=np.float32)
    cor_np = cor.numpy().astype(int)
    for i, (r, c) in enumerate(cor_np):
        img[r, c] = values[i]
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- parse arguments --------------------------------------------------
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    # ---- load checkpoint --------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)

    cfg        = ckpt["cfg"]
    input_dim  = ckpt.get("input_dim", 2)
    output_dim = ckpt.get("output_dim", 1)
    graph_data = ckpt["graph_data"]   # dict with cor, space_emb, feat, time

    # ---- reconstruct model ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.set_default_dtype(torch.float32)
    inr = create_inr_instance(cfg, input_dim=input_dim, output_dim=output_dim, device=device)
    inr.load_state_dict(ckpt["inr"])
    inr.to(device).eval()

    # ---- run inference on full image --------------------------------------
    space_emb = graph_data["space_emb"].to(device)   # (N, input_dim)
    cor       = graph_data["cor"].cpu()               # (N, 2)  integer coords
    feat      = graph_data["feat"].cpu()              # (N, 1) or (N,)

    with torch.no_grad():
        pred = inr(space_emb)                         # (N, output_dim)

    pred_np = pred.cpu().numpy().reshape(-1)
    feat_np = feat.numpy().reshape(-1)

    # ---- build 2-D images -------------------------------------------------
    gt_img   = nodes_to_image(cor, feat_np)
    pred_img = nodes_to_image(cor, pred_np)
    diff_img = np.abs(gt_img - pred_img)

    # ---- compute metrics --------------------------------------------------
    mse  = float(np.mean((gt_img - pred_img) ** 2))
    drange = float(gt_img.max() - gt_img.min())
    psnr = float(20 * np.log10(drange / np.sqrt(mse))) if mse > 0 else float("inf")
    rel_rmse = float(np.sqrt(mse) / (np.sqrt(np.mean(gt_img ** 2)) + 1e-8))

    epoch     = ckpt.get("epoch", "?")
    best_loss = ckpt.get("loss", float("nan"))
    print(f"Epoch: {epoch}  |  best rel-RMSE: {best_loss:.6f}")
    print(f"Full-image MSE: {mse:.6e}  |  PSNR: {psnr:.2f} dB  |  Rel-RMSE: {rel_rmse:.6f}")

    # ---- plot -------------------------------------------------------------
    vmin = gt_img.min()
    vmax = gt_img.max()

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.25)

    ax_gt   = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])
    ax_cbar = fig.add_subplot(gs[3])

    im_gt   = ax_gt.imshow(gt_img,   cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    im_pred = ax_pred.imshow(pred_img, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    im_diff = ax_diff.imshow(diff_img, cmap="hot",    vmin=0,    origin="upper")

    ax_gt.set_title("Ground Truth",      fontsize=13)
    ax_pred.set_title("Prediction",       fontsize=13)
    ax_diff.set_title("|Difference|",     fontsize=13)

    for ax in (ax_gt, ax_pred, ax_diff):
        ax.axis("off")

    # shared colorbar for GT and Prediction
    plt.colorbar(im_gt, cax=ax_cbar)

    # difference colorbar as inset
    cbar_diff = fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
    cbar_diff.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"INR reconstruction  |  epoch {epoch}  |  "
        f"PSNR {psnr:.2f} dB  |  Rel-RMSE {rel_rmse:.4f}",
        fontsize=12,
        y=1.01,
    )

    # ---- save figure ------------------------------------------------------
    if output_path is None:
        stem = Path(checkpoint_path).stem
        # the second of last folder name of checkpoint path
        parent_folder = Path(checkpoint_path).parent.name
        out_dir = Path(f"./Results/checkpoints/{parent_folder}")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{stem}.png")

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
