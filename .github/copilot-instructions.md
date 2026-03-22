# Copilot Instructions for `inr_sampling`

## Project Overview

- **Implicit Neural Representations (INRs):** Neural networks (primarily SIREN) that map spatial/temporal coordinates → signal values (e.g., image pixels, PDE fields).
- **Training-time sampling strategies:** Instead of training on every coordinate at every step, a sampler selects a subset of coordinates each epoch to reduce cost and improve convergence.
- **Adaptive space partitioning:** The domain is recursively divided into cells using a hierarchical quad-tree (`utils/quadtree.py`). Cells are split when their weighted variance exceeds a threshold.
- **Gradient variance-driven sampling:** Per-cell Jacobian/gradient variance is estimated via Taylor expansion (`train_utility_sampling/taylor_estimation.py`) and used to allocate more samples to high-complexity regions (Neyman-style allocation).
- **Sampling strategies available:** `random`, `null` (full domain), `NMT` (Non-parametric Machine Teaching), `2d_grid_linear`, `2d_grid_adaptive`, `2d_cluster_slic`, `EVOS` — configured via `config/inr_sample.yaml` (`sampling.type`).
- **Baseline samplers for comparison:** Uniform random, grid-based, and EVOS samplers serve as benchmarks against the adaptive strategy.
- **Evaluation metrics:** MSE, relative RMSE, PSNR, LPIPS, SSIM — computed during validation in `train_utility_sampling/train_utility.py` and `train_utility_sampling/losses.py`.

---

## Where Things Live

| Concern | Path |
|---|---|
| **INR model / SIREN architecture** | `utils/siren.py` |
| **INR model loading / instantiation** | `utils/load_inr.py` |
| **Modulation utilities** | `utils/modulation_utility.py` |
| **All sampler implementations** | `train_utility_sampling/SamplerWrapper.py` — main class: `INRSingle2dSamplerWrapper.sample()` |
| **Additional sampling helpers** | `train_utility_sampling/sampler.py` |
| **NMT sampling algorithm** | `components/nmt/nmt.py`, `components/nmt/sampler.py`, `components/nmt/scheduler.py` |
| **EVOS sampler** | `inr_sample/evos_sampler.py` |
| **Gradient / Jacobian estimation** | `train_utility_sampling/taylor_estimation.py` — key functions: `frobenius_norm_via_jacrev`, `cell_grad_variance_estimate_with_jacrev` |
| **Adaptive spatial partitioning (quad-tree)** | `utils/quadtree.py` — classes: `ImageCell`, `HierarchicalImageGrid` |
| **Training loop (single image)** | `train_utility_sampling/train_utility.py` — function: `train_step_single_image` |
| **Meta-learning training loop** | `train_utility_sampling/metalearning_sampling.py` |
| **Main entry point** | `inr_sample/single_image_inr.py` (Hydra-configured) |
| **Meta-learning entry point** | `inr_sample/meta_learning_inr.py` |
| **Experiment configs** | `config/inr_sample.yaml`, `config/siren.yaml`, `config/ode.yaml` |
| **Loss functions** | `train_utility_sampling/losses.py` |
| **Metrics / perceptual losses** | `components/lpips.py`, `components/ssim.py`, `components/laplacian.py` |
| **Visualization / result display** | `utils/result_show.py`, `util/plotter.py`, `inr_sample/inr_sampling_show.ipynb` |
| **Data loading (structured)** | `utils/data/structure_dataset.py`, `utils/data/load_data.py` |
| **Data loading (graph/unstructured)** | `utils/data/unstructure_dataset.py` — classes: `GraphNavierStokes`, `GraphBurgers`, `GraphSomaDataset` |
| **Experiment tracking / logging** | `util/logger.py`, `util/tensorboard.py`, `util/recorder.py` (WandB via `wandb.*` config keys) |
| **Bash execution scripts** | `script/inr_sample/` |
| **Test files** | `test/test.py`, `test/grad_GT_calculation.py`, `test/grad_estimation.py`, `test/grad_generation.py` |
| **Interactive test notebooks** | `inr_sample/Run_Test.ipynb`, `inr_sample/inr_sampling_show.ipynb` |
| **Data directories** | Repo-specific TODO — set `data.data_path` in the config or CLI override; supported formats: `mmap`, `hdf5`, `other` (numpy `.npy`) |

---

## Standard Workflows

### Environment Setup

```bash
# Python 3.11 required — higher versions break Hydra + torch-geometric
conda env create -f environment.yml
conda activate torchgeo
```

### Train an INR with Baseline Random Sampling

```bash
python inr_sample/single_image_inr.py \
    data.dataset_name=NS \
    data.data_path=/path/to/data.npy \
    data.data_type=other \
    data.single_time_frame=100 \
    inr.model_type=siren \
    inr.depth=6 inr.hidden_dim=155 inr.w0=30 \
    optim.lr_inr=1e-4 optim.epochs=9000 optim.inner_steps=6 \
    sampling.type=random \
    sampling.rate=0.2
```

### Train with Adaptive 2D Grid Sampling

```bash
python inr_sample/single_image_inr.py \
    data.dataset_name=NS \
    data.data_path=/path/to/data.npy \
    data.data_type=other \
    data.single_time_frame=100 \
    inr.model_type=siren \
    inr.depth=6 inr.hidden_dim=155 inr.w0=30 \
    optim.lr_inr=1e-4 optim.epochs=9000 optim.inner_steps=6 \
    sampling.type=2d_grid_adaptive \
    sampling.rate=0.001 \
    sampling.n_clusters_2d_start=11 \
    sampling.n_clusters_2d_end=128
```

### Run Comparisons Across Samplers (via Bash Script)

```bash
# Edit script parameters (sample_type, sampling_rate, etc.) then submit:
bash script/inr_sample/nersc-scent-single-inr-sample.sh

# Or loop manually:
for sample_type in random NMT 2d_grid_linear 2d_grid_adaptive EVOS; do
  python inr_sample/single_image_inr.py \
      sampling.type=$sample_type \
      wandb.use_wandb=True wandb.name="compare_${sample_type}"
done
```

### Evaluate Reconstruction Quality

Validation metrics (MSE, RMSE, PSNR, LPIPS, SSIM) are logged automatically during training. To evaluate a saved checkpoint:

```bash
python inr_sample/single_image_inr.py \
    saved_checkpoint=True \
    checkpoint_path=/path/to/checkpoint.pt \
    sampling.type=null   # full-domain evaluation
```

### Visualize Sampled Points or Learned Fields

```bash
# Interactive notebooks:
jupyter notebook inr_sample/inr_sampling_show.ipynb
jupyter notebook inr_sample/Run_Test.ipynb
```

### Run Tests

```bash
cd test && bash run_test.sh
# or
python test/test.py
```

---

## Coding Conventions

- **Style:** Follow PEP 8. No enforced formatter detected — match surrounding code style.
- **Type hints:** Use where present in the codebase; avoid adding unannotated arguments to already-typed functions.
- **Separation of concerns:**
  - `utils/siren.py` — model definition only (no training logic).
  - `train_utility_sampling/SamplerWrapper.py` — sampler logic only (no model-specific code).
  - `train_utility_sampling/train_utility.py` — training step only (orchestrates model + sampler).
  - `train_utility_sampling/taylor_estimation.py` — pure gradient math, no side effects.
- **Configuration:** All hyperparameters flow through Hydra (`config/inr_sample.yaml`). Add new parameters there; do not hardcode values in training scripts.
- **Experiment tracking:** WandB is the primary tracker. Toggle with `wandb.use_wandb=True/False`. Run names go in `wandb.name`.

### Adding a New Sampling Strategy

1. Implement the sampler as a method `_sample_<name>(self, graph, ...)` inside `train_utility_sampling/SamplerWrapper.py` following the pattern of `_sample_random` or `_sample_nmt`.
2. Register the new type in `INRSingle2dSamplerWrapper.sample()` dispatch block.
3. Add the new `sampling.type` string to `config/inr_sample.yaml` as a comment under `# sample type:`.
4. If the sampler requires new hyperparameters, add them to the `sampling:` section of `config/inr_sample.yaml`.
5. Test with `inr_sample/Run_Test.ipynb`.

### Plugging a Sampler into the Training Loop

`train_step_single_image` in `train_utility_sampling/train_utility.py` calls `sampler.sample()` to obtain coordinate indices before the forward pass. The sampler wrapper is created via `create_inr_sampler()` in `SamplerWrapper.py`. Pass the constructed sampler object to `train_step_single_image`.

---

## Data Contract / Interface Notes

### Sampler API

```python
# Inputs
sampler.sample(
    graph,          # torch_geometric.data.Data — full coordinate graph
    model,          # nn.Module — current INR
    modulations,    # Tensor [N_lat, latent_dim] — current latent codes
)
# Output
indices           # LongTensor [K] — indices into graph.pos for selected coordinates
```

### Tensor Shapes

| Variable | Shape | Description |
|---|---|---|
| `graph.pos` | `[N, d]` | Spatial coordinates (`d=2` for images, `d=3` for 3D fields) |
| `graph.x` | `[N, C]` | Target signal values (`C=1` grayscale, `C=3` RGB, etc.) |
| `modulations` | `[B, latent_dim]` | Per-sample latent codes |
| Gradient Frobenius norm | `[M]` | One scalar per cell (`M` = number of cells) |
| Cell variance | `[M]` | Weighted variance estimate per cell |

### Gradient Statistics Per Cell

- Computed by `cell_grad_variance_estimate_with_jacrev` in `train_utility_sampling/taylor_estimation.py`.
- Returns per-cell gradient variance as a 1-D tensor aligned with the cell list in `HierarchicalImageGrid`.
- Do **not** average across cells before using for allocation — allocation decisions require per-cell values.

### Partition Structure

- `HierarchicalImageGrid` (in `utils/quadtree.py`) stores a list of `ImageCell` objects.
- Each `ImageCell` holds `(x_start, y_start, x_end, y_end)` with **inclusive** end indices.
- Cells are split by calling `cell.subdivide()`, which returns four child `ImageCell` objects.
- Minimum cell size is set at `utils/quadtree.py:37` — do not subdivide below this threshold.

---

## Safety / Constraints

- **Do not change sampling semantics** (e.g., sampling distribution, cell allocation formula) without updating `SamplerWrapper.py`, `taylor_estimation.py`, and `quadtree.py` together.
- **Gradient estimates must be consistent:** `frobenius_norm_via_jacrev` and `cell_grad_variance_estimate_with_jacrev` are the canonical reference implementations. Do not swap estimation methods silently.
- **Fair baseline comparison:** Random and NMT samplers must sample the same total number of coordinates (`sampling.rate * N`) as the adaptive method under equal conditions.
- **Do not silently change default `sampling.type`** in `config/inr_sample.yaml` — the default `"random"` is the baseline.
- **Reproducibility:** Set `data.seed` in the config (default `15`). Any new stochastic component must accept and use a seed parameter.
- **No large refactors without explicit request.** Prefer additive changes (new methods/classes) over modifying existing algorithmic logic.

---

## Validation Checklist

Before merging any change:

- [ ] Run a small training job and confirm loss decreases:
  ```bash
  python inr_sample/single_image_inr.py optim.epochs=100 sampling.type=random
  ```
- [ ] Verify per-cell sample counts match `sampling.rate * N` (total) within floating-point tolerance.
- [ ] Compare loss curves against random baseline for at least one dataset to confirm no regression.
- [ ] Re-run with `data.seed=15` twice and confirm identical results (reproducibility check).
- [ ] Ensure no tensor shape or device mismatches (watch for `cpu`/`cuda` errors after device-dependent changes).
- [ ] Run `python test/test.py` and `bash test/run_test.sh` — both should complete without errors.
- [ ] If gradient estimation was changed, validate against `test/grad_GT_calculation.py` ground-truth output.
