# Copilot Instructions for INR_SAMPLE

Whenever you update the code, please also update this instructions file to reflect the current state of the codebase and provide clear guidance for future contributors, if the change is minor, no need to update this file, while if the change is major (e.g., new sampling method, new dataset, significant refactor), please update this file with relevant sections such as "Where Things Live", "Standard Workflows", "Coding Conventions", etc. to ensure that the instructions remain accurate and helpful for future contributors.

## Project Overview
- This repo trains Implicit Neural Representations (INRs), mainly SIREN-family models (`siren`, `ssn`) for spatiotemporal fields.
- Single-image training also supports a non-sinusoidal baseline: Fourier-feature MLP (`single_image_fourier_mlp`) using fixed Gaussian Fourier encoding plus an MLP head.
- Sampling is part of training, not just preprocessing: the model is fit on sampled coordinates each step.
- Adaptive sampling is implemented with cell/cluster decomposition (`grid` or `slic`) and score-based point selection.
- Gradient/error statistics drive selection in adaptive modes such as `NMT`, `2d_cluster`, `3d_cluster`, and `EVOS`.
- Baseline samplers include `random` and cluster-based baselines for fair comparisons.
- The training loop measures reconstruction quality with MSE and relative RMSE; single-image flow also computes PSNR/SSIM.
- Comparisons should keep architecture, optimizer, and data split fixed while only changing sampler config.

## Where Things Live
- Core INR models and builders:
  - `utils/siren.py` (SIREN, ModulatedSiren, sampling helpers)
  - `utils/fourier_mlp.py` (FourierFeatureMLP for coordinate regression)
  - `utils/load_inr.py` (`create_inr_instance`, `load_inr_model`)
- Sampling algorithms (baseline + adaptive):
  - `train_utility_sampling/SamplerWrapper.py` (`InrSamplerWrapper`, `INRSingle2dSamplerWrapper`, `INRSingle2dAdaptiveSamplerWrapper`, `EVOSSampler`)
  - `train_utility_sampling/sampler.py` (NMT-style top-k/random selection utility)
  - `components/nmt/` (scheduler/strategy/sampler helpers for NMT)
  - `train_utility_sampling/taylor_estimation.py` (cell score functions used by adaptive wrapper)
  - `utils/quadtree.py` (`ImageCell`, `HierarchicalImageGrid` used for iterative adaptive subdivision)
- Partitioning / spatial decomposition:
  - `train_utility_sampling/SamplerWrapper.py` (`graph_2d_cluster`, `graph_3d_cluster`, `graph_2d_cluster_single_image`)
  - Partitions are stored as `graph.cluster_set` (list/dict of cluster_id -> node index tensor).
- Training loops / trainers:
  - `inr_sample/single_image_inr.py` (Hydra entrypoint for single-image INR training + sampling)
  - `inr_sample/meta_learning_inr.py` (Hydra entrypoint for multi-trajectory/meta-style training)
  - `train_utility_sampling/train_utility.py` and `train_utility_sampling/metalearning_sampling.py`
- Configs (datasets, model, optimizer, sampling):
  - `config/inr_sample.yaml` (primary for INR sampling experiments)
  - `config/siren.yaml`, `config/ode.yaml`
- Evaluation / metrics / visualization:
  - `train_utility_sampling/losses.py` (`relative_rmse`, MSE-based losses)
  - `train_utility_sampling/metalearning_sampling.py` (PSNR/SSIM in single-image path)
  - `utils/result_show.py` and notebooks in `inr_sample/*.ipynb`, `test/*.ipynb`
- Data-related modules and dirs:
  - `utils/data/unstructure_dataset.py`, `utils/data/load_data.py`
  - `utils/data/structure_dataset.py`
  - `utils/data/generate_burgers2d.py` (standalone script; generates 2D Burgers HDF5 files)
  - `utils/data/`
  - Dataset root used by current experiments: `./data/` (equivalent to `../data/` when launched from `inr_sample/`)
  - Current `data/` snapshot (workspace, April 2026):
    - Pre-generated 1D Burgers HDF5 files: `1D_Burgers_Sols_Nu{0.001,0.002,0.004,0.01,0.02,0.04,0.1}.hdf5`.
    - Pre-generated 2D Burgers HDF5 files: `2D_Burgers_Sols_Nu{0.001,0.002,0.005}.hdf5`.
    - Additional dataset directories: `NS2d/`, `2D_Burgers_1024/`, `2D_Burgers_Sols/`, `KuramotoSivashinsky/`, `cylinder_flow/`, `airfoil/`, `PoolBoiling-SubCooled-FC72-2D/`, `scent_data/`, `xarray_tutorial_data/`.
    - Standalone large NS files also exist: `ns_incom_inhom_2d_512-0.h5`, `ns_V1e-3_N5000_T50.mat`.
    - Utility/data access artifacts: `download.sh`, `wget-log`, `dataload.ipynb`.
    - Approximate footprint highlights: `1D_Burgers_Sols_Nu*.hdf5` ~7.7G each, `2D_Burgers_Sols_Nu*.hdf5` ~3.5G each, `NS2d/` ~12G, `cylinder_flow/` ~16G, `PoolBoiling-SubCooled-FC72-2D/` ~16G.
  - Data generation code lives outside this repo root in `../Data_generation/`
  - 2D Burgers data files: `../data/2D_Burgers_Sols_Nu{nu}.hdf5` — pre-generated,
    NOT solved on the fly (on-the-fly 2D PDE solving is too slow for DataLoader workers)
  - `GraphPiecewise2D` / `create_piecewise_dataset`: fully in-memory synthetic dataset,
    generated on-the-fly in `__init__` (closed-form evaluation, no disk I/O needed)
- Scripts:
  - `script/inr_sample/*.sh` (cluster launch examples; verify paths before use)
  - `Results/generate_performance_figures.py` (builds `trainepoch.pdf`, `traintime.pdf`, `trainrel_loss.pdf` from W&B runs)
- Paper writing source:
  - `Results/latex/INR-Refined-Grid-Sampling-Neurips-Workshop/ICLR_INR_RGS.tex` (main paper LaTeX file)
  - `Results/latex/NIPS-2026-INR-Sampling/neurips_2026.tex` (NeurIPS 2026 main LaTeX source)
  - Third-experiment insertion point for gradient-alignment figure/caption/discussion:
    `\section{Experiments}` -> `\subsection{Ablation Study}` -> `\subsubsection{The weight function effect}`
  - Figure asset used by the third experiment:
    `Results/latex/NIPS-2026-INR-Sampling/figure/sampler_gradient_alignment_metric.pdf`

## W&B Offline Run Tracking (BNL cluster)
Use this skill when you work on sh file: `.github/skills/wandb-offline-sync/SKILL.md` for the full pattern (before/after snapshot, TSV log, sync-script generation) and reference implementations.

## Standard Workflows

### 1) Environment Setup
```bash
conda env create -f environment.yml
conda activate torchgeo
```

### 2) Train INR with Baseline Random Sampling (single-image flow)
```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=NS \
  inr.model_type=siren \
  sampling.type=random \
  sampling.rate=0.01 \
  optim.epochs=2000 \
  data.single_time_frame=100 \
  wandb.use_wandb=False
```

Fourier-feature MLP variant (same training loop/sampler API):
```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=NS \
  inr.model_type=single_image_fourier_mlp \
  inr.fourier_mapping_size=256 \
  inr.fourier_scale=10.0 \
  sampling.type=random \
  sampling.rate=0.01 \
  optim.epochs=2000 \
  data.single_time_frame=100 \
  wandb.use_wandb=False
```

### 3) Train INR with Adaptive Sampling
Example: `2d_cluster` (cell-based partition + score selection)
```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=NS \
  inr.model_type=siren \
  sampling.type=2d_cluster \
  sampling.rate=0.01 \
  sampling.n_clusters_2d_start=11 \
  sampling.n_clusters_2d_end=128 \
  optim.epochs=2000 \
  data.single_time_frame=100 \
  wandb.use_wandb=False
```

Example: `EVOS`
```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=NS \
  inr.model_type=siren \
  sampling.type=EVOS \
  sampling.rate=0.01 \
  sampling.sample_num_schedular=constant \
  sampling.mutation_method=constant \
  sampling.profile_interval_method=lin_dec \
  sampling.profile_guide=value \
  optim.epochs=2000 \
  wandb.use_wandb=False
```

### 4) Run Comparisons Across Samplers
Use identical settings except `sampling.type` and run one job per sampler.
```bash
for s in random NMT 2d_cluster EVOS; do
  python inr_sample/single_image_inr.py \
    data.dataset_name=NS \
    inr.model_type=siren \
    sampling.type=$s \
    sampling.rate=0.01 \
    optim.epochs=1000 \
    data.single_time_frame=100 \
    wandb.name=cmp_${s} \
    wandb.use_wandb=False
 done
```
Repo-specific TODO: add a dedicated benchmark driver that aggregates metrics into one table.

### 4.1) Adaptive Single-Image Wrapper (`2d_grid_adaptive`)

Current implementation path for adaptive sampling in single-image INR training:

1) Sampler construction / wiring
- `inr_sample/single_image_inr.py:create_inr_sampler(...)`
- When `sampling.type=2d_grid_adaptive`, it returns `INRSingle2dAdaptiveSamplerWrapper`.

2) Training-step call site
- `train_utility_sampling/metalearning_sampling.py:single_image_step(...)`
- For non-EVOS samplers, it calls `sampler.sample(inner_step=step, graph=graph_ori, save_image=False)`.

3) Adaptive wrapper core
- `train_utility_sampling/SamplerWrapper.py:INRSingle2dAdaptiveSamplerWrapper.sample(...)`
- Creates `HierarchicalImageGrid(image_width, image_width, initial_grid_size=16)`.
- Builds evaluation callback via `_create_evaluation_function(graph, mode)` where:
  - `mode='gradient'` uses `cell_grad_variance_estimate_with_jacrev(...)`.
  - `mode='loss'` uses `loss_variance_ground_truth(...)`.
- Runs `grid.iterative_subdivision(...)` with configurable iteration count via `sampling.adaptive_iterations` (default 8).
- Reads leaf-cell tensors via `grid.get_leaf_properties_tensor(evaluation_function=eval_fn, device=self.device)`.
- Allocates stochastic sample counts per cell with `sample_counts_poisson(values, expected_total=n_samples)`.
- Draws variable samples with `sample_variable_from_2d_intervals_vcounts(bounds, counts, device=...)`.
- Converts `(x, y)` to flat index: `sampled_idx = y * image_width + x`.
- Attaches per-sample weights in returned `Data(..., weight=sampled_weights)`.

Optional variant (new):
- Config flag: `sampling.adaptive_equal_cell_topk=True`.
- After adaptive partition is produced, candidate counts can be configured with `sampling.adaptive_equal_cell_topk_count_mode`:
  - `same`: equal-per-cell counts.
  - `poisson`: stochastic counts via `sample_counts_poisson(values, expected_total=n_samples)`.
- It computes per-sampled-point prediction difference `abs(feat - pred)`, keeps global top-k points (`k = int(sample_rate * N)`), and by default returns sampled graph **without** `weight` so loss is plain unweighted MSE on retained samples.
- Optional override for equal-topk branch: `sampling.adaptive_equal_cell_topk_weight_mode="area_over_count"` adds weights computed from `count / cell_area` per selected cell (normalized/clipped with the same utility).
- When this flag is enabled, `adaptive_weight_mode` is ignored unless `adaptive_equal_cell_topk_weight_mode` explicitly requests equal-topk weights.

Adaptive evaluation modes:
- `sampling.adaptive_mode="loss"`: `cell_value = sqrt(loss_variance) * cell_area`.
- `sampling.adaptive_mode="loss_no_sqrt"`: `cell_value = loss_variance * cell_area`.

4) Loss consumption of adaptive weights
- `train_utility_sampling/metalearning_sampling.py:single_image_step(...)`
- If sampled graph has `weight`, training uses weighted MSE branch instead of plain `F.mse_loss`.

5) Related helper APIs and file locations
- `train_utility_sampling/taylor_estimation.py`
  - `cell_grad_variance_estimate_with_jacrev(cell_cor_range, graph, inr, device, ...)`
  - `loss_variance_ground_truth(cell_cor_ranges, graph, inr, device)`
- `utils/quadtree.py`
  - `ImageCell` uses inclusive boundaries.
  - `HierarchicalImageGrid.evaluate_cells(...)`
  - `HierarchicalImageGrid.iterative_subdivision(...)`
  - `HierarchicalImageGrid.get_leaf_properties_tensor(...)` returning bounds/cell sizes/cell values.
- `train_utility_sampling/SamplerWrapper.py`
  - `sample_counts_poisson(values, expected_total, eps=...)`
  - `sample_variable_from_2d_intervals_vcounts(bounds, counts, device=...)`

Notes/caveats for future edits:
- Keep boundary conventions consistent (quadtree bounds are inclusive).
- Do not change adaptive sample weighting semantics without also updating weighted-loss logic in `single_image_step(...)`.
- Preserve coordinate-order expectations across helper functions (`(x, y)` in sampling, row/column order in some taylor-estimation routines).

### 5) Train INR on 2D Burgers dataset

**Step 1 — generate data (one-time setup):**
```bash
# Generates /pscratch/sd/g/gzhao27/INR/data/2D_Burgers_Sols_Nu{nu}.hdf5
# for nu in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
python utils/data/generate_burgers2d.py \
  --out_dir /pscratch/sd/g/gzhao27/INR/data \
  --H 64 --W 64 --T_steps 100 --dt 0.005 --N_train_nu 100
```
The solver uses a pseudo-spectral integrating-factor Euler scheme with 2/3 de-aliasing.
Data is pre-generated (not on-the-fly) because a single 64×64 Burgers solve takes
~5 s on CPU — unacceptable inside a DataLoader worker.

**Step 2 — train:**
```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=Burgers2D \
  data.data_path=/pscratch/sd/g/gzhao27/INR/data \
  inr.model_type=siren \
  sampling.type=random \
  sampling.rate=0.01 \
  optim.epochs=2000 \
  data.single_time_frame=10 \
  wandb.use_wandb=False
```

**Dataset class:** `GraphBurgers2D` in `utils/data/unstructure_dataset.py`
**Factory function:** `create_burgers2d_dataset(data_dir=..., ...)` — same interface as `create_burgers_dataset`.
**Graph contract:** identical to existing 2D datasets — `cor (N,2)`, `space_emb (N,2)`, `feat (N,1)`, `time (N,)`.

**Comparison with other datasets:**

| Dataset      | Class                       | Spatial dim | `input_dim` | `output_dim` | Data source                     |
|--------------|-----------------------------|-------------|-------------|---------------|---------------------------------|
| NS 2D        | `GraphNavierStokesSampling` | (H, W)      | 2           | 1             | mmap / npy / hdf5               |
| Burgers 1D   | `GraphBurgers`              | (x,)        | 1           | 1             | HDF5 tensor(N,T,x), t/x-coord  |
| Burgers 2D   | `GraphBurgers2D`            | (H, W)      | 2           | 1             | HDF5 tensor(N,T,H,W), t/x/y    |
| Piecewise 2D | `GraphPiecewise2D`          | (H, W)      | 2           | 1             | in-memory, closed-form          |
| SOMA 3D      | `GraphSomaDataset`          | (x,y,z)     | 3           | 1             | HDF5 keys foward_0…foward_99   |

### 6) Train INR on synthetic Piecewise2D dataset

The Piecewise2D dataset is a useful controlled benchmark: the left half (x ≤ 0.5) is
smooth (one frequency) while the right half also carries a high-frequency oscillation in
y, creating a spatial discontinuity in spectral content that challenges uniform sampling.

```bash
python inr_sample/single_image_inr.py \
  data.dataset_name=Piecewise2D \
  data.piecewise_resolution=256 \
  data.piecewise_alpha=10.0 \
  data.piecewise_beta=1.0 \
  data.single_time_frame=0 \
  inr.model_type=siren \
  sampling.type=random \
  sampling.rate=0.01 \
  optim.epochs=2000 \
  wandb.use_wandb=False
```

To use hard-step indicator (default): leave `data.piecewise_eps` empty.
To use a smooth sigmoid transition: set `data.piecewise_eps=0.01`.

**No pre-generation needed** — the dataset is constructed on-the-fly in `__init__`
by evaluating `f(x,y) = sin(2πx) + β·I(x>0.5)·sin(2π·α·y)` over a pixel grid.
This is O(H²) tensor math and takes microseconds.  This contrasts with Burgers 2D
where on-the-fly PDE solving would be too slow.

### 7) Evaluate Reconstruction / Error Metrics
Quick sanity tests and sampler checks:
```bash
python inr_sample/Run_Test.py inr_sampling --sample_type random
python inr_sample/Run_Test.py 2d_cluster
python inr_sample/Run_Test.py 3d_cluster
```
Single-image training path reports MSE, relative RMSE, PSNR, SSIM during validation.

### 8) Visualize Sampled Points / Learned Fields
- During training, sampled frames are saved under `sampled_frames/` and `inr_sample/sampled_frames/`.
- During validation in single-image flow, each eval step now creates
  `.../validation_i{step}/` under the sampler save root and writes:
  - `loss_heatmap.png`
  - `loss_with_samples.png`
  - `loss_with_partitions.png`
  - `reconstruction_data.npz` (for Python-side reconstruction without rerunning inference)
- `reconstruction_data.npz` currently includes:
  - metadata: `step`, `height`, `width`
  - coordinates: `coords`, `sampled_coords`
  - vector values: `gt_values`, `pred_values`, `loss_values`
  - rasterized images: `gt_image`, `pred_image`, `loss_image`
- Use existing notebooks for qualitative analysis:
  - `inr_sample/inr_sampling_show.ipynb`
  - `inr_sample/qualitative_illustration.ipynb`
  - `test/show_grid_split.ipynb`

### 9) Generate Paper Figures from W&B Logs
```bash
python Results/generate_performance_figures.py
```
This script reproduces the paper plotting assets in `Results/`:
- `trainepoch.pdf` (PSNR vs epochs)
- `traintime.pdf` (PSNR vs wall-clock training time)
- `trainrel_loss.pdf` (Relative RMSE vs wall-clock training time)

### 10) Analyze Gradient Variance/Correlation from Checkpoints
Use the standalone analyzer to replay checkpoints with repeated sampling and
track gradient stability over training iterations.

```bash
python script/inr_sample/gradient_trend_analysis.py \
  --checkpoint-parent Results/checkpoints \
  --sampler auto \
  --n-repeats 4 \
  --output-dir Results/gradient_trend_analysis
```

Notes:
- `--sampler auto` keeps each checkpoint's sampler from saved config.
- Set `--sampler random` / `NMT` / `2d_grid_linear` / `2d_grid_adaptive` to evaluate with a specific sampler.
- The script expects numeric-step checkpoints (`0.pt`, `200.pt`, ...), writes per-run CSVs, sampler-aggregated CSVs, and plots:
  - `gradient_variance_vs_step.png`
  - `gradient_correlation_vs_step.png`
  - `orthogonal_gradient_error_vs_step.png`

Orthogonal gradient error definition used in analysis scripts:
- Compute reference gradient with all points (no sampler) for each checkpoint.
- For each sampled-run gradient `g`, project onto reference direction `g_ref`.
- Orthogonal error is squared residual norm: `||g - proj_{g_ref}(g)||^2`.
- Report per-checkpoint mean/std of this error across repeated sampling runs.

For explicit four-case comparisons where multiple runs share the same sampler type
(for example adaptive top-k variants), use run-level plotting:

```bash
python script/inr_sample/compare_four_runs_gradient_trend.py \
  --checkpoint-parent Results/checkpoints \
  --output-dir Results/gradient_trend_comparison_four_runs
```

This script keeps the following as separate curves by run label:
- `random`
- `2d_grid_linear`
- `adaptive_topk_none`
- `adaptive_topk_area_over_count`

The four-run comparison script writes three images:
- `gradient_variance_comparison.png`
- `gradient_correlation_comparison.png`
- `orthogonal_gradient_error_comparison.png`

## Coding Conventions
- Follow PEP 8 and keep function/class names descriptive.
- Add type hints on new public functions and sampler interfaces.
- Keep strict separation:
  - model/network code in `utils/siren.py` (or model modules)
  - sampling logic in `train_utility_sampling/SamplerWrapper.py`
  - training orchestration in `inr_sample/*.py` and `train_utility_sampling/*.py`
- To add a new sampling strategy:
  - Implement selection logic in `train_utility_sampling/SamplerWrapper.py`.
  - Add a `sampling.type` branch in the relevant `sample(...)` method.
  - If needed, add partition preprocessing helpers near `graph_2d_cluster*` / `graph_3d_cluster`.
  - Wire the sampler through `create_inr_sampler(...)` in `inr_sample/single_image_inr.py`.
- Keep logging behavior explicit:
  - Console logs for losses/time.
  - Optional Weights & Biases via `wandb.use_wandb` in config overrides.

## Data Contract and Interfaces
- Graph data objects in training commonly carry:
  - `cor`: coordinates per node.
  - `space_emb`: normalized coordinates fed to INR.
  - `feat`: target signal values.
  - `time`: time/frame index per node.
  - `T`: number of frames (or per-graph frame counts in batches).
  - `latent_vector`: latent modulation tensor for meta-style flows.
- Shape expectations:
  - Coordinates/embeddings: `[N, d]`.
  - Signals: `[N, 1]` (or `[N, C]` for multichannel).
  - Time index: `[N]` integer labels.
- Sampler API patterns in this repo:
  - Input: graph (`cor`, `space_emb`, `feat`, `time`, `T`), model predictions or modulations when needed.
  - Output: sampled graph with subset node indices, preserving feature/coordinate alignment.
- Gradient/error statistics:
  - NMT/cluster modes rank nodes by `abs(feat - pred)`.
  - EVOS uses fitness/crossover bookkeeping in `EVOSSampler.book` and optional Laplacian-guided terms.
- Partition representation:
  - `graph.cluster_set` as list/dict mapping cluster id to node index tensor.
  - 2D single-image clustering can be `grid` or `slic`.

## Safety and Constraints
- Do not change sampling semantics (selection criteria, per-frame balancing, cluster bookkeeping) without updating all dependent training and evaluation paths.
- Keep gradient/error estimation consistent across samplers when comparing methods.
- Do not introduce hidden bias that invalidates fair baseline comparisons.
- Do not silently change default sampling distributions in `config/inr_sample.yaml`.
- Avoid large refactors in sampler/training coupling unless explicitly requested.

## Reproducibility Guidance
- Set `data.seed` and call existing seed utility (`utils/data/load_data.py:set_seed`) in entry scripts.
- For comparisons, keep fixed:
  - `data.split_ratios`, dataset path/type, frame selection, model depth/width, optimizer, total epochs.
- Log full Hydra overrides used for each run.
- Repo-specific TODO: add optional deterministic CUDA/CuDNN flags if strict bitwise reproducibility is required.

## Validation Checklist
- Run a small training job and confirm training/validation loss decreases.
- Verify sampler distribution (for clustered samplers: inspect per-cell sample counts and coverage).
- Compare adaptive sampler output against `sampling.type=random` sanity baseline.
- Re-run with fixed seed and confirm similar curves/metrics.
- Check for shape and device mismatches (`cor`, `space_emb`, `feat`, `time`, `latent_vector`).
- Run available checks:
  - `python inr_sample/Run_Test.py inr_sampling --sample_type random`
  - `python inr_sample/Run_Test.py 2d_cluster`
  - `python inr_sample/Run_Test.py 3d_cluster`
- Repo-specific TODO: add a standard lint/test command (no canonical `pytest`/lint entrypoint is currently defined).
