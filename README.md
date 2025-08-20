## Dataset and Usage
512x512 image training utilized [PDEBench](https://github.com/pdebench/PDEBench) using [this PDE in particular](https://darus.uni-stuttgart.de/file.xhtml?fileId=133280&version=8.0).

To choose a trajectory (out of the 4 total) when using the aforementioned PDE, change the `chosen_N` variable in within the `create_ns_dataset` function in `unstructure_dataset.py`.

To choose a time frame from the PDE, alter the `data.single_time_frame` values in `nersc-scent-single-inr-sample.sh`.

## 2D Cluster Grid Scheduler
To change both the starting and ending number of segmentations that will be found in the grid, refer to `n_start` and `n_finish` in `nersc-scent-single-inr_sample.sh`. 

To enable the scheduler, set the `use_2d_cluster_grid_scheduling` variable to `True` in the sampler wrapper for single image INRs in `SamplerWrapper.py`. If this variable is set to false, the grid will only be created once, using the `n_start` number of segments.

## Visualization
**Weights and Biases**
- Paths and project names need to be swapped for your own use

**Sampled Points Snapshots**
- `generate_sampled_frames` in  `single_image_inr.py` to enable/disable creation of figures showing sampled points
- Can change the interval at which these figures are created right before validation during the training loop
- Not yet implemented for EVOS

**Per-Pixel Visualizations**
- Can generate figures for:
	- Per-pixel losses
	- Per-pixel gradient norms
	- Gradient similarity
- Enable/disable with `gradient_figures` variable in `single_image_step` function in `metalearning_sampling.py`. You can change the interval at which these images are created in this function as well.

**Loss Comparison Figures**
- Ensure `comparison_vis_output` is set to true
- Perform runs for each sampling method
- Utilize `poster_vis.ipynb` in the `visuals` folder