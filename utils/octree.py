import torch


class VoxelCell:
    """
    Represents a rectangular-cuboid cell in a 3D volume that can be subdivided.

    Note: x_end, y_end, and z_end are INCLUSIVE (the last voxel in the cell)
    E.g., cell (0, 0, 0, 63, 63, 63) contains voxels from (0,0,0) to (63,63,63)

    all return: x as column comes first, y as row second, z as depth third
    """
    def __init__(self, x_start, y_start, z_start, x_end, y_end, z_end, level=0, parent=None):
        self.x_start = x_start
        self.y_start = y_start
        self.z_start = z_start
        self.x_end = x_end  # inclusive
        self.y_end = y_end  # inclusive
        self.z_end = z_end  # inclusive
        self.level = level
        self.parent = parent
        self.children = []  # Will hold 8 children if subdivided
        self.is_subdivided = False
        self.properties = {}  # Store any cell properties (color, variance, etc.)
        self.cell_value = None  # Evaluation result, set by evaluate_cells()

    @property
    def width(self):
        return self.x_end - self.x_start + 1  # inclusive

    @property
    def height(self):
        return self.y_end - self.y_start + 1  # inclusive

    @property
    def depth(self):
        return self.z_end - self.z_start + 1  # inclusive

    @property
    def volume(self):
        return self.width * self.height * self.depth

    @property
    def center(self):
        # Center of inclusive range
        return (
            (self.x_start + self.x_end) // 2,
            (self.y_start + self.y_end) // 2,
            (self.z_start + self.z_end) // 2,
        )

    def can_subdivide(self):
        """Check if cell is large enough to subdivide (at least 2x2x2)"""
        return self.width >= 2 and self.height >= 2 and self.depth >= 2

    def edge(self, axis):
        """Inclusive edge length along axis (0=x, 1=y, 2=z)."""
        return (self.width, self.height, self.depth)[axis]

    def can_subdivide_axis(self, axis):
        """Whether this cell can be binary-split along `axis` (edge >= 2)."""
        return self.edge(axis) >= 2

    def subdivide_axis(self, axis):
        """Binary-split this cell into TWO children along `axis` at the midpoint.

        Anisotropic counterpart of `subdivide()`. The midpoint uses the same
        floor-division convention as the octree split (children are
        [start..mid] and [mid+1..end]), so a point with coord <= mid falls in
        the first child — this is exactly the partition the axis-wise ANOVA
        `gain_j` is computed against. Adds 1 region per split (vs 2^3-1 = 7 for
        the isotropic octree split).
        """
        if not self.can_subdivide_axis(axis):
            raise ValueError(f"Cell edge along axis {axis} too small to split: {self.edge(axis)}")

        if self.is_subdivided:
            return self.children

        lo = (self.x_start, self.y_start, self.z_start)
        hi = (self.x_end, self.y_end, self.z_end)
        mid = (lo[axis] + hi[axis]) // 2

        lo_a, hi_a = list(lo), list(hi)
        lo_b, hi_b = list(lo), list(hi)
        hi_a[axis] = mid
        lo_b[axis] = mid + 1

        self.children = [
            VoxelCell(lo_a[0], lo_a[1], lo_a[2], hi_a[0], hi_a[1], hi_a[2], self.level + 1, self),
            VoxelCell(lo_b[0], lo_b[1], lo_b[2], hi_b[0], hi_b[1], hi_b[2], self.level + 1, self),
        ]
        self.is_subdivided = True
        return self.children

    def subdivide(self):
        """Split this cell into 8 octants (2x2x2)"""
        if not self.can_subdivide():
            raise ValueError(f"Cell too small to subdivide: {self.width}x{self.height}x{self.depth}")

        if self.is_subdivided:
            return self.children

        mid_x = (self.x_start + self.x_end) // 2
        mid_y = (self.y_start + self.y_end) // 2
        mid_z = (self.z_start + self.z_end) // 2

        x_ranges = [(self.x_start, mid_x), (mid_x + 1, self.x_end)]
        y_ranges = [(self.y_start, mid_y), (mid_y + 1, self.y_end)]
        z_ranges = [(self.z_start, mid_z), (mid_z + 1, self.z_end)]

        self.children = [
            VoxelCell(x0, y0, z0, x1, y1, z1, self.level + 1, self)
            for (x0, x1) in x_ranges
            for (y0, y1) in y_ranges
            for (z0, z1) in z_ranges
        ]

        self.is_subdivided = True
        return self.children

    def get_voxel_range(self):
        """Return the voxel range as (x_start, y_start, z_start, x_end, y_end, z_end)"""
        return (self.x_start, self.y_start, self.z_start, self.x_end, self.y_end, self.z_end)

    def __repr__(self):
        return (
            f"VoxelCell(x: {self.x_start}-{self.x_end}, y: {self.y_start}-{self.y_end}, "
            f"z: {self.z_start}-{self.z_end}, L{self.level})"
        )


class HierarchicalVoxelGrid:
    """
    Manages a hierarchical grid of voxel cells that can be dynamically subdivided.

    Unlike HierarchicalImageGrid, grid_shape is a (Dx, Dy, Dz) tuple rather than a
    single square width — 3D volumetric datasets (e.g. SOMA ocean data) are not cubic.
    """
    def __init__(self, grid_shape, initial_grid_size=4):
        """
        initial_grid_size: scalar (same cell COUNT on every axis -- the original
            behaviour; for a non-cubic volume this makes non-cubic initial cells,
            e.g. 8 on a (512,512,64) volume gives 64x64x8 cells) or a length-3
            sequence (nx, ny, nz) of per-axis counts, e.g. to make the initial
            cells actual cubes: (16, 16, 2) on (512,512,64) gives 32x32x32 cells.
        """
        self.grid_shape = tuple(grid_shape)
        if isinstance(initial_grid_size, (list, tuple)):
            gs = tuple(int(v) for v in initial_grid_size)
            if len(gs) != 3:
                raise ValueError(
                    f"initial_grid_size must be a scalar or length-3 sequence, got {initial_grid_size}"
                )
            self.initial_grid_size = gs
        else:
            self.initial_grid_size = (int(initial_grid_size),) * 3
        self.root_cells = []
        self.all_cells = {}  # id -> cell mapping for quick lookup
        self.cell_counter = 0
        self._evaluation_cache = {}  # Cache for evaluation results

        self._initialize_grid()

    def _initialize_grid(self):
        """Create the initial nx*ny*nz grid with inclusive boundaries"""
        Dx, Dy, Dz = self.grid_shape
        nx, ny, nz = self.initial_grid_size
        cell_width = max(1, Dx // nx)
        cell_height = max(1, Dy // ny)
        cell_depth = max(1, Dz // nz)

        for xi in range(nx):
            x_start = xi * cell_width
            if x_start >= Dx:
                break
            x_end = (x_start + cell_width - 1) if xi < nx - 1 else (Dx - 1)
            for yi in range(ny):
                y_start = yi * cell_height
                if y_start >= Dy:
                    break
                y_end = (y_start + cell_height - 1) if yi < ny - 1 else (Dy - 1)
                for zi in range(nz):
                    z_start = zi * cell_depth
                    if z_start >= Dz:
                        break
                    z_end = (z_start + cell_depth - 1) if zi < nz - 1 else (Dz - 1)

                    cell = VoxelCell(x_start, y_start, z_start, x_end, y_end, z_end)
                    self.root_cells.append(cell)
                    self.all_cells[self.cell_counter] = cell
                    self.cell_counter += 1

    def subdivide_cell(self, cell):
        """Subdivide a specific cell and return its children"""
        if not isinstance(cell, VoxelCell):
            raise TypeError("Expected VoxelCell object")

        children = cell.subdivide()

        for child in children:
            self.all_cells[self.cell_counter] = child
            self.cell_counter += 1

        return children

    def subdivide_cell_axis(self, cell, axis):
        """Binary-split a specific cell along `axis` and return its two children."""
        if not isinstance(cell, VoxelCell):
            raise TypeError("Expected VoxelCell object")

        children = cell.subdivide_axis(axis)

        for child in children:
            self.all_cells[self.cell_counter] = child
            self.cell_counter += 1

        return children

    def get_leaf_cells(self):
        """Get all cells that are not subdivided (leaf nodes)"""
        leaf_cells = []

        def collect_leaves(cells):
            for cell in cells:
                if cell.is_subdivided:
                    collect_leaves(cell.children)
                else:
                    leaf_cells.append(cell)

        collect_leaves(self.root_cells)
        return leaf_cells

    def get_cells_at_level(self, level):
        """Get all cells at a specific subdivision level"""
        cells_at_level = []

        def collect_at_level(cells, target_level):
            for cell in cells:
                if cell.level == target_level:
                    cells_at_level.append(cell)
                if cell.is_subdivided and cell.level < target_level:
                    collect_at_level(cell.children, target_level)

        collect_at_level(self.root_cells, level)
        return cells_at_level

    def get_leaf_properties_tensor(self, evaluation_function, device='cpu'):
        """
        Get all leaf cell bounds as a PyTorch tensor.

        Returns:
            tuple: (bounds, cell_volumes, cell_values)
                bounds: Tensor of shape (n_bins, 6), each row
                        [x_start, x_end, y_start, y_end, z_start, z_end] (all inclusive)
                cell_volumes: List of cell volumes
                cell_values: List of cell value estimates (variance of gradient or loss times volume)
        """
        self.evaluate_cells(evaluation_function, use_cache=True, batch_mode=True)
        leaf_cells = self.get_leaf_cells()

        bounds_list = []
        cell_volume = []
        cell_values = []
        for cell in leaf_cells:
            bounds_list.append([cell.x_start, cell.x_end, cell.y_start, cell.y_end, cell.z_start, cell.z_end])
            cell_volume.append(cell.volume)

            assert cell.cell_value is not None, "Cell value not evaluated yet."
            cell_values.append(cell.cell_value)

        bounds = torch.tensor(bounds_list, device=device)
        cell_volume = torch.tensor(cell_volume, device=device)
        cell_values = torch.tensor(cell_values, device=device)

        return bounds, cell_volume, cell_values

    def get_leaf_centers_tensor(self, device='cpu', dtype=None):
        """
        Get all leaf cell centers as a PyTorch tensor.

        Returns:
            tuple: (centers, dimensions)
                centers: Tensor of shape (n_bins, 3) where each row is [center_x, center_y, center_z]
                dimensions: Tensor of shape (n_bins, 3) with [width, height, depth] (cells need not be cubic)
        """
        leaf_cells = self.get_leaf_cells()

        centers_list = []
        dimensions_list = []

        for cell in leaf_cells:
            centers_list.append(cell.center)
            dimensions_list.append((cell.width, cell.height, cell.depth))

        centers = torch.tensor(centers_list, dtype=dtype, device=device)
        dimensions = torch.tensor(dimensions_list, dtype=dtype, device=device)

        return centers, dimensions

    def get_leaf_widths_tensor(self, device='cpu', dtype=None):
        """
        Get all leaf cell widths/heights/depths as a PyTorch tensor (inclusive voxel counts).

        Returns:
            torch.Tensor: Shape (n_bins, 3) where each row is [width, height, depth]
        """
        if dtype is None:
            dtype = torch.long

        leaf_cells = self.get_leaf_cells()

        dimensions_list = []
        for cell in leaf_cells:
            dimensions_list.append([cell.width, cell.height, cell.depth])

        return torch.tensor(dimensions_list, dtype=dtype, device=device)

    def find_cell_containing_point(self, x, y, z):
        """Find the leaf cell that contains the given point"""
        def search_cells(cells):
            for cell in cells:
                if (cell.x_start <= x <= cell.x_end and
                        cell.y_start <= y <= cell.y_end and
                        cell.z_start <= z <= cell.z_end):
                    if cell.is_subdivided:
                        return search_cells(cell.children)
                    else:
                        return cell
            return None

        return search_cells(self.root_cells)

    def get_statistics(self):
        """Get statistics about the grid structure"""
        leaf_cells = self.get_leaf_cells()
        levels = set(cell.level for cell in self.all_cells.values())

        return {
            'total_cells': len(self.all_cells),
            'leaf_cells': len(leaf_cells),
            'max_level': max(levels) if levels else 0,
            'levels': sorted(levels),
        }

    def evaluate_cells(self, evaluation_function, use_cache=True, batch_mode=True):
        """
        Evaluate each leaf cell using the provided function, with caching.

        Args:
            evaluation_function: Function that evaluates cells. Expected signature:
                - If batch_mode=True:
                    def eval_fn(cells: List[VoxelCell]) -> List[float]
                - If batch_mode=False:
                    def eval_fn(cell: VoxelCell) -> float
            use_cache: Whether to use cached values for previously evaluated cells
            batch_mode: If True, uses batch evaluation (more efficient for GPU operations)

        Returns:
            dict: Mapping of VoxelCell objects to their evaluated values
        """
        leaf_cells = self.get_leaf_cells()
        cell_values = {}

        if batch_mode:
            uncached_cells = []

            for cell in leaf_cells:
                cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end, cell.z_start, cell.z_end)
                if use_cache and cell_key in self._evaluation_cache:
                    value = self._evaluation_cache[cell_key]
                    cell.cell_value = value
                    cell_values[cell] = value
                else:
                    uncached_cells.append(cell)

            if uncached_cells:
                batch_values = evaluation_function(uncached_cells)

                if len(batch_values) != len(uncached_cells):
                    raise ValueError(
                        f"Evaluation function returned {len(batch_values)} values "
                        f"but expected {len(uncached_cells)} (one per cell)"
                    )

                for cell, value in zip(uncached_cells, batch_values):
                    cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end, cell.z_start, cell.z_end)
                    self._evaluation_cache[cell_key] = value
                    cell.cell_value = value
                    cell_values[cell] = value
        else:
            for cell in leaf_cells:
                cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end, cell.z_start, cell.z_end)
                if use_cache and cell_key in self._evaluation_cache:
                    value = self._evaluation_cache[cell_key]
                else:
                    value = evaluation_function(cell)
                    self._evaluation_cache[cell_key] = value
                cell.cell_value = value
                cell_values[cell] = value

        return cell_values

    def iterative_subdivision(self, evaluation_function, iterations=3, percentage=5, min_size=2, batch_mode=True,
                              split_mode="isotropic", target_num_regions=None, max_pass_cap=50,
                              min_cell_edge=2):
        """
        Iteratively subdivide cells based on evaluation function for multiple rounds.

        Args:
            evaluation_function: Function that evaluates cells. Its contract
                depends on `split_mode`:
                  - "isotropic": eval_fn(cells) -> list[float] of per-cell scores;
                    each selected cell is split into 8 octants (unchanged behaviour).
                  - "axiswise": eval_fn(cells) -> (scores, best_axis, valid, rho)
                    of length len(cells); each selected cell is binary-split along
                    its own best_axis at the midpoint.
            iterations: Number of subdivision iterations (isotropic only; axiswise
                is driven by `target_num_regions`).
            percentage: Percentage of top cells to subdivide in each iteration.
            min_size: Minimum cell size (per axis) to allow subdivision (isotropic).
            batch_mode: If True, use batch evaluation for efficiency.
            split_mode: "isotropic" (default; octree 2^d split) or "axiswise"
                (binary per-axis split). Isotropic reproduces the pre-change path
                bit-for-bit.
            target_num_regions: axiswise only — keep issuing passes until the leaf
                count K reaches this target (region-count matched to isotropic).
            max_pass_cap: axiswise only — hard cap on refinement passes.
            min_cell_edge: axiswise only — a cell is not split if its edge along the
                chosen axis is below this (in voxels).

        Returns:
            list: Final leaf cells after all iterations
        """
        if split_mode == "axiswise":
            return self._axiswise_subdivision(
                evaluation_function, percentage=percentage,
                target_num_regions=target_num_regions, max_pass_cap=max_pass_cap,
                min_cell_edge=min_cell_edge,
            )
        elif split_mode != "isotropic":
            raise ValueError(f"Unknown split_mode '{split_mode}' (expected 'isotropic' or 'axiswise')")

        for i in range(iterations):
            cell_values = self.evaluate_cells(evaluation_function, use_cache=True, batch_mode=batch_mode)

            if not any(cell.can_subdivide() and cell.width >= min_size
                       and cell.height >= min_size and cell.depth >= min_size
                       for cell in cell_values.keys()):
                break

            divide_cell_num = max(1, int(len(cell_values) * percentage / 100))
            new_children = self.subdivide_top_cells(cell_values, divide_cell_num)

            if not new_children:
                break

        return self.get_leaf_cells()

    def subdivide_top_cells(self, cell_values, n_cells_to_subdivide=5):
        """
        Subdivide the top-valued cells based on their evaluation values.

        Args:
            cell_values: dict mapping cells to their evaluated values
            n_cells_to_subdivide: number of top cells to subdivide

        Returns:
            list: Newly created child cells from subdivision
        """
        if not 0 < n_cells_to_subdivide <= len(cell_values):
            raise ValueError("Number of cells to subdivide must be between 1 and the total number of cells")

        sorted_cells = sorted(cell_values.items(), key=lambda x: x[1], reverse=True)

        new_children = []
        for cell, _ in sorted_cells[:n_cells_to_subdivide]:
            if cell.can_subdivide() and not cell.is_subdivided:
                children = self.subdivide_cell(cell)
                new_children.extend(children)

        return new_children

    def _axiswise_subdivision(self, evaluation_function, percentage=20.0,
                              target_num_regions=None, max_pass_cap=50, min_cell_edge=2):
        """Region-count-driven anisotropic refinement (binary per-axis splits).

        Each pass:
          1. eval_fn(leaves) -> (scores, best_axis, valid, rho) length-K sequences.
             score_k = (1 - rho_k) * N_k * sigma_r,k  (see task derivation);
             valid_k already folds in the sample-count / degenerate-split guards.
          2. Keep cells that are valid, splittable along best_axis, and whose edge
             along best_axis >= min_cell_edge.
          3. Rank by score, binary-split the top `percentage%` each along its own
             best_axis. A binary split adds exactly 1 region.
        Repeats until K >= target_num_regions or no candidate remains or the pass
        cap is hit. Per-pass diagnostics are recorded in self.axiswise_diagnostics.
        """
        self.axiswise_diagnostics = []
        min_edge = max(2, int(min_cell_edge))

        for pass_idx in range(int(max_pass_cap)):
            leaves = self.get_leaf_cells()
            K = len(leaves)
            if target_num_regions is not None and K >= target_num_regions:
                break

            scores, best_axis, valid, rho = evaluation_function(leaves)

            # Assemble split candidates with all guards applied.
            candidates = []
            for cell, s, ax, v in zip(leaves, scores, best_axis, valid):
                ax = int(ax)
                if not v:
                    continue
                if not cell.can_subdivide_axis(ax):
                    continue
                if cell.edge(ax) < min_edge:
                    continue
                candidates.append((float(s), cell, ax))

            if not candidates:
                break

            candidates.sort(key=lambda t: t[0], reverse=True)

            n_split = max(1, int(K * percentage / 100.0))
            if target_num_regions is not None:
                # each binary split adds exactly one region; don't overshoot.
                n_split = min(n_split, target_num_regions - K)
            if n_split <= 0:
                break

            chosen = candidates[:n_split]
            axis_hist = [0, 0, 0]
            for _, cell, ax in chosen:
                self.subdivide_cell_axis(cell, ax)
                axis_hist[ax] += 1

            # ---- per-pass diagnostics (task deliverable) ----
            leaves_after = self.get_leaf_cells()
            aspect = []
            for c in leaves_after:
                edges = (c.width, c.height, c.depth)
                aspect.append(max(edges) / max(1, min(edges)))
            # rho aligned to the cells actually split this pass
            chosen_cells = {id(cell) for _, cell, _ in chosen}
            rho_of_split = [float(r) for cell, r in zip(leaves, rho) if id(cell) in chosen_cells]
            self.axiswise_diagnostics.append({
                "pass": pass_idx,
                "K_before": K,
                "K_after": len(leaves_after),
                "n_split": len(chosen),
                "axis_hist": axis_hist,  # [nx, ny, nz] among split cells
                "rho_split_mean": (sum(rho_of_split) / len(rho_of_split)) if rho_of_split else None,
                "rho_split": rho_of_split,
                "aspect_mean": sum(aspect) / len(aspect),
                "aspect_max": max(aspect),
            })

        return self.get_leaf_cells()

    def print_structure(self, max_level=None):
        """Print the hierarchical structure for debugging"""
        def print_cell(cell, indent=0):
            if max_level is not None and cell.level > max_level:
                return

            prefix = "  " * indent
            status = "SUBDIVIDED" if cell.is_subdivided else "LEAF"
            print(f"{prefix}{cell} - {status}")

            if cell.is_subdivided:
                for child in cell.children:
                    print_cell(child, indent + 1)

        print(f"Grid Structure ({self.grid_shape[0]}x{self.grid_shape[1]}x{self.grid_shape[2]}):")
        for i, cell in enumerate(self.root_cells):
            print(f"Root {i}:")
            print_cell(cell, 1)
