import torch
class ImageCell:
    """
    Represents a rectangular cell in an image that can be subdivided.
    
    Note: x_end and y_end are INCLUSIVE (the last pixel in the cell)
    E.g., cell (0, 0, 63, 63) contains pixels from (0,0) to (63,63)
    
    all return: x as colume comes first, y as row comes second
    """
    def __init__(self, x_start, y_start, x_end, y_end, level=0, parent=None):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end  # inclusive
        self.y_end = y_end  # inclusive
        self.level = level
        self.parent = parent
        self.children = []  # Will hold 4 children if subdivided
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
    def area(self):
        return self.width * self.height
    
    @property
    def center(self):
        # Center of inclusive range
        return ((self.x_start + self.x_end) // 2, (self.y_start + self.y_end) // 2)
    
    def can_subdivide(self):
        """Check if cell is large enough to subdivide (at least 2x2)"""
        return self.width >= 2 and self.height >= 2
    
    def subdivide(self):
        """Split this cell into 4 quadrants (2x2)"""
        if not self.can_subdivide():
            raise ValueError(f"Cell too small to subdivide: {self.width}x{self.height}")
        
        if self.is_subdivided:
            return self.children
        
        mid_x = (self.x_start + self.x_end) // 2
        mid_y = (self.y_start + self.y_end) // 2
        
        # Create 4 children with inclusive boundaries
        # top-left, top-right, bottom-left, bottom-right
        self.children = [
            ImageCell(self.x_start, self.y_start, mid_x, mid_y, self.level + 1, self),  # top-left
            ImageCell(mid_x + 1, self.y_start, self.x_end, mid_y, self.level + 1, self),   # top-right
            ImageCell(self.x_start, mid_y + 1, mid_x, self.y_end, self.level + 1, self),  # bottom-left
            ImageCell(mid_x + 1, mid_y + 1, self.x_end, self.y_end, self.level + 1, self)     # bottom-right
        ]
        
        self.is_subdivided = True
        return self.children
    
    def get_pixel_range(self):
        """Return the pixel range as (x_start, y_start, x_end, y_end)"""
        return (self.x_start, self.y_start, self.x_end, self.y_end)
    
    def __repr__(self):
        return f"ImageCell(x/colume: {self.x_start}-{self.x_end}, y/row: {self.y_start}-{self.y_end}, L{self.level})"


class HierarchicalImageGrid:
    """
    Manages a hierarchical grid of image cells that can be dynamically subdivided.
    """
    def __init__(self, image_width, image_height, initial_grid_size=4):
        self.image_width = image_width
        self.image_height = image_height
        self.initial_grid_size = initial_grid_size
        self.root_cells = []
        self.all_cells = {}  # id -> cell mapping for quick lookup
        self.cell_counter = 0
        self._evaluation_cache = {}  # Cache for evaluation results
        
        self._initialize_grid()
    
    def _initialize_grid(self):
        """Create the initial grid (e.g., 4x4) with inclusive boundaries"""
        cell_width = self.image_width // self.initial_grid_size
        cell_height = self.image_height // self.initial_grid_size
        
        for row in range(self.initial_grid_size):
            for col in range(self.initial_grid_size):
                x_start = col * cell_width
                y_start = row * cell_height
                # x_end and y_end are inclusive (last pixel in cell)
                x_end = (x_start + cell_width - 1) if col < self.initial_grid_size - 1 else (self.image_width - 1)
                y_end = (y_start + cell_height - 1) if row < self.initial_grid_size - 1 else (self.image_height - 1)
                
                cell = ImageCell(x_start, y_start, x_end, y_end)
                self.root_cells.append(cell)
                self.all_cells[self.cell_counter] = cell
                self.cell_counter += 1
    
    def subdivide_cell(self, cell):
        """Subdivide a specific cell and return its children"""
        if not isinstance(cell, ImageCell):
            raise TypeError("Expected ImageCell object")
        
        children = cell.subdivide()
        
        # Add children to the tracking dictionary
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
        
        Args:
            device: torch device ('cpu', 'cuda', etc.)
            
        Returns:
            tuple: (bounds, cell_sizes)
                bounds: Tensor of shape (n_bins, 4) where each row is [x_start, x_end, y_start, y_end]
                        All boundaries are INCLUSIVE
                cell_sizes: List of cell areas
        """
        self.evaluate_cells(evaluation_function, use_cache=True, batch_mode=True)
        leaf_cells = self.get_leaf_cells()
        
        # Extract bounds for each cell (all inclusive)
        bounds_list = []
        cell_size = []
        cell_values = []
        for cell in leaf_cells:
            bounds_list.append([cell.x_start, cell.x_end, cell.y_start, cell.y_end])
            cell_size.append(cell.area)
            
            assert cell.cell_value is not None, "Cell value not evaluated yet."
            cell_values.append(cell.cell_value)
        # Convert to tensor
        bounds = torch.tensor(bounds_list, device=device)
        cell_size = torch.tensor(cell_size, device=device)
        cell_values = torch.tensor(cell_values, device=device)
        
        return bounds, cell_size, cell_values
    
    def get_leaf_centers_tensor(self, device='cpu', dtype=None):
        """
        Get all leaf cell centers as a PyTorch tensor.
        
        Args:
            device: torch device ('cpu', 'cuda', etc.)')
            dtype: torch dtype (default: torch.float32 for fractional centers)
            
        Returns:
            tuple: (centers, dimensions)
                centers: Tensor of shape (n_bins, 2) where each row is [center_x, center_y]
                dimensions: Tensor of widths (assuming square cells)
        """
        
        leaf_cells = self.get_leaf_cells()
        
        # Calculate centers for each cell
        centers_list = []
        dimensions_list = []
        
        for cell in leaf_cells:
            centers_list.append(cell.center)
            dimensions_list.append(cell.width)  # width is inclusive now
        
        # Convert to tensor
        centers = torch.tensor(centers_list, dtype=dtype, device=device)
        dimensions = torch.tensor(dimensions_list, dtype=dtype, device=device)
        
        return centers, dimensions
    
    def get_leaf_widths_tensor(self, device='cpu', dtype=None):
        """
        Get all leaf cell widths as a PyTorch tensor (inclusive width and height).
        
        Args:
            device: torch device ('cpu', 'cuda', etc.)
            dtype: torch dtype (default: torch.long for pixel counts)
            
        Returns:
            torch.Tensor: Shape (n_bins, 2) where each row is [width, height]
                         Width/height represent actual pixel counts (inclusive)
        """
        
        if dtype is None:
            dtype = torch.long  # Use integer type for pixel counts
            
        leaf_cells = self.get_leaf_cells()
        n_bins = len(leaf_cells)
        
        # Calculate inclusive dimensions for each cell
        dimensions_list = []
        for cell in leaf_cells:
            # Inclusive width and height (actual pixel count)
            width = cell.x_end - cell.x_start   # Already correct since x_end is exclusive internally
            height = cell.y_end - cell.y_start  # Already correct since y_end is exclusive internally
            dimensions_list.append([width, height])
        
        # Convert to tensor
        dimensions = torch.tensor(dimensions_list, dtype=dtype, device=device)
        
        return dimensions
    
    def find_cell_containing_point(self, x, y):
        """Find the leaf cell that contains the given point"""
        def search_cells(cells):
            for cell in cells:
                if (cell.x_start <= x < cell.x_end and 
                    cell.y_start <= y < cell.y_end):
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
            'levels': sorted(levels)
        }
    
    def cells_to_tensor_batch(self, cells, device='cpu', dtype=None):
        """
        Convert a list of cells to tensor format for batch processing.
        
        This helper method extracts cell boundaries and centers into tensors
        that can be used with batch evaluation functions.
        
        Args:
            cells: List of ImageCell objects
            device: torch device ('cpu', 'cuda', etc.)
            dtype: torch dtype for coordinates
            
        Returns:
            dict with keys:
                - 'centers': Tensor of shape [N, 2] with (x, y) centers
                - 'bounds': Tensor of shape [N, 4] with (x_start, y_start, x_end, y_end)
                - 'widths': Tensor of shape [N] with cell widths
                - 'cells': Original cell list for reference
        """
        if not cells:
            return {
                'centers': torch.tensor([], device=device, dtype=dtype).reshape(0, 2),
                'bounds': torch.tensor([], device=device, dtype=dtype).reshape(0, 4),
                'widths': torch.tensor([], device=device, dtype=dtype),
                'cells': []
            }
        
        centers_list = []
        bounds_list = []
        widths_list = []
        
        for cell in cells:
            centers_list.append(cell.center)
            bounds_list.append([cell.x_start, cell.y_start, cell.x_end, cell.y_end])
            widths_list.append(cell.x_end - cell.x_start)
        
        return {
            'centers': torch.tensor(centers_list, device=device, dtype=dtype),
            'bounds': torch.tensor(bounds_list, device=device, dtype=dtype),
            'widths': torch.tensor(widths_list, device=device, dtype=dtype),
            'cells': cells
        }
    
    def evaluate_cells(self, evaluation_function, use_cache=True, batch_mode=True):
        """
        Evaluate each leaf cell using the provided function, with caching.
        
        Args:
            evaluation_function: Function that evaluates cells. Expected signature:
                - If batch_mode=True: 
                    def eval_fn(cells: List[ImageCell]) -> List[float]
                    Takes a list of ImageCell objects, returns list of values
                    
                - If batch_mode=False:
                    def eval_fn(cell: ImageCell) -> float
                    Takes a single ImageCell object, returns a single value
                    
            use_cache: Whether to use cached values for previously evaluated cells
            batch_mode: If True, uses batch evaluation (more efficient for GPU operations)
            
        Returns:
            dict: Mapping of ImageCell objects to their evaluated values
            
        Example:
            # Batch mode (recommended for efficiency)
            def batch_eval(cells):
                values = []
                for cell in cells:
                    # Your evaluation logic here
                    value = cell.area * some_computation(cell.center)
                    values.append(value)
                return values
            
            grid.evaluate_cells(batch_eval, batch_mode=True)
            
            # Single cell mode (simpler but slower)
            def single_eval(cell):
                return cell.area * some_computation(cell.center)
            
            grid.evaluate_cells(single_eval, batch_mode=False)
        """
        leaf_cells = self.get_leaf_cells()
        cell_values = {}
        
        if batch_mode:
            # Separate cached and uncached cells
            uncached_cells = []
            
            for cell in leaf_cells:
                cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end)
                if use_cache and cell_key in self._evaluation_cache:
                    # Use cached value
                    value = self._evaluation_cache[cell_key]
                    cell.cell_value = value
                    cell_values[cell] = value
                else:
                    # Need to evaluate
                    uncached_cells.append(cell)
            
            # Batch evaluate all uncached cells at once
            if uncached_cells:
                # Call evaluation function with list of cells
                batch_values = evaluation_function(uncached_cells)
                
                if len(batch_values) != len(uncached_cells):
                    raise ValueError(
                        f"Evaluation function returned {len(batch_values)} values "
                        f"but expected {len(uncached_cells)} (one per cell)"
                    )

                # Update cache and cell_values
                for cell, value in zip(uncached_cells, batch_values):
                    cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end)
                    self._evaluation_cache[cell_key] = value
                    cell.cell_value = value
                    cell_values[cell] = value
        else:
            # Original iterative approach (for backward compatibility)
            for cell in leaf_cells:
                cell_key = (cell.x_start, cell.x_end, cell.y_start, cell.y_end)
                if use_cache and cell_key in self._evaluation_cache:
                    value = self._evaluation_cache[cell_key]
                else:
                    # Call evaluation function with single cell
                    value = evaluation_function(cell)
                    self._evaluation_cache[cell_key] = value
                cell.cell_value = value
                cell_values[cell] = value
        
        return cell_values
        
    def iterative_subdivision(self, evaluation_function, iterations=3, percentage=5, min_size=2, batch_mode=True):
        """
        Iteratively subdivide cells based on evaluation function for multiple rounds.
        
        Args:
            evaluation_function: Function that evaluates cells. Expected signature:
                - If batch_mode=True: 
                    def eval_fn(cells: List[ImageCell]) -> List[float]
                    
                - If batch_mode=False:
                    def eval_fn(cell: ImageCell) -> float
                    
            iterations: Number of subdivision iterations
            percentage: Percentage of top cells to subdivide in each iteration
            min_size: Minimum cell size to allow subdivision
            batch_mode: If True, use batch evaluation for efficiency
            
        Returns:
            list: Final leaf cells after all iterations
        TODO: only evaluate new leaf cells   
        Example:
            # Define evaluation function (batch mode)
            def evaluate_gradient_variance(cells):
                # Access cell properties directly
                values = []
                for cell in cells:
                    # cell.x_start, cell.x_end, cell.y_start, cell.y_end
                    # cell.center, cell.area, cell.width, cell.height
                    value = your_computation(cell)
                    values.append(value)
                return values
            
            # Run subdivision
            grid.iterative_subdivision(evaluate_gradient_variance, 
                                      iterations=5, percentage=10)
        """
        for i in range(iterations):
            # Evaluate current leaf cells using cached results where possible
            cell_values = self.evaluate_cells(evaluation_function, use_cache=True, batch_mode=batch_mode)
            
            # Break if no cells can be subdivided
            if not any(cell.can_subdivide() and cell.width >= min_size and cell.height >= min_size 
                      for cell in cell_values.keys()):
                break
                
            # Subdivide top cells
            divide_cell_num = max(1, int(len(cell_values) * percentage / 100))
            new_children = self.subdivide_top_cells(cell_values, divide_cell_num)
            
            # If no new children were created, stop iteration
            if not new_children:
                break
                
        return self.get_leaf_cells()
    
    def subdivide_top_cells(self, cell_values, n_cells_to_subdivide=5):
        """
        Subdivide the top percentage of cells based on their evaluation values.
        
        Args:
            cell_values: dict mapping cells to their evaluated values
            percentage: float, percentage of top cells to subdivide (default: 5)
        
        Returns:
            list: Newly created child cells from subdivision
        """
        if not 0 < n_cells_to_subdivide <= len(cell_values):
            raise ValueError("Number of cells to subdivide must be between 1 and the total number of cells")
        
        # Sort cells by their values in descending order
        sorted_cells = sorted(cell_values.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate number of cells to subdivide
        # n_cells_to_subdivide = max(1, int(len(sorted_cells) * percentage / 100))
        
        new_children = []
        for cell, _ in sorted_cells[:n_cells_to_subdivide]:
            if cell.can_subdivide() and not cell.is_subdivided:
                children = self.subdivide_cell(cell)
                new_children.extend(children)
        
        return new_children

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
        
        print(f"Grid Structure ({self.image_width}x{self.image_height}):")
        for i, cell in enumerate(self.root_cells):
            print(f"Root {i}:")
            print_cell(cell, 1)
    
    def draw_with_image(self, image, ax=None, show_cells=True, cell_alpha=0.3, 
                       figsize=(12, 10)):
        """
        Draw the grid overlay on top of an actual image.
        
        Args:
            image: numpy array representing the image
            ax: matplotlib axis object (if None, creates new figure)
            show_cells: whether to show cell boundaries
            cell_alpha: transparency of cell overlays
            figsize: figure size if creating new figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Display the image
        ax.imshow(image, aspect='equal')
        
        if show_cells:
            leaf_cells = self.get_leaf_cells()
            
            # Draw cell boundaries
            for cell in leaf_cells:
                x, y = cell.x_start, cell.y_start
                width, height = cell.width, cell.height
                
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    alpha=cell_alpha
                )
                ax.add_patch(rect)
        
        ax.set_xlim(0, self.image_width)
        ax.set_ylim(0, self.image_height)
        ax.set_title('Image with Hierarchical Grid Overlay')
        
        return ax


# Example usage and demonstration
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a grid for a 1000x1000 image with initial 4x4 subdivision
    grid = HierarchicalImageGrid(1000, 1000, initial_grid_size=4)
    
    print("Initial grid statistics:")
    print(grid.get_statistics())
    print()
    
    # Get some cells to work with
    leaf_cells = grid.get_leaf_cells()
    
    # Let's subdivide the first few cells
    print("Subdividing first 3 cells...")
    for i in range(min(3, len(leaf_cells))):
        cell = leaf_cells[i]
        print(f"Subdividing {cell}")
        children = grid.subdivide_cell(cell)
        print(f"  Created children: {children}")
    
    print("\nAfter subdivision:")
    print(grid.get_statistics())
    print()
    
    # Subdivide one of the children further
    new_leaf_cells = grid.get_leaf_cells()
    if len(new_leaf_cells) > 0:
        print(f"Further subdividing {new_leaf_cells[0]}")
        grid.subdivide_cell(new_leaf_cells[0])
    
    print("\nFinal statistics:")
    print(grid.get_statistics())
    
    # Print the structure (limited to first 3 levels for readability)
    print("\nGrid structure (first 3 levels):")
    grid.print_structure(max_level=2)
    
    # Demonstrate finding a cell by coordinates
    test_point = (150, 150)
    found_cell = grid.find_cell_containing_point(*test_point)
    print(f"\nCell containing point {test_point}: {found_cell}")
    
    # VISUALIZATION EXAMPLES
    print("\nGenerating visualizations...")
    
    # Example 1: Basic leaf cell visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Draw with level-based coloring
    grid.draw_leaf_cells(ax=axes[0,0], color_by_level=True)
    axes[0,0].set_title('Cells Colored by Subdivision Level')
    
    # Draw with uniform coloring
    grid.draw_leaf_cells(ax=axes[0,1], color_by_level=False, alpha=0.5)
    axes[0,1].set_title('Uniform Cell Coloring')
    
    # Create a more complex subdivision pattern for demonstration
    complex_grid = HierarchicalImageGrid(800, 600, initial_grid_size=3)
    
    # Subdivide in a pattern
    cells_to_subdivide = complex_grid.get_leaf_cells()[:6]
    for cell in cells_to_subdivide:
        complex_grid.subdivide_cell(cell)
    
    # Further subdivide some children
    new_cells = complex_grid.get_leaf_cells()[:4]
    for cell in new_cells:
        if cell.can_subdivide():
            complex_grid.subdivide_cell(cell)
    
    complex_grid.draw_leaf_cells(ax=axes[1,0])
    axes[1,0].set_title('Complex Subdivision Pattern')
    
    # Example with synthetic image overlay
    import numpy as np
    synthetic_image = np.random.rand(600, 800, 3)  # Random RGB image
    complex_grid.draw_with_image(synthetic_image, ax=axes[1,1])
    axes[1,1].set_title('Grid Overlay on Image')
    
    plt.tight_layout()
    plt.savefig('grid_visualization.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    # Example 2: Interactive-style single plot
    plt.figure(figsize=(12, 8))
    ax = grid.draw_leaf_cells(color_by_level=True, alpha=0.6, line_width=2)
    plt.savefig('leaf_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Example 3: Batch Evaluation Mode (NEW FEATURE)
    # ========================================================================
    print("\n" + "="*70)
    print("BATCH EVALUATION EXAMPLE")
    print("="*70)
    
    # Define a batch evaluation function
    # This function receives a list of cells and returns a list of values
    def batch_variance_evaluator(cells):
        """
        Example batch evaluator that computes a variance metric for each cell.
        In practice, this could be replaced with your neural network gradient
        estimation or other computationally intensive operation.
        
        Args:
            cells: List of ImageCell objects
            
        Returns:
            List of evaluation values (one per cell)
        """
        import numpy as np
        
        print(f"  Batch evaluating {len(cells)} cells at once...")
        
        # Extract all cell information at once
        centers = np.array([cell.center for cell in cells])
        areas = np.array([cell.area for cell in cells])
        
        # Simulate some computation (e.g., gradient variance estimation)
        # In real use, you might process all cells through a neural network here
        values = np.random.rand(len(cells)) * areas * 0.01
        
        return values.tolist()
    
    # Create a new grid for this demo
    batch_grid = HierarchicalImageGrid(512, 512, initial_grid_size=4)
    
    print("\nUsing batch mode (efficient):")
    import time
    
    # Batch mode - evaluates all cells at once
    start = time.time()
    cell_values_batch = batch_grid.evaluate_cells(
        batch_variance_evaluator, 
        use_cache=False, 
        batch_mode=True  # Enable batch mode
    )
    batch_time = time.time() - start
    print(f"  Evaluated {len(cell_values_batch)} cells in {batch_time:.4f} seconds")
    
    # Perform iterative subdivision using batch mode
    print("\nIterative subdivision with batch evaluation:")
    batch_grid2 = HierarchicalImageGrid(512, 512, initial_grid_size=4)
    final_cells = batch_grid2.iterative_subdivision(
        batch_variance_evaluator,
        iterations=3,
        percentage=10,
        batch_mode=True  # Use batch mode for efficiency
    )
    print(f"  Final grid has {len(final_cells)} leaf cells")
    
    # Compare with iterative mode (for backward compatibility)
    print("\nUsing iterative mode (backward compatible):")
    
    def single_variance_evaluator(x_start, y_start, x_end, y_end):
        """Traditional single-cell evaluator"""
        import numpy as np
        area = (x_end - x_start) * (y_end - y_start)
        return np.random.rand() * area * 0.01
    
    batch_grid3 = HierarchicalImageGrid(512, 512, initial_grid_size=4)
    start = time.time()
    cell_values_iter = batch_grid3.evaluate_cells(
        single_variance_evaluator,
        use_cache=False,
        batch_mode=False  # Disable batch mode
    )
    iter_time = time.time() - start
    print(f"  Evaluated {len(cell_values_iter)} cells in {iter_time:.4f} seconds")
    
    # Demonstrate the helper method for converting cells to tensors
    print("\nUsing helper method to convert cells to tensors:")
    leaf_cells = batch_grid.get_leaf_cells()
    tensor_batch = batch_grid.cells_to_tensor_batch(leaf_cells[:5], device='cpu')
    print(f"  Converted {len(tensor_batch['cells'])} cells")
    print(f"  Centers shape: {tensor_batch['centers'].shape}")
    print(f"  Bounds shape: {tensor_batch['bounds'].shape}")
    print(f"  Widths shape: {tensor_batch['widths'].shape}")
    print(f"  First cell center: {tensor_batch['centers'][0].tolist()}")
    print(f"  First cell bounds: {tensor_batch['bounds'][0].tolist()}")
    
    print("\n" + "="*70)
    print("Batch evaluation allows for efficient GPU processing!")
    print("="*70)
 