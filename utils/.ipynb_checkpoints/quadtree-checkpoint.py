class ImageCell:
    """
    Represents a rectangular cell in an image that can be subdivided.
    """
    def __init__(self, x_start, y_start, x_end, y_end, level=0, parent=None):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.level = level
        self.parent = parent
        self.children = []  # Will hold 4 children if subdivided
        self.is_subdivided = False
        self.properties = {}  # Store any cell properties (color, variance, etc.)
    
    @property
    def width(self):
        return self.x_end - self.x_start
    
    @property
    def height(self):
        return self.y_end - self.y_start
    
    @property
    def area(self):
        return self.width * self.height
    
    @property
    def center(self):
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
        
        # Create 4 children: top-left, top-right, bottom-left, bottom-right
        self.children = [
            ImageCell(self.x_start, self.y_start, mid_x, mid_y, self.level + 1, self),  # top-left
            ImageCell(mid_x, self.y_start, self.x_end, mid_y, self.level + 1, self),   # top-right
            ImageCell(self.x_start, mid_y, mid_x, self.y_end, self.level + 1, self),  # bottom-left
            ImageCell(mid_x, mid_y, self.x_end, self.y_end, self.level + 1, self)     # bottom-right
        ]
        
        self.is_subdivided = True
        return self.children
    
    def get_pixel_range(self):
        """Return the pixel range as (x_start, y_start, x_end, y_end)"""
        return (self.x_start, self.y_start, self.x_end, self.y_end)
    
    def __repr__(self):
        return f"ImageCell({self.x_start},{self.y_start}-{self.x_end},{self.y_end}, L{self.level})"


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
        
        self._initialize_grid()
    
    def _initialize_grid(self):
        """Create the initial grid (e.g., 4x4)"""
        cell_width = self.image_width // self.initial_grid_size
        cell_height = self.image_height // self.initial_grid_size
        
        for row in range(self.initial_grid_size):
            for col in range(self.initial_grid_size):
                x_start = col * cell_width
                y_start = row * cell_height
                x_end = x_start + cell_width if col < self.initial_grid_size - 1 else self.image_width
                y_end = y_start + cell_height if row < self.initial_grid_size - 1 else self.image_height
                
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


# Example usage and demonstration
if __name__ == "__main__":
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