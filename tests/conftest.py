"""Pytest fixtures for channel_heads tests.

This module provides mock objects for TopoToolbox StreamObject, FlowObject,
and GridObject to enable integration testing without requiring actual DEMs.
"""


import numpy as np
import pytest

# ============================================================================
# Mock GridObject
# ============================================================================


class MockGridObject:
    """Mock TopoToolbox GridObject for testing.

    Attributes
    ----------
    z : np.ndarray
        Elevation data (2D array).
    shape : tuple
        Shape of the z array.
    crs : str
        Coordinate reference system (mock value).
    transform : tuple
        Affine transform (mock value).
    """

    def __init__(self, z: np.ndarray, crs: str = "EPSG:4326"):
        self.z = z.astype(float)
        self.shape = z.shape
        self.crs = crs
        # Mock affine transform: (scale_x, 0, origin_x, 0, -scale_y, origin_y)
        self.transform = (1.0, 0.0, 0.0, 0.0, -1.0, z.shape[0])

    def duplicate_with_new_data(self, new_z: np.ndarray) -> "MockGridObject":
        """Create a new GridObject with different data but same metadata."""
        return MockGridObject(new_z, crs=self.crs)


# ============================================================================
# Mock FlowObject
# ============================================================================


class MockFlowObject:
    """Mock TopoToolbox FlowObject for testing.

    This mock implements the essential methods used by CouplingAnalyzer:
    - shape: grid dimensions
    - dependencemap(): compute upstream contributing area mask
    - unravel_index(): convert linear indices to (row, col)

    The flow direction is defined by a dictionary mapping each cell to its
    downstream neighbor.

    Parameters
    ----------
    shape : tuple
        Grid dimensions (rows, cols).
    flow_direction : Dict[Tuple[int, int], Tuple[int, int]]
        Mapping from (row, col) to downstream (row, col). Cells not in dict
        are sinks/outlets.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        flow_direction: dict[tuple[int, int], tuple[int, int]],
    ):
        self.shape = shape
        self._flow_direction = flow_direction
        # Build reverse mapping for upstream traversal
        self._upstream: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for src, dst in flow_direction.items():
            if dst not in self._upstream:
                self._upstream[dst] = []
            self._upstream[dst].append(src)

    def unravel_index(self, linear_idx: int) -> tuple[int, int]:
        """Convert linear index to (row, col)."""
        idx = int(linear_idx)
        r = idx // self.shape[1]
        c = idx % self.shape[1]
        return r, c

    def ravel_index(self, row: int, col: int) -> int:
        """Convert (row, col) to linear index."""
        return row * self.shape[1] + col

    def dependencemap(self, seed_grid: MockGridObject) -> MockGridObject:
        """Compute upstream contributing area from seed points.

        Returns a boolean GridObject where True indicates cells that drain
        to any of the seed points. This includes both the seed cells and all
        cells that eventually flow to them.

        Parameters
        ----------
        seed_grid : MockGridObject
            Boolean grid with True at seed locations.

        Returns
        -------
        MockGridObject
            Boolean grid with True for all upstream contributing cells.
        """
        result = np.zeros(self.shape, dtype=bool)

        # Find seed points
        seeds = list(zip(*np.where(seed_grid.z)))

        # For each seed, traverse upstream using the upstream adjacency
        for seed in seeds:
            stack = [seed]
            while stack:
                cell = stack.pop()
                if result[cell]:
                    continue
                result[cell] = True
                # Add upstream neighbors
                if cell in self._upstream:
                    for upstream_cell in self._upstream[cell]:
                        if not result[upstream_cell]:
                            stack.append(upstream_cell)

        # If no upstream cells found (seed is a channel head), just mark the seed
        # Also, we need to find ALL cells that drain to the seed, not just direct upstream
        # This requires checking all cells to see if they eventually drain to any seed
        if not any(result.flat):
            for seed in seeds:
                result[seed] = True

        return MockGridObject(result, crs=seed_grid.crs)


# ============================================================================
# Mock StreamObject
# ============================================================================


class MockStreamObject:
    """Mock TopoToolbox StreamObject for testing.

    This mock implements the essential attributes and methods used by
    coupling analysis:
    - source, target: edge arrays defining stream network topology
    - node_indices: (row_array, col_array) of all stream nodes
    - streampoi(key): return boolean mask for channel heads, outlets, confluences

    Parameters
    ----------
    node_positions : List[Tuple[int, int]]
        List of (row, col) for each stream node. Node ID is the list index.
    edges : List[Tuple[int, int]]
        List of (source_node_id, target_node_id) edges (upstream -> downstream).
    channelheads : List[int]
        Node IDs of channel heads.
    outlets : List[int]
        Node IDs of outlets.
    confluences : List[int]
        Node IDs of confluences.
    grid_shape : Tuple[int, int]
        Shape of the underlying grid (rows, cols).
    """

    def __init__(
        self,
        node_positions: list[tuple[int, int]],
        edges: list[tuple[int, int]],
        channelheads: list[int],
        outlets: list[int],
        confluences: list[int],
        grid_shape: tuple[int, int],
    ):
        self._node_positions = node_positions
        self._edges = edges
        self._channelheads = set(channelheads)
        self._outlets = set(outlets)
        self._confluences = set(confluences)
        self._grid_shape = grid_shape
        self._n_nodes = len(node_positions)

        # Build source/target arrays
        if edges:
            self.source = np.array([e[0] for e in edges], dtype=np.intp)
            self.target = np.array([e[1] for e in edges], dtype=np.intp)
        else:
            self.source = np.array([], dtype=np.intp)
            self.target = np.array([], dtype=np.intp)

        # Build node_indices as (row_array, col_array)
        rows = np.array([p[0] for p in node_positions], dtype=np.intp)
        cols = np.array([p[1] for p in node_positions], dtype=np.intp)
        self.node_indices = (rows, cols)

        # Mock transform for coordinate conversion
        self.transform = (1.0, 0.0, 0.0, 0.0, -1.0, grid_shape[0])

    def streampoi(self, key: str) -> np.ndarray:
        """Return boolean mask for points of interest.

        Parameters
        ----------
        key : str
            One of 'channelheads', 'outlets', 'confluences'.

        Returns
        -------
        np.ndarray
            Boolean array of length n_nodes, True where POI exists.
        """
        mask = np.zeros(self._n_nodes, dtype=bool)
        if key == "channelheads":
            for h in self._channelheads:
                mask[h] = True
        elif key == "outlets":
            for o in self._outlets:
                mask[o] = True
        elif key == "confluences":
            for c in self._confluences:
                mask[c] = True
        else:
            raise ValueError(f"Unknown POI key: {key}")
        return mask

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x, y coordinates of stream segments for plotting."""
        rows, cols = self.node_indices
        # Simple transform: x = col, y = grid_height - row
        xs = cols.astype(float)
        ys = (self._grid_shape[0] - rows).astype(float)
        return xs, ys


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def simple_y_network():
    """Create a simple Y-shaped stream network for testing.

    Network topology:
    ```
        0       1       (channel heads at nodes 0 and 1)
         \\     /
          \\   /
           \\ /
            2           (confluence at node 2)
            |
            3           (outlet at node 3)
    ```

    Grid layout (10x10):
    ```
    Row 0: . . 0 . . . . 1 . .   (heads at (0,2) and (0,7))
    Row 1: . . . \\ . . / . . .
    Row 2: . . . . 2 . . . . .   (confluence at (2,4))
    Row 3: . . . . | . . . . .
    Row 4: . . . . 3 . . . . .   (outlet at (4,4))
    ```

    Returns
    -------
    dict
        Dictionary with keys: 'dem', 'fd', 's', 'channelheads', 'outlets',
        'confluences', 'grid_shape'
    """
    grid_shape = (10, 10)

    # Create DEM with two ridges draining to center
    z = np.ones(grid_shape, dtype=float) * 100
    # Higher elevations at channel heads
    z[0, 2] = 150  # head 0
    z[0, 7] = 145  # head 1
    z[1, 3] = 130  # intermediate
    z[1, 6] = 125  # intermediate
    z[2, 4] = 110  # confluence
    z[3, 4] = 105  # downstream
    z[4, 4] = 100  # outlet

    dem = MockGridObject(z)

    # Define flow direction (flows downhill toward outlet)
    flow_direction = {
        (0, 2): (1, 3),  # head 0 -> intermediate
        (1, 3): (2, 4),  # intermediate -> confluence
        (0, 7): (1, 6),  # head 1 -> intermediate
        (1, 6): (2, 4),  # intermediate -> confluence
        (2, 4): (3, 4),  # confluence -> downstream
        (3, 4): (4, 4),  # downstream -> outlet
        # outlet (4,4) has no downstream - it's a sink
    }

    fd = MockFlowObject(grid_shape, flow_direction)

    # Define stream nodes (positions as (row, col))
    # Node IDs: 0=head_left, 1=head_right, 2=confluence, 3=outlet
    # Adding intermediate nodes for more realistic topology
    node_positions = [
        (0, 2),  # 0: head_left
        (0, 7),  # 1: head_right
        (1, 3),  # 2: intermediate left
        (1, 6),  # 3: intermediate right
        (2, 4),  # 4: confluence
        (3, 4),  # 5: downstream
        (4, 4),  # 6: outlet
    ]

    # Edges: upstream -> downstream
    edges = [
        (0, 2),  # head_left -> intermediate_left
        (2, 4),  # intermediate_left -> confluence
        (1, 3),  # head_right -> intermediate_right
        (3, 4),  # intermediate_right -> confluence
        (4, 5),  # confluence -> downstream
        (5, 6),  # downstream -> outlet
    ]

    channelheads = [0, 1]
    outlets = [6]
    confluences = [4]

    s = MockStreamObject(
        node_positions=node_positions,
        edges=edges,
        channelheads=channelheads,
        outlets=outlets,
        confluences=confluences,
        grid_shape=grid_shape,
    )

    return {
        "dem": dem,
        "fd": fd,
        "s": s,
        "channelheads": channelheads,
        "outlets": outlets,
        "confluences": confluences,
        "grid_shape": grid_shape,
    }


@pytest.fixture
def complex_network():
    """Create a more complex network with multiple confluences.

    Network topology:
    ```
        0   1       2   3       (4 channel heads)
         \\ /         \\ /
          4           5         (2 confluences)
           \\         /
            \\       /
             \\     /
              \\   /
               \\ /
                6               (main confluence)
                |
                7               (outlet)
    ```

    Returns
    -------
    dict
        Dictionary with network components.
    """
    grid_shape = (12, 12)

    # Create DEM
    z = np.ones(grid_shape, dtype=float) * 100
    # Heads
    z[0, 1] = 160  # head 0
    z[0, 3] = 155  # head 1
    z[0, 8] = 158  # head 2
    z[0, 10] = 152  # head 3
    # First level confluences
    z[2, 2] = 140  # confluence 4
    z[2, 9] = 138  # confluence 5
    # Main confluence
    z[6, 5] = 115  # confluence 6
    # Outlet
    z[8, 5] = 100  # outlet 7

    dem = MockGridObject(z)

    # Flow direction
    flow_direction = {
        (0, 1): (2, 2),
        (0, 3): (2, 2),
        (0, 8): (2, 9),
        (0, 10): (2, 9),
        (2, 2): (6, 5),
        (2, 9): (6, 5),
        (6, 5): (8, 5),
    }

    fd = MockFlowObject(grid_shape, flow_direction)

    # Stream nodes
    node_positions = [
        (0, 1),  # 0: head 0
        (0, 3),  # 1: head 1
        (0, 8),  # 2: head 2
        (0, 10),  # 3: head 3
        (2, 2),  # 4: confluence left
        (2, 9),  # 5: confluence right
        (6, 5),  # 6: main confluence
        (8, 5),  # 7: outlet
    ]

    edges = [
        (0, 4),  # head 0 -> conf left
        (1, 4),  # head 1 -> conf left
        (2, 5),  # head 2 -> conf right
        (3, 5),  # head 3 -> conf right
        (4, 6),  # conf left -> main conf
        (5, 6),  # conf right -> main conf
        (6, 7),  # main conf -> outlet
    ]

    channelheads = [0, 1, 2, 3]
    outlets = [7]
    confluences = [4, 5, 6]

    s = MockStreamObject(
        node_positions=node_positions,
        edges=edges,
        channelheads=channelheads,
        outlets=outlets,
        confluences=confluences,
        grid_shape=grid_shape,
    )

    return {
        "dem": dem,
        "fd": fd,
        "s": s,
        "channelheads": channelheads,
        "outlets": outlets,
        "confluences": confluences,
        "grid_shape": grid_shape,
    }


@pytest.fixture
def touching_basins_network():
    """Create a network where channel head basins touch.

    This is designed to produce touching basins for testing the
    coupling detection algorithm.

    Network layout (6x6 grid) - basins touch at row 1:
    ```
    Row 0: A 0 . . 1 B   (heads at (0,1) and (0,4))
    Row 1: A A A|B B B   (basins touch between cols 2 and 3)
    Row 2: . . C . . .   (confluence at (2,2))
    Row 3: . . | . . .
    Row 4: . . O . . .   (outlet at (4,2))
    ```

    Basin A (head 0): cells (0,0), (0,1), (1,0), (1,1), (1,2)
    Basin B (head 1): cells (0,4), (0,5), (1,3), (1,4), (1,5)
    Touch point: (1,2) in Basin A is adjacent to (1,3) in Basin B
    """
    grid_shape = (6, 6)

    # Create DEM
    z = np.array(
        [
            [140, 150, 130, 130, 145, 135],  # heads at (0,1) and (0,4)
            [130, 140, 120, 120, 135, 125],  # basin cells, touch at col 2-3
            [100, 110, 100, 110, 105, 100],  # confluence at (2,2)
            [95, 100, 95, 100, 95, 95],
            [90, 95, 90, 95, 90, 90],  # outlet at (4,2)
            [85, 90, 85, 90, 85, 85],
        ],
        dtype=float,
    )

    dem = MockGridObject(z)

    # Flow direction - define so that:
    # Basin A (head 0 at (0,1)): (0,0), (0,1), (1,0), (1,1), (1,2) all drain to (0,1)
    # Basin B (head 1 at (0,4)): (0,4), (0,5), (1,3), (1,4), (1,5) all drain to (0,4)
    flow_direction = {
        # Left basin (head 0 at (0,1))
        (0, 0): (0, 1),  # drains to head
        (1, 0): (0, 1),  # drains to head
        (1, 1): (0, 1),  # drains to head
        (1, 2): (0, 1),  # drains to head - this cell will touch basin B
        (0, 1): (2, 2),  # head drains to confluence
        # Right basin (head 1 at (0,4))
        (0, 5): (0, 4),  # drains to head
        (1, 3): (0, 4),  # drains to head - this cell is adjacent to (1,2)
        (1, 4): (0, 4),  # drains to head
        (1, 5): (0, 4),  # drains to head
        (0, 4): (2, 2),  # head drains to confluence
        # Shared downstream path
        (2, 2): (3, 2),
        (3, 2): (4, 2),
    }

    fd = MockFlowObject(grid_shape, flow_direction)

    # Stream nodes
    node_positions = [
        (0, 1),  # 0: head left
        (0, 4),  # 1: head right
        (2, 2),  # 2: confluence
        (3, 2),  # 3: downstream
        (4, 2),  # 4: outlet
    ]

    edges = [
        (0, 2),  # head left -> confluence
        (1, 2),  # head right -> confluence
        (2, 3),  # confluence -> downstream
        (3, 4),  # downstream -> outlet
    ]

    channelheads = [0, 1]
    outlets = [4]
    confluences = [2]

    s = MockStreamObject(
        node_positions=node_positions,
        edges=edges,
        channelheads=channelheads,
        outlets=outlets,
        confluences=confluences,
        grid_shape=grid_shape,
    )

    return {
        "dem": dem,
        "fd": fd,
        "s": s,
        "channelheads": channelheads,
        "outlets": outlets,
        "confluences": confluences,
        "grid_shape": grid_shape,
    }
