"""Tests for Earth plotting helpers in channel_heads.viz.earth."""

from __future__ import annotations

import importlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from channel_heads import viz as viz_package  # noqa: E402
from channel_heads.viz import earth as earth_module  # noqa: E402
from channel_heads.viz.earth import (  # noqa: E402
    plot_all_coupled_pairs_for_outlet,
    plot_all_coupled_pairs_for_outlet_3d,
    plot_coupled_pair,
    plot_outlet_view,
)


class _Transform:
    def __init__(self, height: int):
        self.height = height

    def __mul__(self, coords):
        cols, rows = coords
        return np.asarray(cols, dtype=float), self.height - np.asarray(rows, dtype=float)


class _Grid:
    def __init__(self, z: np.ndarray, extent: tuple[float, float, float, float] | None = None):
        self.z = np.asarray(z)
        self.shape = self.z.shape
        self.extent = extent or (0.0, float(self.shape[1]), 0.0, float(self.shape[0]))
        self.transform = _Transform(self.shape[0])
        x = np.linspace(self.extent[0], self.extent[1], self.shape[1])
        y = np.linspace(self.extent[3], self.extent[2], self.shape[0])
        self.coordinates = np.meshgrid(x, y)

    def duplicate_with_new_data(self, new_z: np.ndarray) -> "_Grid":
        return _Grid(np.asarray(new_z), extent=self.extent)

    def crop(self, **kwargs) -> "_Grid":
        return self

    def plot(self, ax, cmap="terrain", alpha=1.0, **kwargs):
        return ax.imshow(self.z, extent=self.extent, origin="upper", cmap=cmap, alpha=alpha)

    def plot_surface(self, ax, **kwargs):
        x, y = self.coordinates
        return ax.plot_surface(x, y, self.z, **kwargs)


class _Flow:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape

    def dependencemap(self, seed_grid: _Grid) -> _Grid:
        mask = np.asarray(seed_grid.z, dtype=bool)
        return seed_grid.duplicate_with_new_data(mask)


class _Stream:
    def __init__(self):
        self._shape = (6, 6)
        self._rows = np.array([0, 0, 2, 3, 4], dtype=np.intp)
        self._cols = np.array([1, 4, 2, 2, 2], dtype=np.intp)
        self.node_indices = (self._rows, self._cols)
        self.source = np.array([0, 1, 2, 3], dtype=np.intp)
        self.target = np.array([2, 2, 3, 4], dtype=np.intp)
        self.transform = _Transform(self._shape[0])

    def streampoi(self, key: str) -> np.ndarray:
        mask = np.zeros(5, dtype=bool)
        if key == "channelheads":
            mask[[0, 1]] = True
        elif key == "confluences":
            mask[2] = True
        elif key == "outlets":
            mask[4] = True
        else:
            raise ValueError(key)
        return mask

    def upstreamto(self, mask: np.ndarray) -> "_Stream":
        return self

    def xy(self):
        xs, ys = self.transform * np.vstack((self._cols, self._rows))
        return [
            [(xs[int(a)], ys[int(a)]), (xs[int(b)], ys[int(b)])]
            for a, b in zip(self.source, self.target)
        ]


class _Analyzer:
    def __init__(self, dem: _Grid):
        self.dem = dem

    def influence_grid(self, head_id: int) -> _Grid:
        mask = np.zeros(self.dem.shape, dtype=bool)
        if head_id == 0:
            mask[:2, :3] = True
        else:
            mask[:2, 3:] = True
        return self.dem.duplicate_with_new_data(mask)


def _inputs():
    dem = _Grid(
        np.array(
            [
                [140, 150, 130, 130, 145, 135],
                [130, 140, 120, 120, 135, 125],
                [100, 110, 100, 110, 105, 100],
                [95, 100, 95, 100, 95, 95],
                [90, 95, 90, 95, 90, 90],
                [85, 90, 85, 90, 85, 85],
            ],
            dtype=float,
        )
    )
    return _Flow(dem.shape), _Stream(), dem, _Analyzer(dem)


def _touching_df() -> pd.DataFrame:
    return pd.DataFrame([{"outlet": 4, "head_1": 0, "head_2": 1, "confluence": 2}])


def test_viz_earth_has_canonical_functions():
    assert callable(earth_module.plot_coupled_pair)
    assert callable(earth_module.plot_outlet_view)
    assert callable(earth_module.plot_all_coupled_pairs_for_outlet)
    assert callable(earth_module.plot_all_coupled_pairs_for_outlet_3d)


def test_viz_package_exports_earth_plotting_functions():
    assert viz_package.plot_coupled_pair is earth_module.plot_coupled_pair
    assert viz_package.plot_outlet_view is earth_module.plot_outlet_view


def test_plot_outlet_view_runs_headless(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    _, stream, dem, _ = _inputs()

    fig, ax = plot_outlet_view(stream, outlet_id=4, dem=dem, view_mode="crop")

    assert isinstance(fig, Figure)
    assert ax.get_title() == "Outlet 4 (crop)"
    assert len(ax.collections) >= 3
    plt.close(fig)


def test_plot_coupled_pair_runs_and_can_save(tmp_path):
    flow, stream, dem, _ = _inputs()

    fig, ax = plot_coupled_pair(
        flow,
        stream,
        dem,
        confluence_id=2,
        head_i=0,
        head_j=1,
        view_mode="crop",
        focus="masks",
    )
    output = tmp_path / "coupled_pair.png"
    fig.savefig(output)

    assert output.exists() and output.stat().st_size > 0
    assert "Heads 0, 1" in ax.get_title()
    assert len(ax.images) >= 3
    plt.close(fig)


def test_plot_all_coupled_pairs_for_outlet_runs_and_handles_empty():
    flow, stream, dem, analyzer = _inputs()

    fig, ax = plot_all_coupled_pairs_for_outlet(
        flow,
        stream,
        dem,
        analyzer,
        _touching_df(),
        outlet_id=4,
        view_mode="crop",
    )

    assert isinstance(fig, Figure)
    assert "1 coupled pairs" in ax.get_title()
    assert ax.get_legend() is not None
    plt.close(fig)

    empty_df = _touching_df().iloc[0:0]
    assert (
        plot_all_coupled_pairs_for_outlet(flow, stream, dem, analyzer, empty_df, outlet_id=4)
        == (None, None)
    )


def test_plot_all_coupled_pairs_for_outlet_3d_runs():
    flow, stream, dem, analyzer = _inputs()

    fig, ax = plot_all_coupled_pairs_for_outlet_3d(
        flow,
        stream,
        dem,
        analyzer,
        _touching_df(),
        outlet_id=4,
        view_mode="crop",
        dem_stride=2,
    )

    assert isinstance(fig, Figure)
    assert ax.get_zlabel() == "Elevation"
    assert "3D" in ax.get_title()
    plt.close(fig)
