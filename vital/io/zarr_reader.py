"""OME-Zarr v3 Light-Sheet Microscopy Volume Reader.

Reads 3D+time microscopy volumes shaped (T=100, Z=64, Y=256, X=256) with uint16 voxel intensity.
Provides fast frame-level and slice-level access for segmentation, tracking, and visualization.
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import zarr

from vital.config import SCALE_Z, SCALE_Y, SCALE_X


class ZarrReader:
    """Reader for light-sheet microscopy 4D (T, Z, Y, X) OME-Zarr recordings."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Zarr store does not exist at: {self.path}")

        # Open store in read-only mode
        self._store = zarr.open(str(self.path), mode="r")

        # In OME-Zarr v3 for CZ Biohub dataset, volume data resides under group '0'
        if "0" in self._store:
            self._data = self._store["0"]
        elif isinstance(self._store, zarr.Array):
            self._data = self._store
        else:
            # Fallback to first array found in store
            keys = list(self._store.keys())
            if not keys:
                raise ValueError(f"No arrays found in Zarr store at {self.path}")
            self._data = self._store[keys[0]]

        self._shape: Tuple[int, ...] = tuple(self._data.shape)
        self._dtype = self._data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns (T, Z, Y, X) volume dimensions."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Returns array data type (typically uint16)."""
        return self._dtype

    @property
    def chunks(self) -> Tuple[int, ...]:
        """Returns chunk dimensions."""
        return getattr(self._data, "chunks", ())

    @property
    def num_frames(self) -> int:
        """Returns total number of timepoints (T)."""
        return self._shape[0] if len(self._shape) >= 1 else 0

    @property
    def num_z_slices(self) -> int:
        """Returns total number of Z slices (Z)."""
        return self._shape[1] if len(self._shape) >= 2 else 0

    @property
    def height(self) -> int:
        """Returns Y dimension."""
        return self._shape[2] if len(self._shape) >= 3 else 0

    @property
    def width(self) -> int:
        """Returns X dimension."""
        return self._shape[3] if len(self._shape) >= 4 else 0

    @property
    def voxel_scale(self) -> Tuple[float, float, float]:
        """Returns physical voxel scale (Z, Y, X) in micrometers."""
        return (SCALE_Z, SCALE_Y, SCALE_X)

    def get_frame(self, t: int) -> np.ndarray:
        """Retrieve 3D volume at timepoint t of shape (Z, Y, X)."""
        if not (0 <= t < self.num_frames):
            raise IndexError(f"Timepoint t={t} out of bounds for range [0, {self.num_frames - 1}]")
        volume = np.asarray(self._data[t])
        return volume

    def get_slice(self, t: int, z: int) -> np.ndarray:
        """Retrieve single 2D slice at timepoint t and depth z of shape (Y, X)."""
        if not (0 <= t < self.num_frames):
            raise IndexError(f"Timepoint t={t} out of bounds")
        if not (0 <= z < self.num_z_slices):
            raise IndexError(f"Z-slice z={z} out of bounds for range [0, {self.num_z_slices - 1}]")
        slice_2d = np.asarray(self._data[t, z])
        return slice_2d

    def get_max_projection(self, t: int, axis: int = 0) -> np.ndarray:
        """Compute maximum intensity projection (MIP) for frame t along specified axis.
        axis=0: MIP along Z -> (Y, X)
        axis=1: MIP along Y -> (Z, X)
        axis=2: MIP along X -> (Z, Y)
        """
        frame = self.get_frame(t)
        return np.max(frame, axis=axis)

    def __repr__(self) -> str:
        return (
            f"ZarrReader(path='{self.path.name}', shape={self.shape}, "
            f"dtype={self.dtype}, chunks={self.chunks})"
        )
