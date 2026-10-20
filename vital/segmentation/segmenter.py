"""3D Cell Segmentation Module.

Provides swappable segmenter backends with Cellpose pretrained 'cyto3' in 3D mode as primary.
Includes anisotropy correction based on light-sheet voxel geometry (Z=1.625 µm, Y=X=0.40625 µm).
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any
import numpy as np
import torch

from vital.config import (
    SEGMENTER_BACKEND,
    CELLPOSE_MODEL_TYPE,
    CELLPOSE_DO_3D,
    CELLPOSE_DIAMETER_VOXELS,
    SCALE_Z,
    SCALE_Y,
)


class BaseSegmenter(ABC):
    """Abstract base class for 3D microscopy cell segmenters."""

    @abstractmethod
    def segment(self, volume_3d: np.ndarray, **kwargs) -> np.ndarray:
        """Segments a 3D (Z, Y, X) intensity volume into integer instance masks.

        Args:
            volume_3d: np.ndarray of shape (Z, Y, X) containing fluorescence intensities.

        Returns:
            np.ndarray of shape (Z, Y, X) with integer labels (0=background, 1..N=cells).
        """
        pass


class CellposeSegmenter(BaseSegmenter):
    """3D cell segmenter using pretrained Cellpose models (default: 'cyto3')."""

    def __init__(
        self,
        model_type: str = CELLPOSE_MODEL_TYPE,
        do_3d: bool = CELLPOSE_DO_3D,
        diameter: Optional[float] = CELLPOSE_DIAMETER_VOXELS,
        use_gpu: Optional[bool] = None,
    ):
        from cellpose import models

        self.model_type = model_type
        self.do_3d = do_3d
        self.diameter = diameter
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

        # Initialize Cellpose model
        self.model = models.CellposeModel(
            gpu=self.use_gpu,
            model_type=self.model_type,
        )

        # Compute optical anisotropy factor: Z voxel thickness / XY voxel thickness
        self.anisotropy = SCALE_Z / SCALE_Y  # 1.625 / 0.40625 = 4.0

    def segment(self, volume_3d: np.ndarray, **kwargs) -> np.ndarray:
        """Run 3D segmentation on input volume.

        Args:
            volume_3d: Shape (Z, Y, X) numpy array.

        Returns:
            masks: Shape (Z, Y, X) integer mask array.
        """
        if volume_3d.ndim != 3:
            raise ValueError(f"Expected 3D volume (Z, Y, X), got shape {volume_3d.shape}")

        diameter = kwargs.get("diameter", self.diameter)
        anisotropy = kwargs.get("anisotropy", self.anisotropy)

        # Run 3D inference using cellpose
        eval_result = self.model.eval(
            volume_3d,
            channels=[0, 0],
            do_3D=self.do_3d,
            diameter=diameter,
            anisotropy=anisotropy,
            min_size=kwargs.get("min_size", 15),
            progress=False,
        )

        # eval returns (masks, flows, styles) or (masks, flows, styles, diams)
        masks = eval_result[0]
        return masks.astype(np.int32)


def get_segmenter(backend: Optional[str] = None, **kwargs) -> BaseSegmenter:
    """Factory function to instantiate segmenters based on configuration flag."""
    selected_backend = (backend or SEGMENTER_BACKEND).lower()

    if selected_backend == "cellpose":
        return CellposeSegmenter(**kwargs)
    elif selected_backend == "monai_unet":
        raise NotImplementedError(
            "MONAI U-Net backend is reserved as a swappable interface slot for future model upgrades."
        )
    else:
        raise ValueError(
            f"Unsupported segmenter backend: '{selected_backend}'. Supported backends: ['cellpose', 'monai_unet']"
        )
