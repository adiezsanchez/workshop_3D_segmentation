from czifile import CziFile
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from skimage.segmentation import relabel_sequential
from skimage.morphology import remove_small_objects
from xml.etree import ElementTree as ET
from cellpose import models, core, io
from typing import Optional, Tuple

io.logger_setup()  # run this to get printing of progress

_CELLPOSE_MODEL = None

def extract_scaling_metadata(file_path: str) -> tuple[float, float, float]:
    """
    Extract physical pixel scaling information from a CZI file's metadata.

    Args:
        file_path (str): Path to the CZI file.

    Returns:
        tuple[float, float, float]: Physical pixel sizes (in micrometers) along x, y, and z axes,
            as (scaling_x_um, scaling_y_um, scaling_z_um).
    """
    with CziFile(file_path) as czi:
        # Extract the metadata XML from the CZI file
        metadata_xml = czi.metadata()

    # Parse the XML metadata
    root = ET.fromstring(metadata_xml)

    # Extract scaling information
    scaling_x = root.find('.//Scaling/Items/Distance[@Id="X"]/Value').text
    scaling_y = root.find('.//Scaling/Items/Distance[@Id="Y"]/Value').text
    scaling_z = root.find('.//Scaling/Items/Distance[@Id="Z"]/Value').text

    # Convert the string values to floats and then to micrometers for readability
    scaling_x_um = round((float(scaling_x) * 1e6), 2)
    scaling_y_um = round((float(scaling_y) * 1e6), 2)
    scaling_z_um = round((float(scaling_z) * 1e6), 2)

    return scaling_x_um, scaling_y_um, scaling_z_um

def _get_cellpose_model(require_gpu: bool = True):
    """
    Lazily initialize and cache the Cellpose model.

    This keeps module import lightweight and allows workflows that only load
    precomputed results to run on CPU-only environments.
    """
    global _CELLPOSE_MODEL

    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL

    has_gpu = core.use_gpu()
    if require_gpu and not has_gpu:
        raise RuntimeError(
            "Cellpose nuclei prediction requires GPU, but no GPU was detected. "
            "You can still run workflows that load precomputed nuclei labels."
        )

    _CELLPOSE_MODEL = models.CellposeModel(gpu=has_gpu)
    return _CELLPOSE_MODEL

def _resolve_napari_viewer(viewer):
    """
    Return a napari Viewer for optional visualization.

    If ``viewer`` is passed, it is used. Otherwise tries ``napari.current_viewer()``;
    if none exists, creates ``napari.Viewer()``. Import is deferred until visualization runs.
    """
    if viewer is not None:
        return viewer
    import napari

    v = napari.current_viewer()
    if v is not None:
        return v
    return napari.Viewer()

def _keep_objects_in_size_range(labels: np.ndarray, min_max_size: tuple[int, int]) -> np.ndarray:
    """
    Keep only labeled objects whose voxel count is within a min/max range.

    Args:
        labels (np.ndarray): Labeled image where 0 is background.
        min_max_size (tuple[int, int]): Inclusive size range as (min_size, max_size).

    Returns:
        np.ndarray: Filtered labels, relabeled sequentially from 1..N.
    """
    min_size, max_size = min_max_size
    counts = np.bincount(labels.ravel())
    keep = (counts >= max(min_size, 0)) & (counts <= max_size)
    keep[0] = False  # keep background as 0

    filtered = labels.copy()
    filtered[~keep[labels]] = 0
    filtered, _, _ = relabel_sequential(filtered)
    return filtered

def predict_nuclei_labels(image: np.ndarray, rescale_factor: float, nuclei_channel: int, min_max_nuclei_volume: tuple[int, int] | None = None, visualize=False, viewer=None) -> np.ndarray:
    """
    Predict nuclei labels using CellposeSAM using anisotropy correction.

    Args:
        image (np.ndarray): Image to predict nuclei labels from.
        rescale_factor (float): Rescale factor to apply to the Z-axis for isotropic scaling (z_um / mean([x_um, y_um])).
        nuclei_channel (int): Channel index of the nuclei channel in the image.
        min_max_nuclei_volume (tuple[int, int] | None, optional): Inclusive min/max nuclei
            volume used to filter predicted labels. Set to ``None`` to skip filtering.
            Defaults to ``None``.
        visualize (bool, optional): If True, display the predicted nuclei labels in Napari.
        viewer (optional): Napari ``Viewer`` instance. If ``visualize`` is True and this is omitted,
            the current viewer (if any) is used, otherwise a new ``napari.Viewer()`` is created.

    Returns:
        np.ndarray: Nuclei labels.
    """
    model = _get_cellpose_model(require_gpu=True)

    # Predict nuclei labels
    nuclei_labels, _ , _ = model.eval(image[nuclei_channel], do_3D=True, anisotropy=rescale_factor, z_axis=0, niter=1000)
    # Filter nuclei labels to keep only those within the specified size range.
    if min_max_nuclei_volume is not None:
        nuclei_labels = _keep_objects_in_size_range(nuclei_labels, min_max_nuclei_volume)

    # Display the resulting nuclei labels in Napari if requested.
    if visualize:
        v = _resolve_napari_viewer(viewer)
        v.add_labels(nuclei_labels)

    return nuclei_labels

def simulate_cytoplasm(
    nuclei_labels: np.ndarray,
    cytoplasm_thickness: int = 2,
    nuclei_padding: int = 0
) -> np.ndarray:
    """
    Generates approximate cytoplasmic regions surrounding each nucleus by dilating nuclear label masks.
    Optionally, an initial padding can be applied to the nuclei before cytoplasm simulation. 
    The 'cytoplasm_thickness' parameter defines how many pixels/voxels outward from the nucleus (or optionally padded nucleus) 
    to assign cytoplasmic regions. The resulting cytoplasm region for each nucleus 
    is assigned the same label, excluding the nucleus itself. Background remains 0.

    Args:
        nuclei_labels (np.ndarray): Integer-labeled nuclei mask (0 = background, >0 = nucleus).
        cytoplasm_thickness (int, optional): Number of pixels/voxels outward from the (optionally padded) 
            nucleus used to define the cytoplasmic region. Defaults to 2.
        nuclei_padding (int, optional): Number of pixels/voxels to optionally expand nuclei before cytoplasm simulation. 
            Useful to create a buffer zone between nuclei and cytoplasm. Defaults to 0 (no padding).

    Returns:
        np.ndarray: Integer-labeled cytoplasmic regions, where each region surrounding a nucleus receives the nucleus's label, and the nuclei remain excluded (set to 0).
    """
    labels_for_cytoplasm = nuclei_labels
    if nuclei_padding > 0:
        # Optionally pre-dilate nuclei to create a buffer between nuclei and cytoplasm border
        padded_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=nuclei_padding)
        labels_for_cytoplasm = cle.pull(padded_nuclei_labels)

    # Dilate labels outward by cytoplasm_thickness to generate cytoplasmic regions
    cyto_labels = cle.dilate_labels(labels_for_cytoplasm, radius=cytoplasm_thickness)
    cytoplasm = cle.pull(cyto_labels)

    # Remove (optionally padded) nucleus core regions by setting them to background (zero)
    cytoplasm[labels_for_cytoplasm > 0] = 0

    return cytoplasm

def segment_organoids_from_cp_labels(
    nuclei_labels: np.ndarray,
    cytoplasm_thickness: int,
    min_size: int = 15000
) -> np.ndarray:
    """
    Segment whole organoids from a 3D image using individual Cellpose nuclei labels as seeds.

    This function creates cytoplasmic regions around nuclei, projects them onto 2D,
    merges touching regions to form organoids, and post-processes the result to remove small objects
    and close holes. The resulting organoid labels are expanded back to 3D.

    Args:
        nuclei_labels (np.ndarray): Integer-labeled nuclei mask (3D; 0 = background, >0 = nucleus).
        cytoplasm_thickness (int): Number of pixels/voxels to dilate nuclei for simulating cytoplasm.
        min_size (int, optional): Minimum pixel area for an organoid in 2D projection. Organisms smaller
            than this are removed. Defaults to 15000.

    Returns:
        np.ndarray: Integer-labeled 3D organoid mask. Each organoid receives a unique label
            (0 = background), broadcast along the z-axis.
    """
    # Generate cytoplasm labels via dilation
    cytoplasm_labels = cle.dilate_labels(nuclei_labels, radius=cytoplasm_thickness)

    # Maximum projection across z-axis to flatten 3D labels into 2D space
    mip_labels = np.max(cytoplasm_labels, axis=0)

    # Merge touching labels to establish first organoid entities
    merged_mip = cle.merge_touching_labels(mip_labels)

    # Dilation-Erosion cycle to close holes
    dilated_labels = cle.dilate_labels(merged_mip, radius=5)
    eroded_labels = cle.erode_labels(dilated_labels, radius=1)

    # Pull from GPU in order to filter out small disconnected components and relabel using skimage
    org_labels = cle.pull(eroded_labels)
    org_labels = remove_small_objects(org_labels, min_size)

    # Relabel starting from 1
    organoid_labels = relabel_sequential(org_labels)[0]

    # Make organoid labels 3D (extend across z-axis)
    organoid_labels = np.broadcast_to(organoid_labels, cytoplasm_labels.shape)

    return organoid_labels

def map_df_column_to_labels(
    nuclei_labels: np.ndarray,
    df: "pd.DataFrame",
    value_column: str,
    label_column: str = "label",
    normalize: bool = False,
    clip_percentiles: Optional[Tuple[float, float]] = (1, 99),
    background_value: float = 0.0,
    colormap: str = "turbo",
    contrast_limits: Optional[Tuple[float, float]] = None,
    visualize: bool = False,
    viewer=None,
    layer_name: Optional[str] = None,
    ) -> np.ndarray:
    """
    Efficiently map per-label values from a DataFrame to a labeled image using vectorized lookup.

    This function assigns a value (e.g., FRET ratio) to each voxel in a labeled image
    (`nuclei_labels`) based on a lookup table constructed from a pandas DataFrame.
    It avoids per-label masking loops by using a NumPy array for direct indexing,
    making it suitable for large 2D/3D images.

    Args:
        nuclei_labels (np.ndarray):
            Labeled image where each integer value corresponds to a nucleus (0 = background).

        df (pd.DataFrame):
            DataFrame containing per-label measurements. Must include `label_column`
            and `value_column`.

        value_column (str):
            Column in `df` containing values to map (e.g., 'FRET_ratio_sum_norm').

        label_column (str, optional):
            Column in `df` containing label IDs. Default is "label".

        normalize (bool, optional):
            If True, normalize mapped values to [0, 1] after optional percentile clipping.
            Use this for raw ratios. If values are already normalized, keep False.

        clip_percentiles (tuple or None, optional):
            Percentiles (low, high) for clipping before normalization (e.g., (1, 99)).
            Set to None to disable clipping.

        background_value (float, optional):
            Value assigned to background (label 0). Default is 0.0.

        colormap (str, optional):
            Colormap to use for visualization. Default is "turbo".

        contrast_limits (tuple or None, optional):
            Optional (min, max) display limits passed to Napari when visualizing.
            Does not rescale mapped array values; independent of `normalize` and
            `clip_percentiles`.

        visualize (bool, optional):
            If True, display the result in Napari.

        viewer (napari.Viewer, optional):
            Existing Napari viewer. If None, a new one will be created.

        layer_name (str, optional):
            Name of the Napari layer. Defaults to `value_column`.

    Returns:
        np.ndarray:
            Image of same shape as `nuclei_labels`, where each voxel contains
            the mapped value corresponding to its label.
    """

    # --- Build lookup table (label -> value) ---
    max_label = int(nuclei_labels.max())
    lookup = np.full(max_label + 1, background_value, dtype=float)

    # Fill lookup table using dataframe values
    for _, row in df.iterrows():
        label_id = int(row[label_column])
        if label_id <= max_label:
            value = row[value_column]
            if not np.isnan(value):
                lookup[label_id] = value

    # --- Vectorized mapping ---
    # Each voxel gets value = lookup[label]
    out = lookup[nuclei_labels]

    # --- Optional normalization ---
    if normalize:
        mask = nuclei_labels > 0

        if np.any(mask):
            values = out[mask]

            # Optional percentile clipping
            if clip_percentiles is not None:
                p_low, p_high = np.percentile(values, clip_percentiles)
                values = np.clip(values, p_low, p_high)

            # Normalize to [0, 1]
            min_val = values.min()
            max_val = values.max()

            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.zeros_like(values)

            out[mask] = values

    # --- Optional visualization in Napari ---
    if visualize:
        v = _resolve_napari_viewer(viewer)
        name = layer_name if layer_name is not None else value_column
        image_kwargs = {
            "name": name,
            "colormap": colormap,
            "blending": "additive",
        }
        if contrast_limits is not None:
            image_kwargs["contrast_limits"] = contrast_limits
        v.add_image(out, **image_kwargs)