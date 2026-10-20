from pathlib import Path
import numpy as np
import tifffile

def list_images(directory_path: str, file_format: str) -> list:
    """
    List all image files in a given directory with the specified format (extension).

    Args:
        directory_path (str): Path to the directory to search for image files.
        file_format (str): File extension (without the dot), e.g. "tif", "png", "czi".

    Returns:
        list: List of image file paths as strings.
    """
    images = []

    for file_path in sorted(
        Path(directory_path).glob(f"*.{file_format}"),
        key=lambda p: p.name.lower(),
    ):
        # Remove Control and Isotype stainings from the analysis
        if "Control" not in file_path.stem and "Isotype" not in file_path.stem and "noprimaries" not in file_path.stem:
            images.append(str(file_path))

    return images

def ensure_output_dir(
    base_output_dir: str | Path,
    input_folder_id: str,
    results_type: str
) -> Path:
    """
    Create and return the output directory used to store np.array results for one .czi input folder and results type.

    Args:
        base_output_dir (str | Path): Base output directory.
        input_folder_id (str): Name of the .czi input folder.
        results_type (str): Subdirectory name indicating the type of results being stored (e.g., "nuclei_labels").

    Returns:
        Path: Path to the output directory for the specified results type and .czi input folder.
    """
    results_dir = Path(base_output_dir) / results_type / input_folder_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Could not create {results_type} results directory: {results_dir}")

    return results_dir

def load_precomputed_results_if_available(results_dir: str | Path, image_id: str, results_type: str) -> np.ndarray | None:
    """
    Load precomputed np.array results for one image if they are already stored on disk (as .tif).

    Args:
        labels_dir (str | Path): Directory where nuclei labels are stored.
        image_id (str): Name of the image.
        results_type (str): Precomputed results being loaded, match results_type from ensure_output_directory.
        (e.g. "nuclei_labels", "root_mask", "depth_map")

    Returns:
        np.ndarray | None: Loaded results when available, otherwise None.
    """
    precomputed_results_path = Path(results_dir) / f"{image_id}_{results_type}.tif"

    if not precomputed_results_path.is_file():
        return None

    return tifffile.imread(precomputed_results_path)