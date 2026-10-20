import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from pathlib import Path
from skimage.measure import regionprops_table

def _map_small_to_big(
    labels_small: np.ndarray,
    labels_big: np.ndarray,
) -> dict:
    """
    Map each label in a small-labeled array (e.g., cells) to the corresponding region in a big-labeled array (e.g., organoids).

    Args:
        labels_small (np.ndarray): Labeled image (e.g., cells) whose labels will be mapped.
        labels_big (np.ndarray): Labeled image (e.g., organoids) representing larger encompassing regions (parent labels).

    Returns:
        dict: Dictionary mapping each big label (int) to a set of small labels (set of int) it contains.
    """
    mask = labels_small > 0
    pairs = np.stack([
        labels_small[mask],
        labels_big[mask]
    ], axis=1)

    # remove background overlaps
    pairs = pairs[pairs[:, 1] > 0]

    mapping = {}
    for s, b in pairs:
        mapping.setdefault(int(b), set()).add(int(s))

    return mapping


def extract_organoid_stats_and_merge(
    nuclei_labels: np.ndarray,
    organoid_labels: np.ndarray,
    props_df: pd.DataFrame,
    print_log: bool = False
) -> pd.DataFrame:
    """
    Map each cell label to its corresponding organoid, extract organoid region properties, and merge these with per-cell statistics.

    Args:
        nuclei_labels (np.ndarray): Integer-labeled nuclei mask (3D; 0 = background, >0 = nucleus).
        organoid_labels (np.ndarray): Integer-labeled organoid mask (3D; 0 = background, >0 = organoid).
        props_df (pd.DataFrame): DataFrame containing per-nucleus statistics with a 'label' and 'well_id' column.
        print_log (bool, optional): Whether to print the log of the orphan cells mapping. Defaults to False.
    Returns:
        pd.DataFrame: Merged DataFrame containing both per-cell information and associated organoid-level statistics. Includes a new 'organoid' column mapping each cell to its organoid.
    """
    # Map each cell label to each corresponding organoid (mapping in 3D space)
    mapping = _map_small_to_big(nuclei_labels, organoid_labels)

    # Invert mapping to map to props_df
    small_to_big = {}
    for b, smalls in mapping.items():
        for s in smalls:
            small_to_big.setdefault(s, set()).add(b)

    # Add organoid column to props_df
    props_df["organoid_id"] = (
        props_df["label"]
        .map(lambda s: next(iter(small_to_big[s]))
            if s in small_to_big and len(small_to_big[s]) == 1
            else 0)
        .astype(int)
    )

    # Reorder so it appears after label (no well_id in input)
    cols = list(props_df.columns)
    cols.insert(cols.index("label") + 1, cols.pop(cols.index("organoid_id")))
    props_df = props_df[cols]

    # Cells with no organoid
    n_orphans = (props_df["organoid_id"] == 0).sum()

    # Calculate percentage of orphan cells to total cells (use row count to reflect true cells after filtering)
    total_cells = len(props_df)
    perc_orphan = round(((n_orphans / total_cells) * 100), 2)

    if print_log:
        print(f"Cells mapped to no organoid: {n_orphans} - {perc_orphan}% of total cells ({total_cells})")

    # Extract area information at an organoid level and merge with the existing props_df
    organoid_regionprops_properties = [
        "label",                         # region identifier
        "area",                          # number of pixels (region size in 2D)
        "area_bbox",                     # area of axis-aligned bounding box (width × height)
        "area_convex",                   # area of convex hull of the region
        "area_filled",                   # area after filling holes
        "axis_major_length",             # length of major axis from inertia tensor (elongation)
        "axis_minor_length",             # length of minor axis (second principal axis in 2D)
        "equivalent_diameter_area",      # diameter of circle with same area as region
        "perimeter",                     # total boundary length (boundary complexity)
        "eccentricity",                  # round (0) → elongated (1), from ellipse fit
        "euler_number",                  # topology: #objects − #holes (connectivity in 2D)
        "extent",                        # area / bounding-box area (how well the box is filled)
        "feret_diameter_max",            # maximum Feret (caliper) diameter
        "solidity",                      # area / convex-hull area (compact vs lobed)
        "inertia_tensor_eigvals",        # eigenvalues of inertia tensor (2 values in 2D: shape/orientation)
    ]

    # Extract organoid features (flattened 2D labels)
    organoid_2d_labels = np.max(organoid_labels, axis=0)
    organoid_props = regionprops_table(
        label_image=organoid_2d_labels,
        properties=organoid_regionprops_properties,
    )

    # Convert to dataframe
    organoids_props_df = pd.DataFrame(organoid_props)

    # Rename columns from actual DataFrame columns (covers array properties like inertia_tensor_eigvals-0, -1)
    prefix = "organoid_id"
    rename_map = {
        col: "organoid_id" if col == "label" else f"{prefix}_{col}"
        for col in organoids_props_df.columns
    }

    organoids_props_df.rename(columns=rename_map, inplace=True)

    # Merge organoid_props and cell_props Dataframes
    final_df = props_df.merge(
        organoids_props_df,
        how="left",
        on="organoid_id"
    )

    return final_df

def _extract_csv_analysis_context(per_label_results_csv_path: Path | str) -> tuple[str, str, str]:
    """
    Extract context values encoded in the per-label results CSV path.

    Args:
        per_label_results_csv_path (Path | str): Path to a file named like
            `per_label_results_<input_id>_..._CT<ct>_NP<np>.csv`.

    Returns:
        tuple[str, str, str]: `(input_folder_id, cytoplasm_thickness, nuclei_padding)`.
    """
    csv_path = Path(per_label_results_csv_path)
    input_folder_id = csv_path.parent.stem
    filename = csv_path.name
    try:
        cytoplasm_thickness = filename.split("_CT")[1].split("_NP")[0]
        nuclei_padding = filename.split("_NP")[1].split(".")[0]
    except IndexError as exc:
        raise ValueError(
            "Could not parse CT/NP values from per-label CSV filename. "
            f"Expected '*_CT<value>_NP<value>.csv', got '{filename}'."
        ) from exc
    return input_folder_id, cytoplasm_thickness, nuclei_padding

def plot_nuclei_volume_distribution(
    final_df: pd.DataFrame,
    bp_results_directory: Path | str,
) -> None:
    """
    Plot the distribution of nuclei volumes with outlier clipping at the 1st and 99th percentiles.

    This function generates a histogram of the 'volume_nuclei' column from the provided DataFrame,
    clipping outlier values below the 1st percentile and above the 99th percentile for visualization.

    Args:
        final_df (pd.DataFrame): The DataFrame containing the nuclei volume measurements
            with a column named 'volume_nuclei'.
        bp_results_directory (Path | str): Path pointing to the per-sample results directory
            (for example, `.../bp_results/WT_organoids`).

    Returns:
        None: This function only displays the plot and does not return any value.
    """
    # Extract input folder id from per-sample results directory
    input_folder_id = Path(bp_results_directory).stem

    vmin = final_df["volume_nuclei"].quantile(0.01)
    vmax = final_df["volume_nuclei"].quantile(0.99)
    vol_norm = final_df["volume_nuclei"].clip(lower=vmin, upper=vmax)
    fig = px.histogram(
        vol_norm,
        nbins=1000,
        title=f"Nuclei Volume Distribution (1st-99th percentile clipped) - {input_folder_id}",
    )
    fig.update_layout(xaxis_title="volume_nuclei (1-99th percentile clipped)")
    fig.show()

def calculate_technical_replicates_mean_values(
    df: pd.DataFrame,
    per_label_results_csv_path: Path | str,
    min_max_nuclei_volume: tuple[int, int] | None = None
) -> pd.DataFrame:
    """
    Calculate and save mean values for technical replicates grouped by organoid identifiers.

    Groups the input DataFrame by "experiment_id", "mouse_id", "treatment_id", and "replica_id", then computes
    the per-group average for "intensity_mean_nuclei", "volume_nuclei", "intensity_mean_cyto",
    "volume_cyto", and "nuclei_cyto_ratio". The resulting DataFrame is saved as a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame containing the measurements.
        per_label_results_csv_path (Path | str): Path to the per-label CSV file. The output file
            is saved in the same directory.
        min_max_nuclei_volume (tuple[int, int] or None, optional): Tuple specifying the min and max nuclei
            volumes used for filtering, or None if not used. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing mean values of the technical replicates.
    """

    per_label_results_csv_path = Path(per_label_results_csv_path)
    input_folder_id, cytoplasm_thickness, nuclei_padding = _extract_csv_analysis_context(
        per_label_results_csv_path
    )
    
    # Group by organoid identifiers and compute per-organoid averages
    grouped_df = df.groupby(
        ["experiment_id", "mouse_id", "treatment_id", "replica_id"]
    ).agg({
        "intensity_mean_nuclei": "mean",
        "volume_nuclei": "mean",
        "intensity_mean_cyto": "mean",
        "volume_cyto": "mean",
        "nuclei_cyto_ratio": "mean",
    }).reset_index()

    # Save average results
    if min_max_nuclei_volume is not None:
        average_path = per_label_results_csv_path.parent / (
            f"tech_rep_average_nuclear_cyto_intensity_{input_folder_id}_MIN_{min_max_nuclei_volume[0]}_MAX{min_max_nuclei_volume[1]}_CT{cytoplasm_thickness}_NP{nuclei_padding}.csv"
        )
    else:
        average_path = per_label_results_csv_path.parent / (
            f"tech_rep_average_nuclear_cyto_intensity_{input_folder_id}_MIN_None_MAX_None_CT{cytoplasm_thickness}_NP{nuclei_padding}.csv"
        )
    grouped_df.to_csv(average_path, index=False)

    print(f"Cytoplasm thickness: {cytoplasm_thickness}")
    print(f"Nuclei padding: {nuclei_padding}")
    print(f"Technical replicates mean values saved to: {average_path}")

    return grouped_df

def plot_boxplots_by_features(
    df: pd.DataFrame,
    features: list[str],
    bp_results_directory: Path | str,
    hover_data: list[str] = ["experiment_id", "mouse_id", "replica_id"]
) -> None:
    """
    Plots boxplots for each specified feature against 'treatment_id'.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        features (list[str]): List of feature column names to generate boxplots for.
        bp_results_directory (Path | str): Per-sample results directory used to extract the
            input folder id for plot titles.
        hover_data (list[str], optional): List of column names to display in hover data. Defaults to 
            ["experiment_id", "mouse_id", "replica_id"].

    Returns:
        None
    """
    # Extract input folder id from per-sample results directory
    input_folder_id = Path(bp_results_directory).stem

    titles = {
        "intensity_mean_nuclei": "YAP Nuclear Intensity Average by Treatment ID",
        "intensity_mean_cyto": "YAP Cytoplasmic Intensity Average by Treatment ID",
        "nuclei_cyto_ratio": "YAP Nuclear/Cytoplasmic signal Ratio by Treatment ID",
    }

    for feat in features:
        title_str = titles.get(feat, f"{feat} by Treatment ID")
        title_str = f"{title_str} - {input_folder_id}"
        fig = px.box(
            df,
            x="treatment_id",
            y=feat,
            color="treatment_id",
            points="all",
            hover_data=hover_data,
            title=title_str,
        )
        fig.show()

def calculate_biological_replicates_mean_values(
    df: pd.DataFrame,
    per_label_results_csv_path: Path | str,
    min_max_nuclei_volume: tuple[int, int] | None = None
) -> pd.DataFrame:
    """
    Calculates mean values for biological replicates, grouped by mouse and treatment.

    Args:
        df (pd.DataFrame): Input DataFrame containing measurement data.
        per_label_results_csv_path (Path | str): Path to the per-label CSV file used to infer
            metadata and output directory.
        min_max_nuclei_volume (tuple[int, int] | None, optional): Minimum and maximum nuclei volume for filtering or filename. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of mean values per biological replicate and treatment.
    """
    per_label_results_csv_path = Path(per_label_results_csv_path)
    input_folder_id, cytoplasm_thickness, nuclei_padding = _extract_csv_analysis_context(
        per_label_results_csv_path
    )

    # Group by biological replicate and treatment, compute mean per group
    grouped_df = (
        df
        .groupby(["mouse_id", "treatment_id"])
        .agg({
            "intensity_mean_nuclei": "mean",
            "volume_nuclei": "mean",
            "intensity_mean_cyto": "mean",
            "volume_cyto": "mean",
            "nuclei_cyto_ratio": "mean",
        })
        .reset_index()
    )

    # Remove rows where 'treatment_id' contains "dmPGE2"
    grouped_df = grouped_df[~grouped_df["treatment_id"].str.contains("dmPGE2", na=False)]

    # Prepare output file path
    if min_max_nuclei_volume is not None:
        bio_rep_average_path = (
            per_label_results_csv_path.parent /
            f"bio_rep_average_nuclear_cyto_intensity_{input_folder_id}_MIN_{min_max_nuclei_volume[0]}_MAX{min_max_nuclei_volume[1]}_CT{cytoplasm_thickness}_NP{nuclei_padding}.csv"
        )
    else:
        bio_rep_average_path = (
            per_label_results_csv_path.parent /
            f"bio_rep_average_nuclear_cyto_intensity_{input_folder_id}_MIN_None_MAX_None_CT{cytoplasm_thickness}_NP{nuclei_padding}.csv"
        )

    grouped_df.to_csv(bio_rep_average_path, index=False)
    print(f"Cytoplasm thickness: {cytoplasm_thickness}")
    print(f"Nuclei padding: {nuclei_padding}")
    print(f"Biological replicates mean values saved to: {bio_rep_average_path}")

    return grouped_df

def plot_scatter_nuclei_cyto_ratio(
    df: pd.DataFrame,
    per_label_results_csv_path: Path | str,
    treatment_order: list[str],
) -> None:
    """
    Scatter plot of nuclei/cytoplasm signal ratio by treatment, with means.

    Generates a scatter plot where each point corresponds to the nuclei/cytoplasm signal
    ratio for individual biological replicates, and adds a mean indicator for each
    treatment group. The treatments are plotted in the given order.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'treatment_id',
            'mouse_id', and 'nuclei_cyto_ratio' columns.
        per_label_results_csv_path (Path | str): Path to the per-label CSV file used to
            infer input folder id for plot titles.
        treatment_order (list[str]): Ordered list of treatment IDs to control
            the appearance order on the x-axis.

    Returns:
        None: Displays the plot using plotly.
    """
    # Extract input folder id from per-label CSV path
    input_folder_id = Path(per_label_results_csv_path).parent.stem

    # Calculate means for each treatment in order
    means = (
        df
        .set_index('treatment_id')
        .loc[treatment_order]  # preserve order
        .reset_index()
        [["treatment_id", "nuclei_cyto_ratio"]]
        .groupby("treatment_id")["nuclei_cyto_ratio"]
        .mean()
        .reindex(treatment_order)
        .values
    )

    # Scatter plot for individual data points, enforcing the order on x
    fig = go.Figure()

    for treatment in treatment_order:
        sub_df = df[df["treatment_id"] == treatment]
        fig.add_trace(
            go.Scatter(
                x=[treatment] * len(sub_df),
                y=sub_df["nuclei_cyto_ratio"],
                mode="markers",
                name=treatment,
                text=sub_df["mouse_id"],
                marker=dict(size=10),
                showlegend=False  # disables individual duplicate legend entries
            )
        )

    # Add mean lines for each treatment
    for idx, (treatment, mean) in enumerate(zip(treatment_order, means)):
        fig.add_trace(
            go.Scatter(
                x=[treatment],
                y=[mean],
                mode="markers",
                name="Mean" if idx == 0 else None,
                marker=dict(
                    symbol="line-ew",
                    size=28,
                    color="black",
                    line=dict(width=3, color="black"),
                ),
                showlegend=(idx == 0),  # Only show legend entry once
            )
        )

    fig.update_layout(
        xaxis=dict(title="treatment_id", categoryorder="array", categoryarray=treatment_order),
        yaxis_title="nuclei_cyto_ratio",
        title=f"YAP Nuclear/Cytoplasmic signal Ratio by Treatment ID (Mean ± Individual Points) - {input_folder_id}",
    )

    fig.show()

def calculate_statistical_tests(
    df: pd.DataFrame,
    bp_results_directory: Path | str,
) -> None:
    """
    Calculates statistical tests for the nuclei/cytoplasm signal ratio by treatment.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'treatment_id',
            'mouse_id', and 'nuclei_cyto_ratio' columns.
        bp_results_directory (Path | str): Per-sample results directory to infer input folder
            ID for reporting output.

    Returns:
        None: Prints the statistical tests to the console.
    """
    # Extract input folder id from per-sample results directory
    input_folder_id = Path(bp_results_directory).stem
    
    # Define comparison groups
    ttest_comparisons = [
        ("MSN", "BCM"),
        ("PGE2", "BCM")
    ]

    anova_groups = [
        (["BCM", "MSN", "PGE2"], "BCM")
    ]

    print(f"==== T-TEST COMPARISONS BETWEEN GROUPS (nuclei_cyto_ratio) ==== for {input_folder_id} ====")
    for case, ctrl in ttest_comparisons:
        case_vals = df[df["treatment_id"] == case]["nuclei_cyto_ratio"]
        ctrl_vals = df[df["treatment_id"] == ctrl]["nuclei_cyto_ratio"]
        tstat, pval = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False, nan_policy='omit')
        print(f"{case} vs {ctrl}: t={tstat:.3f}, p={pval:.4g}")

    print(f"\n==== 1-WAY ANOVA AND PAIRWISE COMPARISONS (nuclei_cyto_ratio) ==== for {input_folder_id} ====")
    for groups, control in anova_groups:
        # Collect data for all groups
        group_vals = [df[df["treatment_id"] == g]["nuclei_cyto_ratio"].dropna() for g in groups]
        anova_stat, anova_p = stats.f_oneway(*group_vals)
        print(f"ANOVA for {groups}: F={anova_stat:.3f}, p={anova_p:.4g}")
        # Pairwise t-test for each group vs control
        for grp in groups:
            if grp == control:
                continue
            grp_vals = df[df["treatment_id"] == grp]["nuclei_cyto_ratio"]
            ctrl_vals = df[df["treatment_id"] == control]["nuclei_cyto_ratio"]
            tstat, pval = stats.ttest_ind(grp_vals, ctrl_vals, equal_var=False, nan_policy='omit')
            print(f"  {grp} vs {control}: t={tstat:.3f}, p={pval:.4g}")