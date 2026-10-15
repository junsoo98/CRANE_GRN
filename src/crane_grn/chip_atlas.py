"""ChIP-Atlas querying and region-by-factor matrix construction."""

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .regions import canonicalize_regions, parse_region


CHIP_ATLAS_COLUMNS = [
    "file_name",
    "genome_assembly",
    "track_type_class",
    "track_type",
    "cell_type_class",
    "cell_type",
    "threshold",
    "experimental_ids",
]

# PAPER_MODIFIER_MARKERS = {
#     "EP300": ["H3K27ac", "H3K9ac"],
#     "CREBBP": ["H3K27ac", "H3K9ac"],
#     "KAT2A": ["H3K27ac", "H3K9ac"],
#     "KAT2B": ["H3K27ac", "H3K9ac"],
#     "KDM6A": ["H3K27me3"],
#     "KDM6B": ["H3K27me3"],
# }

# PAPER_MODIFIER_MODES = {
#     "EP300": "overlap",
#     "CREBBP": "overlap",
#     "KAT2A": "overlap",
#     "KAT2B": "overlap",
#     "KDM6A": "exclude",
#     "KDM6B": "exclude",
# }


def select_available_factors(factor_candidates, rna_gene_names):
    """Split candidate regulatory factors by presence in the RNA feature axis."""
    candidates = pd.Index(pd.Series(factor_candidates).dropna().astype(str).unique())
    genes = set(map(str, rna_gene_names))
    available = [factor for factor in candidates if factor in genes]
    missing = [factor for factor in candidates if factor not in genes]
    return available, missing


def load_chip_atlas_filelist(path):
    """Load a ChIP-Atlas ``fileList.tab`` table, including gzip-compressed input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("ChIP-Atlas file list was not found: {}".format(path))
    with path.open("rb") as handle:
        magic = handle.read(2)
    compression = "gzip" if magic == b"\x1f\x8b" else None
    table = pd.read_csv(
        str(path), sep="\t", header=None, compression=compression, dtype=str
    )
    if table.shape[1] != len(CHIP_ATLAS_COLUMNS):
        raise ValueError(
            "ChIP-Atlas file list has {} columns; expected {}.".format(
                table.shape[1], len(CHIP_ATLAS_COLUMNS)
            )
        )
    table.columns = CHIP_ATLAS_COLUMNS
    return table


def filter_chip_atlas_metadata(
    metadata,
    factors,
    histone_marks,
    genome="hg38",
    cell_types="All cell types",
    threshold="10",
):
    """Filter ChIP-Atlas metadata for factors, marks, tissue, and threshold."""
    required = set(CHIP_ATLAS_COLUMNS)
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError("ChIP-Atlas metadata is missing columns: {}".format(missing))
    if factors is None or histone_marks is None:
        raise ValueError("Both factors and histone_marks must be provided.")
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    requested = set(map(str, factors)) | set(map(str, histone_marks))
    selected = metadata[
        (metadata["genome_assembly"].astype(str) == str(genome))
        & metadata["track_type"].astype(str).isin(requested)
        & (metadata["threshold"].astype(str) == str(threshold))
        & (
            metadata["cell_type_class"].astype(str).isin(cell_types)
            | metadata["cell_type"].astype(str).isin(cell_types)
        )
    ].copy()
    return selected.reset_index(drop=True)


def download_chip_atlas_tracks(
    selected_metadata,
    output_dir,
    base_url="https://chip-atlas.dbcls.jp/data",
    overwrite=False,
):
    """Download selected assembled BED tracks with atomic finalization.

    A failed transfer leaves no apparently complete destination file. The
    returned table records the local path for each metadata row.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in selected_metadata.iterrows():
        genome = str(row["genome_assembly"])
        file_name = str(row["file_name"])
        destination = output_dir / (file_name + ".bed")
        url = "{}/{}/assembled/{}.bed".format(
            base_url.rstrip("/"), genome, file_name
        )
        if overwrite or not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".part")
            if partial.exists():
                partial.unlink()
            try:
                with urllib.request.urlopen(url) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if partial.stat().st_size == 0:
                    raise IOError("Downloaded track is empty: {}".format(url))
                os.replace(str(partial), str(destination))
            except (OSError, urllib.error.URLError) as error:
                if partial.exists():
                    partial.unlink()
                raise RuntimeError(
                    "Failed to download ChIP-Atlas track {}: {}".format(url, error)
                )
        record = row.to_dict()
        record["local_path"] = str(destination)
        rows.append(record)
    return pd.DataFrame(rows)


def query_chip_atlas(
    filelist_path,
    output_dir,
    factors,
    histone_marks,
    genome="hg38",
    cell_types="All cell types",
    threshold="10",
    overwrite=False,
):
    """Filter a ChIP-Atlas file list and download the corresponding tracks."""
    metadata = load_chip_atlas_filelist(filelist_path)
    selected = filter_chip_atlas_metadata(
        metadata,
        factors=factors,
        histone_marks=histone_marks,
        genome=genome,
        cell_types=cell_types,
        threshold=threshold,
    )
    if selected.empty:
        raise ValueError("The ChIP-Atlas query returned no tracks.")
    return download_chip_atlas_tracks(
        selected, output_dir=output_dir, overwrite=overwrite
    )


def resolve_track_files(data_dir, targets):
    """Resolve one unambiguous BED file for each requested target.

    Exact stem matches are preferred. Otherwise a target must occur in exactly
    one BED filename; ambiguous or missing targets raise an error.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError("Track directory was not found: {}".format(data_dir))
    files = sorted(path for path in data_dir.iterdir() if path.suffix == ".bed")
    resolved = {}
    for target in map(str, targets):
        exact = [path for path in files if path.stem == target]
        matches = exact if exact else [path for path in files if target in path.name]
        if not matches:
            raise FileNotFoundError("No BED track matched target {!r}.".format(target))
        if len(matches) > 1:
            raise ValueError(
                "Multiple BED tracks matched target {!r}: {}".format(
                    target, [path.name for path in matches]
                )
            )
        resolved[target] = str(matches[0])
    return resolved


def _pyranges_module():
    try:
        import pyranges as pr
    except ImportError as error:
        raise ImportError(
            "PyRanges is required for ChIP-Atlas interval operations."
        ) from error
    return pr


def _compression_for_bed(path):
    with Path(path).open("rb") as handle:
        return "gzip" if handle.read(2) == b"\x1f\x8b" else None


def load_bed_as_ranges(path, header=0):
    """Load the first three columns of a BED-like file as a PyRanges object."""
    pr = _pyranges_module()
    frame = pd.read_csv(
        path,
        sep="\t",
        header=header,
        usecols=[0, 1, 2],
        names=["Chromosome", "Start", "End"],
        compression=_compression_for_bed(path),
    )
    frame["Chromosome"] = frame["Chromosome"].astype(str)
    frame["Start"] = pd.to_numeric(frame["Start"], errors="raise").astype(np.int64)
    frame["End"] = pd.to_numeric(frame["End"], errors="raise").astype(np.int64)
    return pr.PyRanges(frame)


def filter_modifier_track(
    modifier_bed_path,
    marker_bed_paths,
    mode="overlap",
    use_overlap_segments=False,
    min_overlaps=1,
):
    """Filter a modifier track by overlap with one or more histone-mark tracks."""
    pr = _pyranges_module()
    modifier = load_bed_as_ranges(modifier_bed_path)
    if isinstance(marker_bed_paths, (str, Path)):
        marker_bed_paths = [marker_bed_paths]
    markers = [load_bed_as_ranges(path) for path in marker_bed_paths]
    if not markers:
        raise ValueError("At least one histone-marker track is required.")
    marker_union = pr.concat(markers).merge() if len(markers) > 1 else markers[0]

    if mode == "overlap" and use_overlap_segments:
        return modifier.overlap(marker_union)
    annotated = modifier.count_overlaps(marker_union)
    if mode == "overlap":
        return annotated[annotated.NumberOverlaps >= int(min_overlaps)]
    if mode == "exclude":
        if use_overlap_segments:
            raise ValueError("use_overlap_segments is incompatible with exclusion.")
        return annotated[annotated.NumberOverlaps < int(min_overlaps)]
    raise ValueError("Unknown histone-modifier filter mode: {!r}".format(mode))


def build_filtered_modifier_tracks(
    factor_tracks,
    histone_tracks,
    modifier_to_markers=None,
    modifier_to_mode=None,
    use_overlap_segments=False,
    min_overlaps=1,
):
    """Apply the paper histone-marker rules to available modifier tracks."""
    marker_map = modifier_to_markers 
    mode_map = modifier_to_mode 
    filtered = {}
    for modifier, markers in marker_map.items():
        if modifier not in factor_tracks:
            continue
        usable = [histone_tracks[mark] for mark in markers if mark in histone_tracks]
        if not usable:
            continue
        filtered[modifier] = filter_modifier_track(
            factor_tracks[modifier],
            usable,
            mode=mode_map.get(modifier, "overlap"),
            use_overlap_segments=use_overlap_segments,
            min_overlaps=min_overlaps,
        )
    return filtered


def regions_to_bed(regions, canonical_chromosomes_only=False):
    """Convert region labels into a three-column BED DataFrame."""
    rows = []
    for region in regions:
        chromosome, start, end = parse_region(region)
        rows.append((chromosome, start, end))
    frame = pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])
    if canonical_chromosomes_only:
        canonical = {"chr{}".format(i) for i in range(1, 23)}
        canonical.update(["chrX", "chrY", "chrM"])
        frame = frame[frame["Chromosome"].isin(canonical)]
    return frame.drop_duplicates().sort_values(
        ["Chromosome", "Start", "End"]
    ).reset_index(drop=True)


def build_region_by_factor(factor_tracks, region_bed_path):
    """Build a binary region-by-factor matrix from BED interval overlaps.

    Parameters
    ----------
    factor_tracks
        Mapping from factor names to BED paths or PyRanges objects.
    region_bed_path
        Headerless BED file defining the ordered region universe.
    """
    regions = load_bed_as_ranges(region_bed_path, header=None)
    labels = canonicalize_regions(
        regions.df.apply(
            lambda row: "{}:{}-{}".format(
                row["Chromosome"], row["Start"], row["End"]
            ),
            axis=1,
        )
    )
    if labels.has_duplicates:
        raise ValueError("Region BED contains duplicate coordinates.")
    matrix = pd.DataFrame(0, index=labels, columns=list(factor_tracks), dtype=np.int8)
    for factor, track in factor_tracks.items():
        factor_ranges = (
            load_bed_as_ranges(track) if isinstance(track, (str, Path)) else track
        )
        overlap = regions.join(factor_ranges)
        overlap_labels = canonicalize_regions(
            overlap.df.apply(
                lambda row: "{}:{}-{}".format(
                    row["Chromosome"], row["Start"], row["End"]
                ),
                axis=1,
            )
        ).unique()
        matrix.loc[matrix.index.intersection(overlap_labels), factor] = 1
    return matrix


def filter_sparse_factors(region_by_factor, min_fraction=0.01, min_count=None):
    """Filter factor columns by their occupied-region count or fraction."""
    if region_by_factor.shape[0] == 0:
        raise ValueError("The region-by-factor matrix has no rows.")
    counts = region_by_factor.sum(axis=0).astype(int)
    statistics = pd.DataFrame({"count": counts})
    statistics["fraction"] = statistics["count"] / float(region_by_factor.shape[0])
    if min_count is None:
        statistics["keep"] = statistics["fraction"] >= float(min_fraction)
    else:
        statistics["keep"] = statistics["count"] >= int(min_count)
    kept = statistics.index[statistics["keep"]].tolist()
    dropped = statistics.index[~statistics["keep"]].tolist()
    return region_by_factor.loc[:, kept].copy(), kept, dropped, statistics


def align_factor_columns(region_by_factor, factors, fill_missing_with_zeros=True):
    """Align a region-by-factor matrix to an explicit ordered factor axis."""
    factors = list(map(str, factors))
    missing = [factor for factor in factors if factor not in region_by_factor.columns]
    if missing and not fill_missing_with_zeros:
        raise ValueError("Matrix is missing requested factors: {}".format(missing))
    aligned = region_by_factor.copy()
    for factor in missing:
        aligned[factor] = 0
    return aligned.reindex(columns=factors).astype(region_by_factor.dtypes.iloc[0])
