"""Genomic-region parsing, canonicalization, and label alignment."""

import re
from decimal import Decimal, InvalidOperation

import pandas as pd


_NUMBER_PATTERN = re.compile(
    r"^[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
)
_REGION_PATTERN = re.compile(
    r"^(chr[\w]+)[\:\-_]\s*([0-9eE\+\.\,]+)"
    r"[\:\-_]\s*([0-9eE\+\.\,]+)$"
)


def _integer_coordinate(value):
    """Convert an integer-like coordinate, including scientific notation."""
    text = str(value).replace(",", "").strip()
    if not _NUMBER_PATTERN.match(text):
        return None
    try:
        return int(text) if text.isdigit() else int(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def parse_region(region):
    """Parse a genomic-region label and return ``(chromosome, start, end)``.

    Colon, hyphen, and underscore delimiters are accepted. Coordinates may use
    decimal or scientific notation. 
    """
    raw = str(region)
    text = raw.strip()
    match = _REGION_PATTERN.match(text)
    if match:
        chromosome, start, end = match.groups()
        start = _integer_coordinate(start)
        end = _integer_coordinate(end)
        if start is not None and end is not None:
            if start > end:
                start, end = end, start
            return chromosome, start, end

    parts = re.split(r"[\:\-_]", text)
    if len(parts) >= 3 and parts[0].startswith("chr"):
        start = _integer_coordinate(parts[1])
        end = _integer_coordinate(parts[2])
        if start is not None and end is not None:
            if start > end:
                start, end = end, start
            return parts[0], start, end
    raise ValueError("Unrecognized genomic-region label: {!r}".format(raw))


def canonicalize_region(region, strict=False):
    """Return a canonical ``chr:start-end`` label.

    Unparseable labels are retained unchanged by default.
    Set ``strict=True`` to raise instead.
    """
    try:
        chromosome, start, end = parse_region(region)
        return "{}:{}-{}".format(chromosome, start, end)
    except ValueError:
        if strict:
            raise
        return str(region)


def canonicalize_regions(labels, strict=False):
    """Canonicalize an iterable of region labels and return a pandas Index."""
    return pd.Index(
        [canonicalize_region(label, strict=strict) for label in labels],
        dtype=object,
    )


def align_region_frames(reference_regions, *frames, require_all=True):
    """Align region-indexed DataFrames to a named reference universe.

    Parameters
    ----------
    reference_regions
        Ordered region labels defining the output row axis.
    frames
        DataFrames whose rows are genomic regions.
    require_all
        If true, raise when a reference region is absent from any frame.

    Returns
    -------
    tuple
        Copies of the input frames with canonicalized, reference-ordered rows.
    """
    reference = canonicalize_regions(reference_regions)
    if reference.has_duplicates:
        raise ValueError("Reference region labels are not unique after canonicalization.")

    aligned = []
    for position, frame in enumerate(frames):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Region object {} is not a pandas DataFrame.".format(position))
        current = frame.copy()
        current.index = canonicalize_regions(current.index)
        if current.index.has_duplicates:
            raise ValueError(
                "Region labels in frame {} are not unique after canonicalization.".format(
                    position
                )
            )
        missing = reference.difference(current.index)
        if require_all and len(missing):
            raise ValueError(
                "Frame {} is missing {} requested regions; examples: {}".format(
                    position, len(missing), missing[:3].tolist()
                )
            )
        use = reference if require_all else reference.intersection(current.index, sort=False)
        aligned.append(current.loc[use].copy())
    return tuple(aligned)

