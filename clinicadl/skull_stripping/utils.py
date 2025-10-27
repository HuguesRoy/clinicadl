import re
from pathlib import Path

# Simple BIDS-ish parser for a basename like:
# sub-01_ses-1_acq-highres_desc-Crop_run-1_T1w.pt
_BIDS_KV = re.compile(r"(?P<key>[a-zA-Z0-9]+)-(?P<val>[^_]+)")


# suffix is the last underscore-separated token before extension
def parse_bids_basename(name: str):
    stem, ext = name.split(".", 1)
    parts = stem.split("_")
    entities = {}
    desc_parts = []
    suffix = None
    for i, p in enumerate(parts):
        m = _BIDS_KV.fullmatch(p)
        if m:
            k, v = m["key"], m["val"]
            if k == "desc":
                desc_parts.append(v)
            else:
                entities[k] = v
        else:
            # assume the last non key-val part is the suffix
            # (rare to have multiple free parts; keep the rightmost)
            suffix = p if i == len(parts) - 1 else suffix
    if suffix is None:
        # Fallback: if everything looked like key-val, take the last entity-like as suffix?
        # Here we keep None and let caller decide.
        pass
    return entities, desc_parts, suffix, "." + ext


def build_bids_basename(entities: dict, desc_parts, suffix: str, ext: str):
    parts = []
    for k in (
        entities.keys()
    ):  # stable order; BIDS doesn’t require strict order within entities
        parts.append(f"{k}-{entities[k]}")
    for d in desc_parts:
        parts.append(f"desc-{d}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + ext


def make_derivative_names(image_path, use_uncropped_image: bool):
    image_path = Path(image_path)
    entities, desc_parts, suffix, ext = parse_bids_basename(image_path.name)

    # Ensure we have a suffix (modality). If missing, do nothing fancy.
    modality = suffix  # e.g., T1w, T2w, FLAIR, dwi, etc.

    # NAME: skull-stripped image
    if use_uncropped_image:
        # add/replace SkullStripped in desc (without requiring desc-Crop)
        # also emulate your "res → desc-SkullStripped_res" behaviour using desc
        desc_no_skull = [d for d in desc_parts if d.lower() not in ("skullstripped",)]
        desc_img = desc_no_skull + ["SkullStripped"]
    else:
        # keep Crop if present, then add SkullStripped
        desc_no_dups = []
        seen = set()
        for d in desc_parts:
            if d.lower() not in seen:
                desc_no_dups.append(d)
                seen.add(d.lower())
        # Ensure Crop remains if present; then append SkullStripped (once).
        if "crop" in (d.lower() for d in desc_no_dups):
            desc_img = desc_no_dups + (
                ["SkullStripped"]
                if "skullstripped" not in (d.lower() for d in desc_no_dups)
                else []
            )
        else:
            # If not cropped, we still add SkullStripped (matches your intent for the else-branch)
            desc_img = desc_no_dups + (
                ["SkullStripped"]
                if "skullstripped" not in (d.lower() for d in desc_no_dups)
                else []
            )

    name_img = build_bids_basename(entities, desc_img, modality, ext)

    # NAME_MASK: brain/dseg mask
    if use_uncropped_image:
        # add/replace brain in desc, and swap suffix → dseg
        desc_mask = [d for d in desc_parts if d.lower() not in ("brain",)]
        desc_mask += ["brain"]
    else:
        # preserve Crop if present + add brain
        desc_mask = []
        seen = set()
        for d in desc_parts:
            if d.lower() not in seen:
                desc_mask.append(d)
                seen.add(d.lower())
        if "brain" not in (d.lower() for d in desc_mask):
            desc_mask += ["brain"]

    name_mask = build_bids_basename(entities, desc_mask, "dseg", ext)

    # Return full paths (matching your original use of .parent)
    return str(name_img), str(name_mask)
