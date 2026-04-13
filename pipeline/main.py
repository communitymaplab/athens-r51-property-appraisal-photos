from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path
import re
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import default_config_from_path
from models import PageArtifact
from stage_1_ocr import get_cached_ocr, get_content_string
from stage_2_classify import classify_image_path
from stage_3_claude_batch import (
    chunk_requests,
    compute_input_hash,
    extract_text_from_result_item,
    load_page_cache,
    parse_results_payload,
    sanitize_cache_key,
    save_page_cache,
    submit_batch,
    wait_for_batch_completion,
    get_batch_results_response,
)
from stage_3_extract import (
    BLOCK_PARCEL_CONTENT_REGEX,
    build_llm_prompt,
    extract_region_text,
    parse_id_from_llm_response,
    try_parse_block_parcel_from_content_regex,
)
from stage_4_detect import detect_photo_regions
from fill_extracted_id_up import fill_extracted_id_up_per_block
from stage_5_crop import crop_and_save_regions
from utils import iter_image_paths


def ensure_dataset_directories(dataset_name: str) -> Path:
    """Create data/<dataset-name>/{raw,processed,roboflow_workflow} for scans, crops, Stage 4 cache."""
    root = PROJECT_ROOT / "data" / dataset_name
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    (root / "roboflow_workflow").mkdir(parents=True, exist_ok=True)
    (root / "roboflow_workflow" / "visualizations").mkdir(parents=True, exist_ok=True)
    return root


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run appraisal pipeline stages 1-3.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Root directory containing scanned page images. Default: data/<dataset-name>/raw in this repo.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Bradberry appraisals 1962",
        help="Dataset folder under data/: raw/, processed/, stage3_claude/ are created and used by default paths.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override Stage 3 cache root (prompts/results/batches). Default: data/<dataset-name>/stage3_claude under the project.",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-sonnet-4-6",
        help="Anthropic model name for Stage 3 parcel/block extraction.",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=512,
        help="Max tokens for each Claude Stage 3 response.",
    )
    parser.add_argument(
        "--claude-batch-size",
        type=int,
        default=500,
        help="Number of Stage 3 requests per Claude batch submission.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=100.0,
        help="Stage 5: expand each crop by this many pixels on each side before clamping to the page.",
    )
    parser.add_argument(
        "--ocr-crop-top-percent",
        type=float,
        default=0.0,
        help="Stage 1 OCR only: keep top N%% of image before OCR (0 disables).",
    )
    parser.add_argument(
        "--ocr-crop-bottom-percent",
        type=float,
        default=0.0,
        help="Stage 1 OCR only: keep bottom N%% of image before OCR (0 disables).",
    )
    parser.add_argument(
        "--force-no-ocr",
        action="store_true",
        help="Stage 1: if OCR cache is missing, skip Azure (return no OCR) instead of calling the API.",
    )
    parser.add_argument(
        "--use-regex",
        action="store_true",
        help=(
            "Stage 3: parse block/parcel from OCR content via regex first "
            "(BLOCK_PARCEL_CONTENT_REGEX in stage_3_extract.py is required). "
            "Also skips `appraisal_form` pages in Stage 3 (only `photo_sheet`). "
            "Unmatched pages use Claude."
        ),
    )
    parser.add_argument(
        "--only-classify-photo-sheets",
        action="store_true",
        help=(
            "Stage 2: only apply the top-line `PARCEL NO` rule for photo sheets; "
            "if not matched, classify as `other` and skip all other Stage 2 checks."
        ),
    )
    parser.add_argument(
        "--out-processed-photos-csv",
        default=None,
        help=(
            "Stage 5: write one CSV row per saved crop (path under processed/, block, parcel). "
            "Default: data/<dataset-name>/processed_photos.csv in this repo."
        ),
    )
    return parser


def roboflow_cache_key_for_image(image_path: Path) -> str:
    """
    Roboflow JSON filename stem: ``B6__83_jpg`` style (``sanitize`` of ``B6/83.jpg``).

    Uses only the block folder and filename, not ``relative_to(dataset_dir)``, so the
    key stays the same when ``--dataset-dir`` points at another tree. Caches for
    different ``--dataset-name`` values stay separate because ``roboflow_workflow`` lives
    under ``data/<dataset-name>/``.
    """
    logical = f"{image_path.parent.name}/{image_path.name}"
    return sanitize_cache_key(logical)


def get_page_sort_key(image_path: Path) -> tuple[int, str]:
    stem = image_path.stem
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group()), stem
    return float("inf"), stem


def block_sort_key(block: str) -> tuple:
    """Numeric block order (2 before 10); non-numeric blocks sort after, lexicographically."""
    s = (block or "").strip()
    if s.isdigit():
        return (0, int(s))
    return (1, s.casefold())


def get_neighbor_region_text(
    artifacts: list[PageArtifact],
    index: int,
    direction: int,
) -> list[str | None]:
    neighbor_index = index + direction
    region_texts: list[str | None] = []
    while 0 <= neighbor_index < len(artifacts):
        candidate = artifacts[neighbor_index]
        if candidate.classification in {"appraisal_form", "photo_sheet"}:
            region_texts.append(candidate.region_text)
            if len(region_texts) == 2:
                break
        neighbor_index += direction
    return region_texts


def _property_owners_countable_mask(df: pd.DataFrame) -> pd.Series:
    """
    Rows that qualify for parcel listing: Parcel is present (excludes summary rows);
    Area to be acquired is present and non-zero; Remarks is not the literal
    'Deleted' (case-insensitive, stripped).
    """
    parcel_col = "Parcel"
    area_col = "Area to be acquired"
    remarks_col = "Remarks"
    required = (parcel_col, area_col, remarks_col)
    if not all(c in df.columns for c in required):
        return pd.Series(False, index=df.index)

    p_str = df[parcel_col].astype(str).str.strip()
    parcel_ok = df[parcel_col].notna() & (p_str != "") & (p_str.str.lower() != "nan")

    remarks_ok = (
        df[remarks_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        != "deleted"
    )

    area = df[area_col]
    is_na = area.isna()
    str_a = area.astype(str).str.strip()
    empty = is_na | (str_a == "") | (str_a.str.lower() == "nan")
    num = pd.to_numeric(str_a.str.replace(",", "", regex=False), errors="coerce")
    is_zero = (~empty) & num.notna() & (num == 0)
    area_ok = ~empty & ~is_zero

    return parcel_ok & remarks_ok & area_ok


def fetch_parcel_data(url):
    try:
        # 1. Fetch the data from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        data = response.json()
        
        # 2. Extract the 'attributes' from each feature
        # This list comprehension flattens the nested structure
        attributes_list = [feature['attributes'] for feature in data.get('features', [])]
        
        # 3. Create the DataFrame
        df = pd.DataFrame(attributes_list)
        
        # Optional: Ensure only the relevant columns are kept
        df = df[['BLOCK', 'PARCEL']]
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


# Constants for data sources
CENSUS_URL = "https://docs.google.com/spreadsheets/d/1rnc2khaC_HiXRDyp0aE5GpsuQiSvqclWPzGT8iIcUWs/export?format=csv&id=1rnc2khaC_HiXRDyp0aE5GpsuQiSvqclWPzGT8iIcUWs&gid=1980504423"
ARC_GIS_BASE = "https://services2.arcgis.com/PYn6bWCjT6bhw1z3/ArcGIS/rest/services/R51_parcel_boundaries/FeatureServer/{layer}/query?where=1%3D1&outFields=BLOCK%2C+PARCEL&returnGeometry=false&f=pjson"
PROPERTY_OWNERS_URL = "https://docs.google.com/spreadsheets/d/15djqe28_MRwxvHb4mrNM4qaR4Yso_2tYlrT8ompFLxg/export?format=csv"
CENSUS_TABLE = pd.read_csv(CENSUS_URL)
PROPERTY_OWNERS_TABLE = pd.read_csv(PROPERTY_OWNERS_URL)
_1963parcels = fetch_parcel_data(ARC_GIS_BASE.format(layer="2"))
_1968parcels = fetch_parcel_data(ARC_GIS_BASE.format(layer="3"))


def list_parcel_numbers_for_block(block_number: str) -> list[str]:
    """
    Retrieves unique parcel numbers for a specific block from census and GIS layers.
    """
    # 1. Normalize input
    formatted_block = f"B{block_number}"
    all_parcels = set()

    # 2. Process Census Data
    try:
        census_matches = CENSUS_TABLE[CENSUS_TABLE["block"] == formatted_block]["parcel"]
        all_parcels.update(census_matches.dropna().astype(str))
    except Exception as e:
        print(f"Error loading census data: {e}")

    # 3. Process GIS Layers (Iterative approach to avoid repetition)
    for layer_id in ["2", "3"]:
        try:
            df_layer = _1963parcels if layer_id == "2" else _1968parcels
            
            layer_matches = df_layer[df_layer["BLOCK"] == formatted_block]["PARCEL"]
            all_parcels.update(layer_matches.dropna().astype(str))
        except Exception as e:
            print(f"Error loading GIS layer {layer_id}: {e}")

    # 4. Process Property Owners Table (only rows with acquisition area and not deleted)
    try:
        pot = PROPERTY_OWNERS_TABLE
        eligible = pot[_property_owners_countable_mask(pot)]
        try:
            block_match = eligible[eligible["Block"].astype(int) == int(block_number)]
        except Exception:
            if block_number != "2A":
                raise
            # Block 2A = Block 31
            block_match = eligible[eligible["Block"].astype(int) == 31]
        property_owners_matches = block_match["Parcel"]
        all_parcels.update(property_owners_matches.dropna().astype(str))
    except Exception as e:
        print(f"Error loading property owners table: {e}")

    # 5. Clean and return results
    # Using a set during collection automatically handles uniqueness
    cleaned_parcels = list({p.replace("P", "").strip() for p in all_parcels})
    contains_noninteger = False
    for p in cleaned_parcels:
        if not p.isdigit():
            contains_noninteger = True
    
    return sorted(cleaned_parcels) if contains_noninteger else sorted(cleaned_parcels, key=int)


def compress_number_list(values: list[str]) -> str:
    nums = sorted(int(x) for x in values)

    full_range = set(range(nums[0], nums[-1] + 1))
    actual = set(nums)
    excluded = full_range - actual

    if excluded:
        excl_str = ", ".join(str(n) for n in sorted(excluded))
        return f"{nums[0]}-{nums[-1]} (excluding {excl_str})"
    else:
        return f"{nums[0]}-{nums[-1]}"


def get_parcel_numbers_for_blocks(blocks: list[str]) -> pd.DataFrame:
    data = []
    
    for block in blocks:
        parcel_numbers = list_parcel_numbers_for_block(block)
        if len(parcel_numbers) == 0:
            continue
        parcel_numbers_not_number = [p for p in parcel_numbers if not p.isdigit()]
        parcel_numbers_numeric = sorted([p for p in parcel_numbers if p.isdigit()])
        data.append({"block": block,
                     "parcel_values_description": compress_number_list(parcel_numbers_numeric),
                     "non_numeric_parcel_values": parcel_numbers_not_number})
    
    return pd.DataFrame(data)


def build_block_constraint_lookup(block_constraints: pd.DataFrame) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for row in block_constraints.to_dict(orient="records"):
        block = str(row["block"])
        lookup[block] = row
    return lookup


def run_full_pipeline(
    dataset_dir: str,
    cache_dir: str,
    roboflow_cache_dir: str,
    processed_dir: str,
    dataset_name: str,
    anthropic_model: str,
    anthropic_max_tokens: int,
    claude_batch_size: int,
    bbox_margin: float,
    ocr_crop_top_percent: float,
    ocr_crop_bottom_percent: float,
    force_no_ocr: bool,
    use_regex: bool,
    only_classify_photo_sheets: bool,
    processed_photos_csv: str | Path | None = None,
) -> pd.DataFrame:
    config = default_config_from_path(dataset_dir)
    artifacts: list[PageArtifact] = []
    block_numbers = list(set([image_path.parent.name.replace("B", "") for image_path in iter_image_paths(config)]))
    block_parcel_numbers = get_parcel_numbers_for_blocks(block_numbers)
    block_constraint_lookup = build_block_constraint_lookup(block_parcel_numbers)

    for image_path in iter_image_paths(config):
        ocr_json = get_cached_ocr(
            image_path,
            config,
            ocr_crop_top_percent=ocr_crop_top_percent,
            ocr_crop_bottom_percent=ocr_crop_bottom_percent,
            force_no_ocr=force_no_ocr,
        )
        content_string = get_content_string(ocr_json)
        # print("\n")
        
        # if "No." in content_string and not content_string.startswith("Parcel"):
        #    print("--------------------------------")
        #    print("WARNING: WARNING: WARNING:")
        # print(f"{content_string} - {image_path}")
        # print("\n\n")

        # Temporarily abort pipeline after stage 1
        classification = classify_image_path(
            image_path,
            content_string,
            only_classify_photo_sheets=only_classify_photo_sheets,
        )

        artifact = PageArtifact(
            image_path=image_path,
            ocr_json=ocr_json,
            ocr_content=content_string,
            classification=classification,
            block_number=image_path.parent.name.replace("B", "")  # Remove the "B" prefix from the block number
        )

        if classification == "appraisal_form":
            artifact.region_text = extract_region_text(ocr_json)
        elif classification == "photo_sheet":
            # If it's a photo sheet, the block and parcel number could be anywhere,
            #  as it's not a set form like the appraisal form
            # Hacky solution is to take substring from the
            # first occurence of the text "BLOCK" until the end of the string
            index = content_string.lower().find("block")
            if index != -1:
                result = content_string[index:]
                artifact.region_text = result
            else:
                artifact.region_text = content_string

        artifacts.append(artifact)

    # raise Exception("Temporarily abort pipeline after stage 1 and stage 2")

    artifacts_by_block: dict[str, list[PageArtifact]] = {}
    for artifact in artifacts:
        block_number = artifact.block_number or ""
        artifacts_by_block.setdefault(block_number, []).append(artifact)

    for block_artifacts in artifacts_by_block.values():
        block_artifacts.sort(key=lambda artifact: get_page_sort_key(artifact.image_path))

    # iter_image_paths uses lexicographic path order (10.jpg before 2.jpg). Canonical order
    # is natural sort per block, matching fill / neighbor logic in artifacts_by_block.
    artifacts = [
        art
        for bn in sorted(artifacts_by_block.keys(), key=block_sort_key)
        for art in artifacts_by_block[bn]
    ]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not use_regex:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Stage 3 (Claude Message Batches)")
    if use_regex and BLOCK_PARCEL_CONTENT_REGEX is None:
        raise RuntimeError(
            "Stage 3: --use-regex requires BLOCK_PARCEL_CONTENT_REGEX to be set in "
            "src/stage_3_extract.py (named groups block and parcel)."
        )

    cache_root = Path(cache_dir)
    prompts_dir = cache_root / "prompts"
    results_dir = cache_root / "results"
    batches_dir = cache_root / "batches"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    request_items: list[dict[str, object]] = []
    request_lookup: dict[str, dict[str, object]] = {}
    stage3_cache_hits = 0
    stage3_need_api = 0
    stage3_regex_hits = 0

    print(f"Stage 3: cache directory {cache_root.resolve()}")

    for artifact in artifacts:
        if artifact.classification not in {"appraisal_form", "photo_sheet"}:
            continue
        if use_regex and artifact.classification == "appraisal_form":
            continue

        if use_regex:
            regex_parsed = try_parse_block_parcel_from_content_regex(
                artifact.ocr_content,
                artifact.block_number,
                str(artifact.image_path.relative_to(config.dataset_dir)),
                pattern=BLOCK_PARCEL_CONTENT_REGEX,
            )
            if regex_parsed is not None:
                artifact.extracted_id = regex_parsed
                stage3_regex_hits += 1
                continue

        block_artifacts = artifacts_by_block.get(artifact.block_number or "", [])
        idx = block_artifacts.index(artifact)
        previous_region_texts = get_neighbor_region_text(block_artifacts, idx, -1)
        next_region_texts = get_neighbor_region_text(block_artifacts, idx, 1)
        constraint = block_constraint_lookup.get(artifact.block_number or "")
        prompt = build_llm_prompt(
            artifact.block_number or "",
            artifact.region_text,
            previous_region_texts=previous_region_texts,
            next_region_texts=next_region_texts,
            possible_parcel_numbers=constraint,
        )

        relative_path = str(artifact.image_path.relative_to(config.dataset_dir))
        cache_key = sanitize_cache_key(relative_path)
        custom_id = f"stage3__{cache_key}"
        input_hash = compute_input_hash(anthropic_model, prompt)
        cache_file = results_dir / f"{cache_key}.json"

        cached = load_page_cache(cache_file)
        if cached and cached.get("input_hash") == input_hash and cached.get("parsed_output"):
            artifact.extracted_id = cached["parsed_output"]
            stage3_cache_hits += 1
            continue


        print(f"Sending to Claude: {relative_path}")
        stage3_need_api += 1
        prompt_manifest = {
            "cache_key": cache_key,
            "custom_id": custom_id,
            "relative_path": relative_path,
            "image_path": str(artifact.image_path),
            "block_number": artifact.block_number,
            "input_hash": input_hash,
            "model": anthropic_model,
            "prompt": prompt,
        }
        save_page_cache(prompts_dir / f"{cache_key}.json", prompt_manifest)

        request_payload = {
            "custom_id": custom_id,
            "params": {
                "model": anthropic_model,
                "max_tokens": anthropic_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        }
        request_items.append(request_payload)
        request_lookup[custom_id] = {
            "artifact": artifact,
            "cache_key": cache_key,
            "cache_file": cache_file,
            "input_hash": input_hash,
            "prompt": prompt,
            "relative_path": relative_path,
        }

    total_stage3 = stage3_cache_hits + stage3_need_api + stage3_regex_hits
    print(
        f"Stage 3: {stage3_cache_hits} cache hit(s), {stage3_regex_hits} regex parse(s), "
        f"{stage3_need_api} to send to Claude (of {total_stage3} eligible page(s))"
    )
    if request_items:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when Stage 3 must call Claude (regex/cache "
                "did not cover all eligible pages)."
            )
        if claude_batch_size <= 0:
            n_chunks = 1
        else:
            n_chunks = (len(request_items) + claude_batch_size - 1) // claude_batch_size
        print(
            f"Stage 3: calling Claude API for {len(request_items)} page(s) "
            f"in {n_chunks} batch chunk(s)"
        )
    elif total_stage3 > 0:
        print("Stage 3: no Claude API calls (all eligible pages loaded from cache).")
    else:
        print("Stage 3: no appraisal_form / photo_sheet pages; skipping extraction.")

    if request_items:
        request_chunks = chunk_requests(request_items, claude_batch_size)
        for chunk_index, request_chunk in enumerate(request_chunks, start=1):
            submitted = submit_batch(api_key=api_key, requests_list=request_chunk)
            batch_id = str(submitted["id"])
            batch_meta = {
                "chunk_index": chunk_index,
                "batch_id": batch_id,
                "request_count": len(request_chunk),
                "custom_ids": [item["custom_id"] for item in request_chunk],
                "submitted": submitted,
            }
            save_page_cache(batches_dir / f"{batch_id}_meta.json", batch_meta)
            save_page_cache(batches_dir / f"{batch_id}_request.json", {"requests": request_chunk})

            completed_status = wait_for_batch_completion(api_key=api_key, batch_id=batch_id)
            result_response = get_batch_results_response(api_key=api_key, batch_status=completed_status)
            result_items = parse_results_payload(result_response)
            save_page_cache(
                batches_dir / f"{batch_id}_status.json",
                {"status": completed_status, "result_count": len(result_items)},
            )

            for item in result_items:
                custom_id = item.get("custom_id")
                if not isinstance(custom_id, str):
                    continue
                lookup = request_lookup.get(custom_id)
                if not lookup:
                    continue

                output_text = extract_text_from_result_item(item)
                parsed_output = None
                parse_error = None
                if output_text:
                    try:
                        parsed_output = parse_id_from_llm_response(output_text)
                    except json.JSONDecodeError as exc:
                        parse_error = str(exc)

                cache_payload = {
                    "cache_key": lookup["cache_key"],
                    "custom_id": custom_id,
                    "relative_path": lookup["relative_path"],
                    "image_path": str(lookup["artifact"].image_path),
                    "input_hash": lookup["input_hash"],
                    "model": anthropic_model,
                    "prompt": lookup["prompt"],
                    "raw_result_item": item,
                    "raw_output_text": output_text,
                    "parsed_output": parsed_output,
                    "parse_error": parse_error,
                }
                save_page_cache(lookup["cache_file"], cache_payload)
                if parsed_output:
                    lookup["artifact"].extracted_id = parsed_output
                else:
                    lookup["artifact"].extracted_id = {
                        "parcel": None,
                        "block": lookup["artifact"].block_number,
                        "confidence": "low",
                        "raw_output_text": output_text,
                        "parse_error": parse_error,
                    }

    n_filled = fill_extracted_id_up_per_block(artifacts_by_block)
    print(
        f"Fill extracted_id (up, per block): {n_filled} page(s) inherited parcel/block "
        f"from a later eligible page (loose_photo / photo_sheet / appraisal_form)."
    )

    roboflow_root = Path(roboflow_cache_dir)
    roboflow_root.mkdir(parents=True, exist_ok=True)
    stage4_stats: dict[str, int] = {}
    print(f"Stage 4: Roboflow cache directory {roboflow_root.resolve()}")

    for artifact in artifacts:
        if artifact.classification == "loose_photo":
            artifact.photo_regions = []
        elif artifact.classification in {"appraisal_form", "photo_sheet"}:
            rf_cache_key = roboflow_cache_key_for_image(artifact.image_path)
            artifact.photo_regions = detect_photo_regions(
                artifact.image_path,
                artifact.classification,
                dataset_name=dataset_name,
                cache_file=roboflow_root / f"{rf_cache_key}.json",
                cache_stats=stage4_stats,
            )
        else:
            artifact.photo_regions = []

    if stage4_stats:
        print(
            "Stage 4: "
            f"cache hits {stage4_stats.get('roboflow_cache_hits', 0)}, "
            f"API calls {stage4_stats.get('roboflow_api_calls', 0)}, "
            f"skipped (no ROBOFLOW_API_KEY) {stage4_stats.get('roboflow_skipped_no_key', 0)}"
        )

    processed_root = Path(processed_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    stage5_crop_count = 0
    print(f"Stage 5: processed (crops) directory {processed_root.resolve()}")

    photos_csv_path = Path(processed_photos_csv) if processed_photos_csv else None
    photos_csv_file = None
    photos_row_writer = None
    if photos_csv_path is not None:
        photos_csv_path.parent.mkdir(parents=True, exist_ok=True)
        photos_csv_file = photos_csv_path.open("w", newline="", encoding="utf-8")
        photos_row_writer = csv.writer(photos_csv_file)
        photos_row_writer.writerow(["path", "block", "parcel"])

    def on_processed_photo_saved(
        rel: str, block: str | None, parcel: str | None
    ) -> None:
        if photos_row_writer is None or photos_csv_file is None:
            return
        photos_row_writer.writerow(
            [
                rel,
                block if block is not None else "",
                parcel if parcel is not None else "",
            ]
        )
        photos_csv_file.flush()

    try:
        for artifact in artifacts:
            regions = artifact.photo_regions or []
            if not regions and artifact.classification != "loose_photo":
                artifact.photo_crop_paths = []
                continue
            relative_path = str(artifact.image_path.relative_to(config.dataset_dir))
            page_cache_key = sanitize_cache_key(relative_path)
            abs_paths = crop_and_save_regions(
                artifact.image_path,
                regions,
                output_dir=processed_root,
                cache_key=page_cache_key,
                extracted_id=artifact.extracted_id
                if isinstance(artifact.extracted_id, dict)
                else None,
                page_block_number=artifact.block_number,
                block_folder_name=artifact.image_path.parent.name,
                bbox_margin=bbox_margin,
                classification=artifact.classification,
                on_processed_photo_saved=on_processed_photo_saved
                if photos_csv_file
                else None,
            )
            artifact.photo_crop_paths = [
                str(p.relative_to(processed_root)) for p in abs_paths
            ]
            stage5_crop_count += len(abs_paths)
    finally:
        if photos_csv_file is not None:
            photos_csv_file.close()

    print(f"Stage 5: wrote {stage5_crop_count} crop image(s)")
    if photos_csv_path is not None:
        print(f"Stage 5: processed photos CSV {photos_csv_path.resolve()}")

    records: list[dict[str, object]] = []
    for artifact in artifacts:
        records.append(
            {
                "image_path": str(artifact.image_path),
                "file_name": str(artifact.image_path.relative_to(config.dataset_dir)),
                "block_number": artifact.block_number,
                "classification": artifact.classification,
                "extracted_id": artifact.extracted_id,
                "photo_detections": json.dumps(artifact.photo_regions or []),
                "photo_crops": json.dumps(artifact.photo_crop_paths or []),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    args = build_parser().parse_args()
    ensure_dataset_directories(args.dataset_name)
    if args.dataset_dir:
        dataset_dir = str(Path(args.dataset_dir).expanduser())
    else:
        dataset_dir = str(PROJECT_ROOT / "data" / args.dataset_name / "raw")
    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).expanduser())
    else:
        cache_dir = str(PROJECT_ROOT / "data" / args.dataset_name / "stage3_claude")
    roboflow_cache_dir = str(
        PROJECT_ROOT / "data" / args.dataset_name / "roboflow_workflow"
    )
    processed_dir = str(PROJECT_ROOT / "data" / args.dataset_name / "processed")
    if args.out_processed_photos_csv:
        processed_photos_csv: str | Path = Path(
            args.out_processed_photos_csv
        ).expanduser()
    else:
        processed_photos_csv = (
            PROJECT_ROOT / "data" / args.dataset_name / "processed_photos.csv"
        )
    df = run_full_pipeline(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        roboflow_cache_dir=roboflow_cache_dir,
        processed_dir=processed_dir,
        dataset_name=args.dataset_name,
        anthropic_model=args.anthropic_model,
        anthropic_max_tokens=args.anthropic_max_tokens,
        claude_batch_size=args.claude_batch_size,
        bbox_margin=args.bbox_margin,
        ocr_crop_top_percent=args.ocr_crop_top_percent,
        ocr_crop_bottom_percent=args.ocr_crop_bottom_percent,
        force_no_ocr=args.force_no_ocr,
        use_regex=args.use_regex,
        only_classify_photo_sheets=args.only_classify_photo_sheets,
        processed_photos_csv=processed_photos_csv,
    )
    out_path = PROJECT_ROOT / "data" / args.dataset_name / "pipeline_output.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()