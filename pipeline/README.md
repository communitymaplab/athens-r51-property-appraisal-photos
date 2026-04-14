# R51 Appraisal Scans Pipeline

Python pipeline for **R-51** research: it turns digitized **Athens, Georgia city records** appraisal scans (from the [Digital Library of Georgia](https://dlg.usg.edu/)) into structured rows and extracted photo crops. Scans are organized **by dataset** and **by block folder** (`B2/`, `B13/`, …) before processing.

## What the pipeline does

End-to-end flow:

1. **Stage 1 — OCR**  
   Loads per-page JSON from `ocr-api-output/` next to each image. If missing, calls **Azure Document Intelligence** (`prebuilt-read`) and writes cache files. Optional **`--ocr-crop-top-percent`** / **`--ocr-crop-bottom-percent`** send only a vertical strip to Azure (useful when parcel lines sit at the top).

2. **Stage 2 — Classification**  
   Labels each page (`appraisal_form`, `photo_sheet`, `loose_photo`, `other`, etc.). With **`--only-classify-photo-sheets`**, only the “Parcel No…” top-line rule is used: match → `photo_sheet`, else → `other` (for datasets like Diaz where that is enough).

3. **Stage 3 — Block / parcel extraction**  
   For eligible pages, builds prompts and runs **Claude** via the **Message Batches** API (with disk cache under `data/<dataset-name>/stage3_claude/`). With **`--use-regex`**, Stage 3 first parses `ocr_content` using **`BLOCK_PARCEL_CONTENT_REGEX`** in `src/stage_3_extract.py` (required when the flag is set); **`appraisal_form` is skipped** in Stage 3 and only **`photo_sheet`** pages go through regex/Claude. Unmatched pages still use Claude when an API key is available.

4. **Fill**  
   Within each block, propagates `extracted_id` “upward” on eligible types (R `tidyr::fill(..., .direction = "up")` behavior).

5. **Stage 4 — Photo detection**  
   **Roboflow** workflow on `appraisal_form` / `photo_sheet` (cached per dataset). Loose photos skip detection.

6. **Stage 5 — Crops**  
   Saves crops under `data/<dataset-name>/processed/`, with naming that encodes block/parcel (including multi-parcel strings). Loose photos **copy** the original file to avoid re-encoding.

Output: **`--out-csv`** (default `stage1_3_output.csv`) with paths, classification, `extracted_id`, detection JSON, and crop paths.

## Before you run: get images from DLG

Pipeline input is **image files on disk**, not DLG URLs. Use the standalone helper (not part of `main.py`):

```bash
python scripts/download_dlg_manifest.py --url '<dlg-document-page-url>' --out-path /path/to/dataset/raw/B13
```

- **`--url`**: DLG document page URL (query string is stripped; `/presentation/manifest.json` is fetched).
- **`--out-path`**: folder for numbered `1.jpg`, `2.jpg`, … (create block folders such as `B4`, `B5` to match your layout).

The script may also ship with **preset block ranges** that read **`PHOTOS_DOWNLOAD_DIRECTORY`** from the environment; set that if you use those presets.

Arrange downloads under **`--dataset-dir`** (or under `data/<dataset-name>/raw/` by default) with one folder per block, e.g. `.../B13/185.jpg`.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Optional: put secrets in a **`.env`** file in the project root (`python-dotenv` loads it in `main.py` and in the download script).

## Environment variables

| Variable | Used for |
|----------|-----------|
| **`ANTHROPIC_API_KEY`** | Stage 3 Claude Message Batches when regex/cache do not cover all pages that need extraction. If you use **`--use-regex`** and every eligible page is satisfied by regex or cache, you may not need a call (but set the key whenever Claude might run). |
| **`AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`** | Stage 1: Azure Document Intelligence endpoint (cache miss). |
| **`AZURE_DOCUMENT_INTELLIGENCE_KEY`** | Stage 1: Azure key (cache miss). |
| **`ROBOFLOW_API_KEY`** | Stage 4: live Roboflow calls when JSON cache is missing. |
| **`ROBOFLOW_WORKSPACE`** | Optional; default in code if unset. |
| **`ROBOFLOW_WORKFLOW_ID`** | Optional; default in code if unset. |
| **`ROBOFLOW_API_URL`** | Optional; default in code if unset. |
| **`PHOTOS_DOWNLOAD_DIRECTORY`** | Used by `scripts/download_dlg_manifest.py` when running its built-in per-block download presets. |

## Why different `python3 main.py` commands for different sets

- **Layout and OCR**: Diaz-style sheets often need **top-cropped OCR** (`--ocr-crop-top-percent`) so “Parcel No…” is at the top of the text Azure sees; Bradberry may not need that.
- **Classification**: Diaz can use **`--only-classify-photo-sheets`** when the Parcel-No header is enough; Bradberry uses the full classifier (`appraisal_form`, loose photos, etc.).
- **Stage 3**: Diaz can use **`--use-regex`** plus a compiled **`BLOCK_PARCEL_CONTENT_REGEX`** in `src/stage_3_extract.py` to avoid Claude on most photo sheets; Bradberry typically relies on Claude for forms/sheets.
- **Crops**: **`--bbox-margin`** adjusts padding around Roboflow boxes (tighter margin for one set vs another).
- **Namespacing**: **`--dataset-name`** chooses `data/<dataset-name>/` for Stage 3 cache, Roboflow cache, and processed crops, so two runs never overwrite each other even if block folders look the same.

## Example commands

**1. Bradberry appraisals** (full classifier + Claude path; paths are examples—use yours):

```bash
python3 main.py \
  --dataset-dir "/Users/liam/Documents/R-51 appraisal reports_files/Bradberry appraisals 1962" \
  --dataset-name "Bradberry appraisals 1962"
```

**2. Diaz appraisals** (top OCR crop, regex-first Stage 3 on photo sheets only, strict photo-sheet classification, tighter crop margin):

```bash
python3 main.py \
  --dataset-dir "/Users/liam/Documents/R-51 appraisal reports_files/Diaz appraisals 1964" \
  --dataset-name "Diaz appraisals 1964" \
  --ocr-crop-top-percent 10 \
  --use-regex \
  --only-classify-photo-sheets \
  --bbox-margin 5
```

Other useful flags:

- **`--force-no-ocr`**: do not call Azure when OCR JSON is missing (empty OCR for those pages).
- **`--cache-dir`**: override Stage 3 root (default `data/<dataset-name>/stage3_claude/`).
- **`--out-csv`**: CSV output path.

## Project structure (high level)

- `data/<dataset-name>/raw/` — input scans (your `--dataset-dir` or default here)
- `data/<dataset-name>/processed/` — Stage 5 crops
- `data/<dataset-name>/stage3_claude/` — Stage 3 prompts, results, batch metadata
- `data/<dataset-name>/roboflow_workflow/` — Stage 4 JSON cache (+ `visualizations/` when applicable)
- `<dataset-dir>/<Bn>/ocr-api-output/` — per-page OCR JSON (Stage 1)
- `scripts/download_dlg_manifest.py` — DLG / IIIF full-page JPEG download helper
- `src/stage_*.py` — stage logic; `main.py` orchestrates
- `src/fill_extracted_id_up.py` — per-block `extracted_id` fill

## Pipeline output columns

The CSV includes `image_path`, `file_name`, `block_number`, `classification`, `extracted_id`, `photo_detections` (JSON), and `photo_crops` (JSON paths under `processed/`).

## Manual errata

- Delete `/pipeline/data/Bradberry appraisals 1962/processed/B5/B5_P15__photo1.jpg` as it's not a real photo - placeholder image due to image 57 being broken on DLG here `https://dlg.usg.edu/record/guan_1633_055-006?canvas=56&x=400&y=400&w=1946`
- Manually rename `/pipeline/data/Bradberry appraisals 1962/processed/B8/B8__67_jpg__photo1.jpg` to `B8_P12__photo4.jpg` 
- Run `scripts/merge_photo_csvs.py` to merge the photo,block,parcel csvs from each set
