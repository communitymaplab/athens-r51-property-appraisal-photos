from pathlib import Path

from PIL import Image


def content_after_first_newline(s: str) -> str:
    return s[s.find('\n')+1:] if '\n' in s else s

def classify_page(
    content_string: str,
    aspect_ratio: float,
    file_size_kb: float,
    *,
    only_classify_photo_sheets: bool = False,
) -> str:
    """
    Stage 2 page classification based on OCR text + image heuristics.
    """
    content_upper = (content_string or "").upper()

    if content_after_first_newline(content_upper).startswith("PARCEL NO"):
        # Photo sheets and sketches in the Diaz appraisals have the text "Parcel No. [xx-x]" at the very top. 
        # By using this basic logic plus --ocr-crop-top-percent 10 we can skip unnecessary OCR calls and quickly classify the page.
        return "photo_sheet"
    if only_classify_photo_sheets:
        return "other"

    if abs(aspect_ratio - 1.0) < 0.05 or file_size_kb < 650:
        return "loose_photo"

    elif (
        "APPRAISAL REPORT" in content_upper
        or "DESCRIPTION OF IMPROVEMENT" in content_upper
    ) and "USE SEPARATE SHEET FOR EACH" not in content_upper:
        return "appraisal_form"

    elif "USE SEPARATE SHEET FOR EACH" in content_upper:
        return "description_of_improvement"

    elif "BLOCK" in content_upper and len(content_string) < 200:
        return "photo_sheet"

    else:
        return "other"


def classify_image_path(
    image_path: Path,
    content_string: str,
    *,
    only_classify_photo_sheets: bool = False,
) -> str:
    with Image.open(image_path) as img:
        width, height = img.size
    aspect_ratio = height / width
    file_size_kb = image_path.stat().st_size / 1024
    return classify_page(
        content_string,
        aspect_ratio,
        file_size_kb,
        only_classify_photo_sheets=only_classify_photo_sheets,
    )
