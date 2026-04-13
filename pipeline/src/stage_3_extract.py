import json
import re
from re import Pattern
from typing import Any

# When using ``--use-regex`` in ``main.py``, set this to a compiled pattern with
# named groups ``block`` and ``parcel``, e.g.:
#   BLOCK_PARCEL_CONTENT_REGEX = re.compile(
#       r"Parcel\s*(?P<parcel>[\w\s,&]+?)\s*Block\s*(?P<block>\d+)", re.I
#   )
BLOCK_PARCEL_CONTENT_REGEX = re.compile(
    r"Parcel Nos?[. ]{0,4}(?P<block>[\d]+[A-Z]?)-(?P<parcel>[\d]+[A-Z]?(?: & [\d]+[A-Z]?)?)", re.IGNORECASE
    )


def try_parse_block_parcel_from_content_regex(
    content_string: str,
    block_number: str,
    image_path: str,
    *,
    pattern: Pattern[str] | None = None,
) -> dict[str, Any] | None:
    """
    Parse ``block`` and ``parcel`` from full OCR ``content_string`` using a regex.

    Uses ``pattern`` if given, else :data:`BLOCK_PARCEL_CONTENT_REGEX`.
    The pattern must define ``(?P<block>...)`` and ``(?P<parcel>...)``.
    Returns a dict aligned with Claude JSON output, plus ``"source": "regex"``.
    """
    #print(f"{content_string}")

    pat = pattern if pattern is not None else BLOCK_PARCEL_CONTENT_REGEX
    if pat is None:
        return None
    m = pat.search(content_string or "")
    if not m:
        return None
    gd = m.groupdict()
    block = gd.get("block")
    if block != block_number:
        print(f"WARNING: Block number mismatch: regex returned {block}, but expected {block_number} for {image_path}")
        return {"block": block_number, "parcel": None, "confidence": "low", "source": "regex"}
    parcel = gd.get("parcel")
    if block is None or parcel is None:
        return None
    block_s = str(block).strip()
    parcel_s = str(parcel).strip()
    if not block_s or not parcel_s:
        return None
    return {
        "block": block_s,
        "parcel": parcel_s,
        "confidence": "high",
        "source": "regex",
    }

def extract_region_text(ocr_data: dict[str, Any] | None) -> str | None:
    """
    Stage 3a: isolate OCR text from bottom-left parcel/block region.
    """
    if not ocr_data:
        return None
    pages = ocr_data.get("pages") or []
    if not pages:
        return None

    page = pages[0]
    page_width = page.get("width")
    page_height = page.get("height")
    words = page.get("words", [])
    if not page_width or not page_height or not words:
        return None

    horizontal_limit = page_width / 2
    vertical_search_start = page_height * 0.80

    anchor_y_top = None
    for word in words:
        content = str(word.get("content", "")).lower()
        polygon = word.get("polygon") or []
        if len(polygon) < 2:
            continue
        y_top = polygon[1]
        if "parcel" in content and y_top > vertical_search_start:
            anchor_y_top = y_top - 100
            break

    if anchor_y_top is None:
        return None

    captured_words: list[dict[str, Any]] = []
    for word in words:
        polygon = word.get("polygon") or []
        if len(polygon) < 2:
            continue
        x_left = polygon[0]
        y_top = polygon[1]
        if 0 <= x_left <= horizontal_limit and y_top >= anchor_y_top:
            captured_words.append(word)

    captured_words.sort(key=lambda w: (w["polygon"][0], w["polygon"][1]))

    rows: list[list[dict[str, Any]]] = []
    y_tolerance = 45
    for word in captured_words:
        y_val = word["polygon"][1]
        placed = False
        for row in rows:
            if abs(row[0]["polygon"][1] - y_val) < y_tolerance:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])

    rows.sort(key=lambda row: row[0]["polygon"][1])
    for row in rows:
        row.sort(key=lambda w: w["polygon"][0])

    flattened = [word for row in rows for word in row]

    target_label = ["Parcel", "Parcel.", "No.", "No ."]
    found_indices: list[int] = []
    for idx, word in enumerate(flattened):
        content = str(word.get("content", ""))
        if any(target.lower() in content.lower() for target in target_label):
            found_indices.append(idx)

    if found_indices:
        label_words = [flattened[idx] for idx in found_indices]
        for idx in sorted(found_indices, reverse=True):
            flattened.pop(idx)
        flattened = label_words + flattened

    return " ".join(str(w.get("content", "")) for w in flattened).strip()


def build_llm_prompt(
    block_number: str,
    region_text: str | None,
    previous_region_texts: list[str | None] | None = None,
    next_region_texts: list[str | None] | None = None,
    possible_parcel_numbers: dict[str, Any] | None = None,
) -> str:
    previous_region_texts = previous_region_texts or []
    next_region_texts = next_region_texts or []
    previous_1 = previous_region_texts[0] if len(previous_region_texts) > 0 else None
    previous_2 = previous_region_texts[1] if len(previous_region_texts) > 1 else None
    next_1 = next_region_texts[0] if len(next_region_texts) > 0 else None
    next_2 = next_region_texts[1] if len(next_region_texts) > 1 else None
    current_region_text = region_text or "OCR_MISSING"
    context_lines: list[str] = []
    if previous_2 is not None or len(previous_region_texts) > 1:
        context_lines.append(
            f"[Contextual evidence]: Previous 2 page text: \"{previous_2 or 'OCR_MISSING'}\""
        )
    if previous_1 is not None or len(previous_region_texts) > 0:
        context_lines.append(
            f"[Contextual evidence]: Previous 1 page text: \"{previous_1 or 'OCR_MISSING'}\""
        )
    if next_1 is not None or len(next_region_texts) > 0:
        context_lines.append(
            f"[Contextual evidence]: Next 1 page text: \"{next_1 or 'OCR_MISSING'}\""
        )
    if next_2 is not None or len(next_region_texts) > 1:
        context_lines.append(
            f"[Contextual evidence]: Next 2 page text: \"{next_2 or 'OCR_MISSING'}\""
        )
    context_block = "\n".join(context_lines)

    possible_parcel_numbers_text = ""

    if possible_parcel_numbers is not None:
        parcel_values_description = str(
            possible_parcel_numbers.get("parcel_values_description", "")
        ).strip()
        non_numeric_parcel_values = possible_parcel_numbers.get(
            "non_numeric_parcel_values", []
        )
        if parcel_values_description:
            possible_parcel_numbers_text = (
                f"For this block, the only possible parcel numbers are: "
                f"{parcel_values_description}."
            )
        if non_numeric_parcel_values:
            non_numeric_text = ", ".join(str(value) for value in non_numeric_parcel_values)
            if possible_parcel_numbers_text:
                possible_parcel_numbers_text += " "
            possible_parcel_numbers_text += (
                f"Additionally, the following non-numeric parcel values are possible: "
                f"{non_numeric_text}."
            )
        else:
            possible_parcel_numbers_text += "No non-numeric or alphanumeric parcel values are possible for this block. DO NOT RETURN a parcel number containing a letter."
            possible_parcel_numbers_text += "If the OCR text indicates a parcel number containing a letter, that is in conflict with this instruction, and you should remove the letter and return only the numeric part."
    return f"""
Extract block and parcel numbers from this OCR text of a property appraisal
page. Rules:
- "Blk", "BLK", "BCK", "BIK", "BIR", "Bek", "Blb", "BIBER", "B&k" etc.
  all mean "Block"
- Parcel identifiers are usually numeric, but in rare valid cases may include a
  trailing letter, such as "2A", "4A", or "6B". These are valid parcel
  identifiers, not sub-parcels.
- OCR may confuse letters and digits. Common confusions include "A" <-> "4",
  "B" <-> "8", "I" <-> "1", and "O" <-> "0".
- If a token like "24" is implausible in sequence, consider whether it may
  actually be "2A" or another parcel identifier with a trailing letter.
- Descriptive qualifiers like "Rear" or unrelated labels 
("Bldg A" or letters like "A", "B", "C") are not parcel
  identifiers unless clearly attached as part of the parcel notation.
- Ignore dates, "SUMMARY SHEET", "House", "Building" labels
- Some entries have multiple parcels e.g. "5 & 4" — list them all

The correct block number is given below. Since you are given the correct block number,
you may use this information to help you extract the correct block and parcel numbers
from unclear OCR text containing ambiguous numbers or weakly structured text.
Always return the correct block number, even if it conflicts with the block number in the OCR text.

The only possible parcel numbers for this block are given below. Do not return a parcel number that is not in the list,
if the OCR text is ambiguous, return the most likely parcel number from the list. If the OCR text indicates a
parcel number not in the list, return the most likely parcel number from the list.

Use neighboring page text only as contextual evidence. If OCR failed to extract
anything for this page, use the contextual evidence to infer the parcel number.
The current page is primary.
The literal token "OCR_MISSING" means that a page exists but OCR extracted no
usable text for the target region.

Neighboring pages may refer to:
- the same parcel
- a parcel identifier with a trailing letter
- the previous or next parcel
- a completely different parcel

In the majority of cases within a block folder, pages are ordered by increasing
parcel number. Therefore, if the current page OCR is ambiguous, use up to two
neighbors on each side to judge what values are plausible in sequence.
As a last resort, infer whether the current parcel is likely the same as, one
higher than, or one lower than nearby neighbors.
Do not copy a neighbor parcel number unless that is the only plausible interpretation.
If the OCR text and the sequential pattern disagree, prefer the OCR text unless
it is too ambiguous to trust.
Return low confidence when still uncertain.

Correct block number: {block_number}
{possible_parcel_numbers_text}

[Extraction target]: Current page text: \"{current_region_text}\"

{context_block}


Return ONLY a single JSON object with exactly these keys (no markdown, no code
fences, no explanation before or after):
{{
  "parcel": "string",
  "block": "string",
  "confidence": "high/medium/low"
}}
""".strip()


def _strip_markdown_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_from_json_code_fence(text: str) -> str | None:
    """Content between the first ```json and the next ``` (case-insensitive open tag)."""
    marker = "```json"
    lower = text.lower()
    idx = lower.find(marker.lower())
    if idx < 0:
        return None
    start = idx + len(marker)
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    end = text.find("```", start)
    if end < 0:
        return None
    return text[start:end].strip()


def _extract_first_json_object(text: str) -> str | None:
    """Find the first balanced {...} slice, respecting strings and escapes."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_id_from_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse Claude/model output that may include prose or markdown-wrapped JSON.

    Tries: ```json ... ``` substring, then full text, fenced strip, first balanced {...}.
    """
    if not response_text or not response_text.strip():
        raise json.JSONDecodeError("Expecting value: empty model response", str(response_text), 0)

    candidates: list[str] = []
    stripped = response_text.strip()

    fenced = _extract_json_from_json_code_fence(stripped)
    if fenced:
        candidates.append(fenced)
        inner = _extract_first_json_object(fenced)
        if inner and inner != fenced:
            candidates.append(inner)

    candidates.append(stripped)
    candidates.append(_strip_markdown_code_fences(stripped))

    inner = _extract_first_json_object(stripped)
    if inner:
        candidates.append(inner)
        fenced_inner = _strip_markdown_code_fences(inner)
        if fenced_inner != inner:
            candidates.append(fenced_inner)

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise json.JSONDecodeError("no JSON found", stripped, 0)
