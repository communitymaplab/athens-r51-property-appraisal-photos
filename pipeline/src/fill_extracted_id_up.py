"""
Propagate ``extracted_id`` upward within each block (R tidyr ``fill(..., .direction = \"up\")``).

Only ``loose_photo``, ``photo_sheet``, and ``appraisal_form`` participate; ``other`` is ignored
and does not break ordering (those pages are omitted from the fill sequence).
"""

from __future__ import annotations

import copy
from typing import Any

from models import PageArtifact

_FILL_CLASSIFICATIONS = frozenset({"loose_photo", "photo_sheet", "appraisal_form"})


def _extracted_id_has_parcel(extracted_id: Any) -> bool:
    if not isinstance(extracted_id, dict):
        return False
    parcel = extracted_id.get("parcel")
    if parcel is None:
        return False
    if isinstance(parcel, str) and not parcel.strip():
        return False
    return True


def fill_extracted_id_up_per_block(artifacts_by_block: dict[str, list[PageArtifact]]) -> int:
    """
    For each block, consider only eligible classifications in existing filename order.
    Walk from last page toward first; propagate the nearest ``extracted_id`` with a parcel
    upward into rows that lack one.

    Returns the number of artifacts that received a filled ``extracted_id``.
    """
    filled_count = 0
    for block_arts in artifacts_by_block.values():
        eligible = [a for a in block_arts if a.classification in _FILL_CLASSIFICATIONS]
        carry: dict[str, Any] | None = None
        for art in reversed(eligible):
            if _extracted_id_has_parcel(art.extracted_id):
                carry = copy.deepcopy(art.extracted_id)
            elif carry is not None:
                art.extracted_id = copy.deepcopy(carry)
                filled_count += 1
    return filled_count
