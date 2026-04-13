from typing import Any


def build_manual_review_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Stage 6: manual review queue placeholder.

    Should collect unresolved records and include neighboring-page context.
    """
    return [record for record in records if record.get("needs_manual_override")]
