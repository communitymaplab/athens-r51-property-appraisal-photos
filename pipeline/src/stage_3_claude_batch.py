import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"


def sanitize_cache_key(relative_path: str) -> str:
    stem = relative_path.replace("\\", "/").replace("/", "__")
    for char in [".", " ", ":", "(", ")", "[", "]", "{", "}", ","]:
        stem = stem.replace(char, "_")
    return stem


def compute_input_hash(model: str, prompt: str) -> str:
    payload = f"{model}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_page_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_page_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def chunk_requests(requests_list: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        return [requests_list]
    return [requests_list[i:i + chunk_size] for i in range(0, len(requests_list), chunk_size)]


def submit_batch(
    api_key: str,
    requests_list: list[dict[str, Any]],
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    response = requests.post(
        ANTHROPIC_BASE_URL,
        headers=anthropic_headers(api_key),
        json={"requests": requests_list},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def get_batch_status(api_key: str, batch_id: str, timeout_seconds: int = 60) -> dict[str, Any]:
    response = requests.get(
        f"{ANTHROPIC_BASE_URL}/{batch_id}",
        headers=anthropic_headers(api_key),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def get_batch_results_response(
    api_key: str,
    batch_status: dict[str, Any],
    timeout_seconds: int = 120,
) -> requests.Response:
    results_url = batch_status.get("results_url")
    if results_url:
        response = requests.get(results_url, headers=anthropic_headers(api_key), timeout=timeout_seconds)
    else:
        batch_id = str(batch_status["id"])
        response = requests.get(
            f"{ANTHROPIC_BASE_URL}/{batch_id}/results",
            headers=anthropic_headers(api_key),
            timeout=timeout_seconds,
        )
    response.raise_for_status()
    return response


def parse_results_payload(response: requests.Response) -> list[dict[str, Any]]:
    content_type = response.headers.get("content-type", "")
    text = response.text.strip()
    if not text:
        return []

    if "application/json" in content_type:
        payload = response.json()
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                return payload["data"]
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

    # Fall back to JSONL parsing
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_text_from_result_item(item: dict[str, Any]) -> str | None:
    result = item.get("result", {})
    if isinstance(result, dict):
        # Common batch shape: result.message.content[].text
        message = result.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text = chunk.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()

        # Alternate shape: result.output_text
        output_text = result.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()

    # Alternate top-level shape
    output_text = item.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    return None


def wait_for_batch_completion(
    api_key: str,
    batch_id: str,
    poll_interval_seconds: int = 10,
    max_wait_seconds: int = 3600,
) -> dict[str, Any]:
    start = time.time()
    while True:
        status = get_batch_status(api_key=api_key, batch_id=batch_id)
        processing_status = str(status.get("processing_status", "")).lower()
        if processing_status in {"ended", "completed", "finished"}:
            return status
        if processing_status in {"failed", "errored", "canceled", "cancelled"}:
            raise RuntimeError(f"Claude batch {batch_id} failed with status: {processing_status}")
        print(f"Claude batch {batch_id} is still processing: {processing_status}")
        if time.time() - start > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for Claude batch {batch_id}")
        time.sleep(poll_interval_seconds)
