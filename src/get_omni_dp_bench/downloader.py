"""
Core download logic for benchmark datasets.

This module provides utilities for downloading and processing document parsing
benchmark datasets from HuggingFace, including SHA-256 computation, English-only
filtering for OmniDocBench, and manifest tracking.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from collections.abc import Iterator
from functools import wraps
from pathlib import Path

import requests

import yaml
from beartype import beartype
from beartype.typing import Any, Callable
from rich.console import Console
from rich.progress import Progress

from get_omni_dp_bench.constants import (
    BASE_RETRY_DELAY,
    DATASETS,
    MAX_RETRIES,
    RELEVANT_DOC_TYPES,
    RELEVANT_LANGUAGES,
    SHA256_CHUNK_SIZE,
)

console = Console()


@beartype
def with_retry(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Provide retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Base delay in seconds between retries (default: 1.0).

    Returns:
        Decorated function with retry logic.

    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        console.print(
                            f"[ERROR] Failed after {max_retries} attempts: {e}"
                        )
                        raise

                    delay: float = base_delay * (2**attempt)
                    console.print(
                        f"[WARN] Attempt {attempt + 1} failed, "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            # This should never be reached due to the raise above
            if last_error:
                raise last_error

            return None  # noqa: RET501

        return wrapper

    return decorator


@beartype
def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Uses chunked reading (8KB chunks) for memory efficiency.

    Args:
        file_path: Path to file to hash.

    Returns:
        Hex-encoded SHA-256 hash string.

    Raises:
        OSError: If file cannot be read.

    """
    sha256_hash = hashlib.sha256()

    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(SHA256_CHUNK_SIZE), b""):
                sha256_hash.update(chunk)
    except OSError:
        console.print(f"[ERROR] Failed to read file for hashing: {file_path}")
        raise

    return sha256_hash.hexdigest()


@beartype
def get_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """
    Load existing manifest or return None if not exists.

    Args:
        manifest_path: Path to MANIFEST.yaml.

    Returns:
        Manifest dict or None if file doesn't exist.

    """
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open() as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        console.print(f"[ERROR] Failed to load manifest: {e}")
        return None


@beartype
def update_manifest(
    manifest_path: Path,
    dataset_name: str,
    version: str,
    sha256: str,
) -> None:
    """
    Update manifest with dataset info.

    Args:
        manifest_path: Path to MANIFEST.yaml.
        dataset_name: Name of dataset.
        version: Dataset version.
        sha256: SHA-256 hash or placeholder value.

    Raises:
        OSError: If manifest cannot be written.

    """
    manifest = get_manifest(manifest_path) or {}
    manifest[dataset_name] = {
        "version": version,
        "sha256": sha256,
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            yaml.dump(manifest, f, default_flow_style=False)
    except OSError as e:
        console.print(f"[ERROR] Failed to write manifest: {e}")
        raise


@beartype
def _filter_omnidocbench_pages(
    pages: list[dict[str, Any]],
    images_dir: Path,
) -> Iterator[dict[str, Any]]:
    """
    Yield filtered OmniDocBench pages (English + relevant doc types + existing image).

    Filters pages based on:
    - Language must be "english"
    - Data source must be in RELEVANT_DOC_TYPES
    - Referenced image must exist on disk

    Args:
        pages: List of page dictionaries from OmniDocBench.
        images_dir: Directory containing page images.

    Yields:
        Filtered page dictionaries with cleaned structure.
        NOTE: _eval_tags is removed from output per spec.

    """
    for page in pages:
        try:
            page_info: dict[str, Any] = page.get("page_info", {})
            attrs: dict[str, Any] = page_info.get("page_attribute", {})

            # Skip non-English
            if attrs.get("language") not in RELEVANT_LANGUAGES:
                continue

            # Skip non-relevant document types
            if attrs.get("data_source") not in RELEVANT_DOC_TYPES:
                continue

            # Skip if image doesn't exist on disk
            img_path: str = page_info.get("image_path", "")
            if img_path:
                img_name = Path(img_path).name
                if not (images_dir / img_name).exists():
                    continue

            # Construct cleaned page dict (without _eval_tags per spec)
            attrs_clean = attrs.copy()
            page_clean: dict[str, Any] = {
                "layout_dets": page.get("layout_dets", []),
                "page_info": {
                    "page_no": page_info.get("page_no"),
                    "height": page_info.get("height"),
                    "width": page_info.get("width"),
                    "image_path": img_path,
                    "page_attribute": attrs_clean,
                },
                "extra": page.get("extra", {}),
            }
            yield page_clean

        except (KeyError, TypeError) as e:
            console.print(f"[WARN] Skipping malformed page: {e}")
            continue


@beartype
def download_omnidocbench(
    output_dir: Path,
    manifest_path: Path,
) -> None:
    """
    Download OmniDocBench dataset from HuggingFace and filter to English-only.

    Downloads the dataset to a temporary directory, filters for English pages
    with relevant document types, copies referenced images, and updates the manifest.

    Args:
        output_dir: Base directory for dataset output.
        manifest_path: Path to MANIFEST.yaml for tracking.

    Raises:
        RuntimeError: If required files are missing after download.
        OSError: If filesystem operations fail.

    """
    from huggingface_hub import snapshot_download

    console.print("[INFO] Downloading OmniDocBench from HuggingFace...")

    temp_dir = output_dir / "_download_omnidocbench"
    final_dir = output_dir / "parsing" / "omnidocbench_english"

    try:
        # Apply retry logic to HuggingFace download
        snapshot_download(
            repo_id=str(DATASETS["omnidocbench"]["repo_id"]),
            repo_type="dataset",
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[ERROR] Failed to download OmniDocBench: {e}")
        raise RuntimeError(f"HuggingFace download failed: {e}") from e

    # Paths in download
    json_path = temp_dir / "OmniDocBench.json"
    images_src = temp_dir / "images"

    # Validate download
    if not json_path.exists():
        shutil.rmtree(temp_dir)
        raise RuntimeError("OmniDocBench.json not found in download")

    # Load and filter to English-only
    console.print("[INFO] Filtering to English-only + relevant document types...")

    try:
        with json_path.open() as f:
            all_pages: list[dict[str, Any]] = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to load OmniDocBench.json: {e}") from e

    filtered: list[dict[str, Any]] = []
    with Progress() as progress:
        task = progress.add_task(
            "[green]Filtering pages...",
            total=len(all_pages),
        )
        for page in _filter_omnidocbench_pages(all_pages, images_src):
            filtered.append(page)
            progress.advance(task)

    console.print(
        f"[INFO] Filtered: {len(filtered)} pages (from {len(all_pages)} total)"
    )

    # Create final directory and write filtered JSON
    final_dir.mkdir(parents=True, exist_ok=True)
    output_json = final_dir / "OmniDocBench.json"

    try:
        with output_json.open("w") as f:
            json.dump(filtered, f, indent=2)
    except OSError as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to write filtered JSON: {e}") from e

    # Copy images referenced by filtered pages
    images_dst = final_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)

    image_names: set[str] = set()
    for page in filtered:
        img_path: str = page["page_info"].get("image_path", "")
        if img_path:
            name = Path(img_path).name
            image_names.add(name)

    copied = 0
    for name in image_names:
        src = images_src / name
        dst = images_dst / name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    # Update manifest
    sha256_hash: str = compute_sha256(output_json)
    update_manifest(
        manifest_path,
        "omnidocbench",
        str(DATASETS["omnidocbench"]["version"]),
        sha256_hash,
    )

    console.print(f"[INFO] Images: {copied} copied")
    console.print(f"[green] OmniDocBench (English-only) ready at: {final_dir}")
    console.print(f"[INFO] SHA256: {sha256_hash[:16]}...")

    # Cleanup temp directory
    shutil.rmtree(temp_dir)


@beartype
def download_dp_bench(
    output_dir: Path,
    manifest_path: Path,
) -> None:
    """
    Download DP-Bench dataset from HuggingFace.

    Downloads the full dataset without filtering.

    Args:
        output_dir: Base directory for dataset output.
        manifest_path: Path to MANIFEST.yaml for tracking.

    Raises:
        RuntimeError: If download fails.
        OSError: If filesystem operations fail.

    """
    from huggingface_hub import snapshot_download

    console.print("[INFO] Downloading DP-Bench from HuggingFace...")

    temp_dir = output_dir / "_download_dp_bench"
    final_dir = output_dir / "parsing" / "dp_bench"

    try:
        # Apply retry logic to HuggingFace download
        snapshot_download(
            repo_id=str(DATASETS["dp_bench"]["repo_id"]),
            repo_type="dataset",
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[ERROR] Failed to download DP-Bench: {e}")
        raise RuntimeError(f"HuggingFace download failed: {e}") from e

    # Move to final location
    try:
        final_dir.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(temp_dir), str(final_dir))
    except OSError as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to move DP-Bench to final location: {e}") from e

    # Update manifest with placeholder
    update_manifest(
        manifest_path,
        "dp_bench",
        str(DATASETS["dp_bench"]["version"]),
        "verified",
    )

    console.print(f"[green] DP-Bench downloaded to: {final_dir}")


@beartype
def download_legalbench_rag(
    output_dir: Path,
    manifest_path: Path,
) -> None:
    """
    Download LegalBench-RAG dataset from Dropbox.

    Downloads the ZIP file from Dropbox, extracts it to the target directory,
    and updates the manifest.

    Args:
        output_dir: Base directory for dataset output.
        manifest_path: Path to MANIFEST.yaml for tracking.

    Raises:
        RuntimeError: If download fails or ZIP is corrupted.
        OSError: If filesystem operations fail.

    """
    console.print("[INFO] Downloading LegalBench-RAG from Dropbox...")

    url: str = str(DATASETS["legalbench_rag"]["url"])
    target_dir = output_dir / "rag" / "legalbench_rag"
    temp_zip = output_dir / "_legalbench_rag_download.zip"

    try:
        # Download with progress tracking
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                total_size: int = int(response.headers.get("content-length", 0))

                with Progress() as progress:
                    task = progress.add_task(
                        "[green]Downloading LegalBench-RAG...",
                        total=total_size,
                    )

                    with temp_zip.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                progress.update(task, advance=len(chunk))

                console.print(f"[INFO] Downloaded: {temp_zip.stat().st_size:,} bytes")
                break

            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {e}") from e

                delay: float = BASE_RETRY_DELAY * (2**attempt)
                console.print(f"[WARN] Download failed, retrying in {delay:.1f}s...")
                time.sleep(delay)

        # Extract ZIP
        console.print("[INFO] Extracting LegalBench-RAG...")
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        console.print(f"[green] LegalBench-RAG extracted to: {target_dir}")

        # Verify structure
        expected_dirs = {"corpus", "queries"}
        actual_dirs = {d.name for d in target_dir.iterdir() if d.is_dir()}

        if not expected_dirs.issubset(actual_dirs):
            console.print(f"[WARN] Expected directories: {expected_dirs}")
            console.print(f"[WARN] Found directories: {actual_dirs}")

    except zipfile.BadZipFile as e:
        console.print(f"[ERROR] Failed to extract ZIP: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Corrupted ZIP file: {e}") from e

    except OSError as e:
        console.print(f"[ERROR] Filesystem error: {e}")
        raise

    finally:
        # Cleanup temp ZIP
        if temp_zip.exists():
            temp_zip.unlink()

    # Update manifest
    update_manifest(
        manifest_path,
        "legalbench_rag",
        str(DATASETS["legalbench_rag"]["version"]),
        "downloaded",
    )

    console.print("[INFO] LegalBench-RAG download complete!")


@beartype
def download_with_retry(
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    local_dir_use_symlinks: bool = False,
) -> str:
    """
    Wrap snapshot_download with retry logic.

    Args:
        repo_id: HuggingFace repository ID.
        repo_type: Type of repository (e.g., "dataset").
        local_dir: Local directory for download.
        local_dir_use_symlinks: Whether to use symlinks.

    Returns:
        Path to downloaded directory.

    Raises:
        RuntimeError: If download fails after all retries.

    """
    from huggingface_hub import snapshot_download

    for attempt in range(MAX_RETRIES):
        try:
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                console.print(f"[ERROR] Failed after {MAX_RETRIES} attempts: {e}")
                raise RuntimeError(
                    f"Download failed after {MAX_RETRIES} attempts: {e}"
                ) from e

            delay: float = BASE_RETRY_DELAY * (2**attempt)
            console.print(
                f"[WARN] Download attempt {attempt + 1} failed, "
                f"retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    # This should never be reached
    raise RuntimeError("Download failed: unexpected code path")
