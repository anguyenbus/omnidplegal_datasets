"""Constants for dataset downloader configuration."""

from __future__ import annotations

from beartype.typing import Dict, Final

# OmniDocBench filter settings
RELEVANT_DOC_TYPES: Final[set[str]] = {
    "academic_literature",
    "research_report",
    "exam_paper",
    "colorful_textbook",
    "book",
    "PPT2PDF",
}

RELEVANT_LANGUAGES: Final[set[str]] = {"english"}

# Dataset repositories and versions
DATASETS: Final[Dict[str, Dict[str, str | None]]] = {
    "omnidocbench": {
        "repo_id": "opendatalab/OmniDocBench",
        "version": "v1.0",
    },
    "dp_bench": {
        "repo_id": "upstage/dp-bench",
        "version": "v1.0",
    },
    "legalbench_rag": {
        "repo_id": None,  # Direct Dropbox download
        "url": "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1",
        "version": "v1.0",
    },
}

# Retry configuration
MAX_RETRIES: Final[int] = 3
BASE_RETRY_DELAY: Final[float] = 1.0

# File operations
SHA256_CHUNK_SIZE: Final[int] = 8192
