# OmniDocBench Filter Approach

## Overview

OmniDocBench is a multilingual document parsing benchmark containing documents in multiple languages and various document types. This document describes the approach used to extract an English-only subset suitable for document parsing evaluation.

## Problem Statement

The raw OmniDocBench dataset contains:
- Multiple languages (English, Chinese, etc.)
- Various document types (some relevant, some not)
- Pages with missing image files

For English-only document parsing benchmarking, we need a filtered subset containing only English documents with all referenced images present.

## Filter Criteria

The filter applies three sequential criteria:

### 1. Language Filter

```python
RELEVANT_LANGUAGES = {"english"}
```

Only pages where `page_info.page_attribute.language == "english"` are retained.

### 2. Document Type Filter

```python
RELEVANT_DOC_TYPES = {
    "academic_literature",
    "research_report",
    "exam_paper",
    "colorful_textbook",
    "book",
    "PPT2PDF",
}
```

Only pages where `page_info.page_attribute.data_source` matches one of these types are retained.

### 3. Image Existence Filter

Each page includes a reference to an image file (`page_info.image_path`). Only pages where the referenced image file exists on disk are retained. This prevents broken references in the filtered dataset.

## Filtering Algorithm

The filtering logic in [`_filter_omnidocbench_pages()`](../src/get_omni_dp_bench/downloader.py#L180) processes pages as a generator:

```python
def _filter_omnidocbench_pages(
    pages: list[dict[str, Any]],
    images_dir: Path,
) -> Iterator[dict[str, Any]]:
    for page in pages:
        page_info = page.get("page_info", {})
        attrs = page_info.get("page_attribute", {})

        # Language filter
        if attrs.get("language") not in RELEVANT_LANGUAGES:
            continue

        # Document type filter
        if attrs.get("data_source") not in RELEVANT_DOC_TYPES:
            continue

        # Image existence filter
        img_path = page_info.get("image_path", "")
        if img_path:
            img_name = Path(img_path).name
            if not (images_dir / img_name).exists():
                continue

        # Pass all filters - yield cleaned page
        yield page_clean
```

## Output Structure

After filtering, the output dataset contains:

```
data/parsing/omnidocbench_english/
├── OmniDocBench.json    # Filtered pages (English-only)
└── images/              # Only images referenced by filtered pages
```

## Page Schema

Each filtered page maintains the original structure:

```json
{
  "layout_dets": [
    {
      "category": "text",
      "bbox": [x1, y1, x2, y2],
      "text": "content"
    }
  ],
  "page_info": {
    "page_no": 1,
    "height": 2339,
    "width": 1654,
    "image_path": "images/xxx.jpg",
    "page_attribute": {
      "language": "english",
      "data_source": "academic_literature",
      "layout": "single_column",
      "fuzzy_scan": false,
      "watermark": false,
      "colorful_backgroud": false
    }
  },
  "extra": {}
}
```

## Key Implementation Details

1. **Generator Pattern**: Uses Python generators (`yield`) for memory-efficient streaming of filtered pages.

2. **Graceful Degradation**: Malformed pages (missing keys, type errors) are logged and skipped rather than causing failure.

3. **Image Copying**: After filtering, only images referenced by filtered pages are copied to the output directory, reducing storage requirements.

4. **Manifest Tracking**: The filtered dataset's SHA-256 hash is computed and stored in `MANIFEST.yaml` for verification.

## Statistics

Typical reduction from full OmniDocBench to English-only subset:

- Raw pages: ~100K+ (all languages, all types)
- Filtered pages: ~20K-40K (English-only, relevant types, with images)
- Reduction: ~60-80% of original size

## References

- Source dataset: [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- Implementation: [`src/get_omni_dp_bench/downloader.py`](../src/get_omni_dp_bench/downloader.py)
- Constants: [`src/get_omni_dp_bench/constants.py`](../src/get_omni_dp_bench/constants.py)
