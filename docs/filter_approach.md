# OmniDocBench Filter Approach

## The Problem

OmniDocBench is a large dataset used to test document parsing systems. It contains:

- Documents in many languages (English, Chinese, and others)
- Different types of documents (academic papers, reports, textbooks, etc.)
- Pages that reference image files

If you want to benchmark English-only document parsing, you can't use the raw dataset directly. You need to filter out:

1. Non-English documents
2. Document types you don't care about
3. Pages with broken or missing image references

## The Solution

We filter the dataset using three simple rules. Each page must pass all three to be included.

### Rule 1: Language Check

Keep only pages marked as English. The language field might have variations like `"English"`, `"ENGLISH"`, or `" english "`, so we normalize by converting to lowercase and trimming spaces.

### Rule 2: Document Type Check

Keep only specific document types that are relevant for parsing benchmarks:

- Academic literature (research papers, journal articles)
- Research reports (industry reports, white papers)
- Exam papers
- Colorful textbooks
- Books
- Presentation slides (PPT converted to PDF)

### Rule 3: Image Check

Every page references an image file. We only keep pages where that image file actually exists on disk. This prevents broken links in the final dataset.

## How It Works

The filtering process follows these steps:

1. **Download** the raw OmniDocBench dataset from HuggingFace to a temporary folder
2. **Filter** each page through the three rules above
3. **Collect** only the pages that pass all filters
4. **Copy** only the images that are referenced by the filtered pages
5. **Write** the cleaned data to the final location
6. **Clean up** the temporary download folder

## What You Get

After filtering, your output looks like this:

```
data/parsing/omnidocbench_english/
├── OmniDocBench.json    # Filtered pages (English-only)
└── images/              # Only the images actually used
```

The JSON file contains pages with this structure:

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
      "layout": "single_column"
    }
  }
}
```

## Key Design Choices

**Memory efficiency**: The filter processes pages one at a time using Python generators, not loading everything into memory at once.

**Graceful failure**: If a page is malformed or missing expected fields, we log a warning and skip it rather than crashing.

**Smaller output**: By only copying the images that are actually referenced by filtered pages, we save significant disk space.

**Verification**: A SHA-256 hash of the filtered dataset is computed and stored in `MANIFEST.yaml` so you can verify integrity later.

## Size Reduction

The raw OmniDocBench dataset has over 100,000 pages in multiple languages. After filtering:

- **Raw**: ~100K+ pages (all languages, all types)
- **Filtered**: ~20K-40K pages (English-only, relevant types, with images)
- **Reduction**: About 60-80% smaller than the original

## Running the Filter

```bash
# Install and run
uv sync --dev
uv run get-data --datasets omnidocbench
```

The filtered dataset will be saved to `data/parsing/omnidocbench_english/`.

## References

- Source dataset: [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- Implementation: [`src/get_omni_dp_bench/downloader.py`](../src/get_omni_dp_bench/downloader.py)
- Configuration: [`src/get_omni_dp_bench/constants.py`](../src/get_omni_dp_bench/constants.py)
