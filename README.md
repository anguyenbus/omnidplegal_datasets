# Get Omni DP Bench

CLI tool to download and prepare English-only document parsing benchmark datasets from HuggingFace.

## Datasets

### OmniDocBench (English-only)

Multi-domain document parsing benchmark covering 6 document types:

| Type | Description |
|------|-------------|
| `academic_literature` | Research papers, journal articles |
| `research_report` | Industry reports, white papers |
| `exam_paper` | Test papers, examinations |
| `colorful_textbook` | Educational textbooks with color |
| `book` | Monographs, books |
| `PPT2PDF` | Presentation slides converted to PDF |

**Source**: [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench)

**Filtering Applied**:
- Language: English only (`language == "english"`)
- Images: Only pages with existing image files
- Quality: All 6 document types included (no type exclusions)

### DP-Bench

Document parsing benchmark from Upstage. Full dataset download without filtering.

**Source**: [upstage/dp-bench](https://huggingface.co/datasets/upstage/dp-bench)

### LegalBench-RAG

Legal document retrieval augmented generation benchmark. Manual download required (instructions provided by CLI).

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install package
pip install -e .

# Install dev dependencies
pip install pytest pytest-cov ruff
```

## Usage

```bash
# Activate environment
source env/bin/activate

# Download all datasets
get-data

# Download specific dataset
get-data --datasets omnidocbench
get-data --datasets dp_bench

# Force re-download (overwrite existing)
get-data --force

# Custom output directory
get-data --output-dir /path/to/data

# Show help
get-data --help
```

## Output Structure

```
data/
├── MANIFEST.yaml                 # Dataset versions and hashes
└── parsing/
    ├── omnidocbench_english/
    │   ├── OmniDocBench.json     # Filtered pages (English-only)
    │   └── images/               # Referenced page images
    └── dp_bench/                 # Full DP-Bench dataset
```

## JSON Schema Contract

### OmniDocBench Page Schema

Each page in `OmniDocBench.json` follows this structure:

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

### Schema Fields

| Field | Type | Description |
|-------|------|-------------|
| `layout_dets` | array | Layout detections (bounding boxes, text) |
| `page_info` | object | Page metadata |
| `page_info.page_no` | int | Page number (1-indexed) |
| `page_info.height` | int | Page height in pixels |
| `page_info.width` | int | Page width in pixels |
| `page_info.image_path` | str | Relative path to page image |
| `page_attribute.language` | str | Document language |
| `page_attribute.data_source` | str | Document type (see types above) |
| `page_attribute.layout` | str | Layout pattern |
| `extra` | object | Additional metadata |

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/get_omni_dp_bench --cov-report=term-missing

# Lint
ruff check --fix
ruff format
```

## Manifest

`data/MANIFEST.yaml` tracks downloaded datasets:

```yaml
omnidocbench:
  version: v1.0
  sha256: abc123...
dp_bench:
  version: v1.0
  sha256: verified
legalbench_rag:
  version: v1.0
  sha256: manual
```

## License

Refer to individual dataset licenses:
- OmniDocBench: See [HuggingFace dataset card](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- DP-Bench: See [HuggingFace dataset card](https://huggingface.co/datasets/upstage/dp-bench)
