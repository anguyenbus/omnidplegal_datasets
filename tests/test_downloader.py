"""Tests for dataset downloader functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from get_omni_dp_bench import constants
from get_omni_dp_bench.cli import main
from get_omni_dp_bench.downloader import (
    _filter_omnidocbench_pages,
    compute_sha256,
    download_dp_bench,
    download_omnidocbench,
    download_with_retry,
    get_manifest,
    show_legalbench_instructions,
    update_manifest,
    with_retry,
)


class TestProjectFoundation:
    """Test suite for project structure and setup."""

    def test_package_imports_work(self) -> None:
        """Test that the package can be imported correctly."""
        import get_omni_dp_bench

        assert get_omni_dp_bench.__version__ == "0.1.0"

    def test_constants_module_exists(self) -> None:
        """Test that constants module can be imported."""

        assert hasattr(constants, "RELEVANT_DOC_TYPES")
        assert hasattr(constants, "RELEVANT_LANGUAGES")
        assert hasattr(constants, "DATASETS")

    def test_relevant_doc_types_has_all_6_types(self) -> None:
        """Test that all 6 required document types are defined."""
        from get_omni_dp_bench.constants import RELEVANT_DOC_TYPES

        expected_types = {
            "academic_literature",
            "research_report",
            "exam_paper",
            "colorful_textbook",
            "book",
            "PPT2PDF",
        }

        assert expected_types == RELEVANT_DOC_TYPES

    def test_relevant_languages_is_english_only(self) -> None:
        """Test that only English is included in relevant languages."""
        from get_omni_dp_bench.constants import RELEVANT_LANGUAGES

        assert {"english"} == RELEVANT_LANGUAGES

    def test_datasets_config_has_all_entries(self) -> None:
        """Test that DATASETS config has all required entries."""
        from get_omni_dp_bench.constants import DATASETS

        assert "omnidocbench" in DATASETS
        assert "dp_bench" in DATASETS
        assert "legalbench_rag" in DATASETS

        assert DATASETS["omnidocbench"]["repo_id"] == "opendatalab/OmniDocBench"
        assert DATASETS["omnidocbench"]["version"] == "v1.0"
        assert DATASETS["dp_bench"]["repo_id"] == "upstage/dp-bench"
        assert DATASETS["legalbench_rag"]["repo_id"] is None


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_compute_sha256_with_known_input(self, tmp_path: Path) -> None:
        """Test SHA-256 computation produces correct hash for known input."""
        # Create a test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Known SHA-256 hash for "hello world"
        expected_hash = (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

        result = compute_sha256(test_file)
        assert result == expected_hash

    def test_compute_sha256_empty_file(self, tmp_path: Path) -> None:
        """Test SHA-256 computation for empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        # Known SHA-256 hash for empty string
        expected_hash = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

        result = compute_sha256(test_file)
        assert result == expected_hash

    def test_filter_omnidocbench_pages_english_only(self, tmp_path: Path) -> None:
        """Test filtering keeps only English pages with relevant doc types."""
        # Create test images directory
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "page1.jpg").touch()

        # Sample pages data
        pages = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 1000,
                    "width": 800,
                    "image_path": "images/page1.jpg",
                    "page_attribute": {
                        "language": "english",
                        "data_source": "academic_literature",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
            {
                "page_info": {
                    "page_no": 2,
                    "height": 1000,
                    "width": 800,
                    "image_path": "images/page2.jpg",
                    "page_attribute": {
                        "language": "chinese",
                        "data_source": "academic_literature",
                    },
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        filtered = list(_filter_omnidocbench_pages(pages, images_dir))

        # Only English page should be kept
        assert len(filtered) == 1
        assert filtered[0]["page_info"]["page_no"] == 1

    def test_filter_omnidocbench_pages_no_eval_tags(self, tmp_path: Path) -> None:
        """Test that _eval_tags is removed from filtered output."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "page1.jpg").touch()

        pages = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 1000,
                    "width": 800,
                    "image_path": "images/page1.jpg",
                    "page_attribute": {"language": "english", "data_source": "book"},
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        filtered = list(_filter_omnidocbench_pages(pages, images_dir))

        # _eval_tags should not be in output
        assert "_eval_tags" not in filtered[0]
        assert "page_info" in filtered[0]
        assert "layout_dets" in filtered[0]
        assert "extra" in filtered[0]

    def test_filter_omnidocbench_pages_missing_image_skipped(
        self, tmp_path: Path
    ) -> None:
        """Test that pages with missing images are skipped."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        pages = [
            {
                "page_info": {
                    "page_no": 1,
                    "height": 1000,
                    "width": 800,
                    "image_path": "images/missing.jpg",
                    "page_attribute": {"language": "english", "data_source": "book"},
                },
                "layout_dets": [],
                "extra": {},
            },
        ]

        filtered = list(_filter_omnidocbench_pages(pages, images_dir))

        # Page should be skipped since image doesn't exist
        assert len(filtered) == 0

    def test_get_manifest_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test get_manifest returns None for non-existent file."""
        manifest_path = tmp_path / "MANIFEST.yaml"

        result = get_manifest(manifest_path)
        assert result is None

    def test_get_manifest_loads_existing(self, tmp_path: Path) -> None:
        """Test get_manifest loads existing manifest."""
        manifest_path = tmp_path / "MANIFEST.yaml"
        manifest_path.write_text(
            yaml.dump({"omnidocbench": {"version": "v1.0", "sha256": "abc123"}}),
        )

        result = get_manifest(manifest_path)
        assert result is not None
        assert result["omnidocbench"]["version"] == "v1.0"
        assert result["omnidocbench"]["sha256"] == "abc123"

    def test_update_manifest_creates_new(self, tmp_path: Path) -> None:
        """Test update_manifest creates new file."""
        manifest_path = tmp_path / "MANIFEST.yaml"

        update_manifest(manifest_path, "dp_bench", "v1.0", "verified")

        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        assert data["dp_bench"]["version"] == "v1.0"
        assert data["dp_bench"]["sha256"] == "verified"

    def test_update_manifest_appends_to_existing(self, tmp_path: Path) -> None:
        """Test update_manifest appends to existing manifest."""
        manifest_path = tmp_path / "MANIFEST.yaml"
        manifest_path.write_text(
            yaml.dump({"omnidocbench": {"version": "v1.0", "sha256": "abc123"}}),
        )

        update_manifest(manifest_path, "dp_bench", "v1.0", "verified")

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        assert "omnidocbench" in data
        assert "dp_bench" in data
        assert len(data) == 2

    def test_retry_decorator_success_on_first_attempt(self) -> None:
        """Test retry decorator succeeds immediately."""

        @with_retry(max_retries=3, base_delay=0.01)
        def successful_func() -> str:
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_decorator_retries_then_succeeds(self) -> None:
        """Test retry decorator retries before succeeding."""
        attempts = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def flaky_func() -> str:
            attempts[0] += 1
            if attempts[0] < 2:
                raise RuntimeError("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert attempts[0] == 2

    def test_retry_decorator_fails_after_max_retries(self) -> None:
        """Test retry decorator fails after exhausting retries."""

        @with_retry(max_retries=2, base_delay=0.01)
        def failing_func() -> str:
            raise RuntimeError("Permanent failure")

        with pytest.raises(RuntimeError, match="Permanent failure"):
            failing_func()


class TestDownloadFunctions:
    """Test suite for download functions."""

    @patch("huggingface_hub.snapshot_download")
    def test_download_omnidocbench_with_mock(
        self,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test OmniDocBench download with mocked HuggingFace API."""
        import json

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest_path = tmp_path / "MANIFEST.yaml"

        # Mock snapshot_download to create the test structure
        def mock_download(*args: object, **kwargs: object) -> str:
            # Create the directory structure where snapshot_download would
            local_dir = kwargs.get("local_dir")
            if local_dir:
                temp_dir = Path(local_dir)
            else:
                temp_dir = output_dir / "_download_omnidocbench"

            temp_dir.mkdir(parents=True, exist_ok=True)
            images_dir = temp_dir / "images"
            images_dir.mkdir()

            # Create mock image file
            (images_dir / "page1.jpg").write_bytes(b"fake image")

            # Create mock JSON data
            mock_pages = [
                {
                    "page_info": {
                        "page_no": 1,
                        "height": 1000,
                        "width": 800,
                        "image_path": "images/page1.jpg",
                        "page_attribute": {
                            "language": "english",
                            "data_source": "book",
                        },
                    },
                    "layout_dets": [],
                    "extra": {},
                },
            ]
            (temp_dir / "OmniDocBench.json").write_text(json.dumps(mock_pages))
            return str(temp_dir)

        mock_snapshot_download.side_effect = mock_download

        # This should succeed with mocked download
        download_omnidocbench(output_dir, manifest_path)

        # Verify output was created
        final_dir = output_dir / "parsing" / "omnidocbench_english"
        assert final_dir.exists()
        assert (final_dir / "OmniDocBench.json").exists()
        assert (final_dir / "images" / "page1.jpg").exists()

        # Verify temp directory was cleaned up
        assert not (output_dir / "_download_omnidocbench").exists()

    @patch("huggingface_hub.snapshot_download")
    def test_download_dp_bench_with_mock(
        self,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test DP-Bench download with mocked HuggingFace API."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest_path = tmp_path / "MANIFEST.yaml"

        def mock_download(*args: object, **kwargs: object) -> str:
            local_dir = kwargs.get("local_dir")
            if local_dir:
                temp_dir = Path(local_dir)
            else:
                temp_dir = output_dir / "_download_dp_bench"

            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "reference.json").write_text("{}")
            return str(temp_dir)

        mock_snapshot_download.side_effect = mock_download

        download_dp_bench(output_dir, manifest_path)

        # Verify output was created
        final_dir = output_dir / "parsing" / "dp_bench"
        assert final_dir.exists()
        assert (final_dir / "reference.json").exists()

        # Verify temp directory was cleaned up (moved, not deleted)
        assert not (output_dir / "_download_dp_bench").exists()

    @patch("huggingface_hub.snapshot_download")
    def test_download_with_retry_success_after_failure(
        self,
        mock_snapshot_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test download_with_retry succeeds after temporary failure."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        attempt_count = [0]

        def mock_download_with_retry(*args: object, **kwargs: object) -> str:
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise RuntimeError("Temporary network error")

            local_dir = kwargs.get("local_dir")
            temp_dir = Path(local_dir) if local_dir else output_dir / "_test"
            temp_dir.mkdir(parents=True, exist_ok=True)
            return str(temp_dir)

        mock_snapshot_download.side_effect = mock_download_with_retry

        # Should succeed after retry
        result = download_with_retry(
            repo_id="test/repo",
            repo_type="dataset",
            local_dir=output_dir / "_test",
        )

        # Verify retry happened
        assert attempt_count[0] == 2

    def test_show_legalbench_instructions(self, tmp_path: Path) -> None:
        """Test LegalBench instructions display."""
        output_dir = tmp_path / "output"
        manifest_path = tmp_path / "MANIFEST.yaml"

        show_legalbench_instructions(output_dir, manifest_path)

        # Verify manifest was updated
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        assert "legalbench_rag" in data
        assert data["legalbench_rag"]["sha256"] == "manual"


class TestCLI:
    """Test suite for CLI interface."""

    def test_cli_help_displays_options(self, runner: CliRunner) -> None:
        """Test that --help displays all options."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "--datasets" in result.output
        assert "--output-dir" in result.output
        assert "--force" in result.output
        assert "Download and prepare benchmark datasets" in result.output

    @patch("get_omni_dp_bench.cli.download_dp_bench")
    @patch("huggingface_hub.snapshot_download")
    def test_cli_datasets_dp_bench(
        self,
        mock_snapshot_download: MagicMock,
        mock_download_dp_bench: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --datasets dp_bench downloads only dp_bench."""

        # Setup mock
        def mock_download(*args: object, **kwargs: object) -> str:
            local_dir = kwargs.get("local_dir")
            temp_dir = Path(local_dir) if local_dir else tmp_path / "_download_dp_bench"
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "reference.json").write_text("{}")
            return str(temp_dir)

        mock_snapshot_download.side_effect = mock_download

        result = runner.invoke(
            main, ["--datasets", "dp_bench", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        mock_download_dp_bench.assert_called_once()

    @patch("get_omni_dp_bench.cli.download_omnidocbench")
    @patch("get_omni_dp_bench.cli.download_dp_bench")
    @patch("get_omni_dp_bench.cli.show_legalbench_instructions")
    def test_cli_datasets_all(
        self,
        mock_legalbench: MagicMock,
        mock_dp_bench: MagicMock,
        mock_omnidocbench: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --datasets all downloads all datasets."""
        result = runner.invoke(
            main, ["--datasets", "all", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        mock_omnidocbench.assert_called_once()
        mock_dp_bench.assert_called_once()
        mock_legalbench.assert_called_once()

    @patch("get_omni_dp_bench.cli.download_omnidocbench")
    def test_cli_force_flag_re_downloads(
        self,
        mock_omnidocbench: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --force flag re-downloads existing datasets."""
        # Create existing output directory
        (tmp_path / "parsing" / "omnidocbench_english").mkdir(parents=True)

        result = runner.invoke(
            main,
            ["--datasets", "omnidocbench", "--force", "--output-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        mock_omnidocbench.assert_called_once()


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for CLI runner."""
    return CliRunner()
