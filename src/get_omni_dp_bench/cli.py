"""CLI entry point for dataset downloader."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from get_omni_dp_bench.downloader import (
    download_dp_bench,
    download_legalbench_rag,
    download_omnidocbench,
)

console = Console()


@click.command()
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    type=click.Choice(["dp_bench", "omnidocbench", "legalbench_rag", "all"]),
    default=["all"],
    help="Datasets to download",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("data"),
    help="Output directory for datasets",
    show_default=True,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-download even if datasets exist",
)
def main(
    datasets: tuple[str, ...],
    output_dir: Path,
    force: bool,
) -> None:
    """
    Download and prepare benchmark datasets.

    Downloads DP-Bench and OmniDocBench from HuggingFace, with optional
    English-only filtering for OmniDocBench. LegalBench-RAG requires
    manual download (instructions provided).
    """
    # Expand "all" to all datasets
    datasets_to_download: list[str] = list(datasets)
    if "all" in datasets_to_download:
        datasets_to_download = ["dp_bench", "omnidocbench", "legalbench_rag"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "MANIFEST.yaml"

    console.print("[INFO] Dataset downloader starting...")
    console.print(f"[INFO] Output directory: {output_dir}")
    console.print(f"[INFO] Datasets: {', '.join(datasets_to_download)}")

    for dataset in datasets_to_download:
        console.print(f"\n[bold yellow]Processing: {dataset}[/bold yellow]")

        try:
            if dataset == "omnidocbench":
                final_dir = output_dir / "parsing" / "omnidocbench_english"

                # Check if already exists
                if final_dir.exists() and not force:
                    console.print(
                        f"[INFO] OmniDocBench already exists at {final_dir}. "
                        "Use --force to re-download."
                    )
                else:
                    if force and final_dir.exists():
                        console.print("[INFO] --force flag set, re-downloading...")
                    download_omnidocbench(output_dir, manifest_path)

            elif dataset == "dp_bench":
                final_dir = output_dir / "parsing" / "dp_bench"

                if final_dir.exists() and not force:
                    console.print(
                        f"[INFO] DP-Bench already exists at {final_dir}. "
                        "Use --force to re-download."
                    )
                else:
                    if force and final_dir.exists():
                        console.print("[INFO] --force flag set, re-downloading...")
                    download_dp_bench(output_dir, manifest_path)

            elif dataset == "legalbench_rag":
                final_dir = output_dir / "rag" / "legalbench_rag"

                if final_dir.exists() and not force:
                    console.print(
                        f"[INFO] LegalBench-RAG already exists at {final_dir}. "
                        "Use --force to re-download."
                    )
                else:
                    if force and final_dir.exists():
                        console.print("[INFO] --force flag set, re-downloading...")
                    download_legalbench_rag(output_dir, manifest_path)

        except Exception as e:
            console.print(f"[ERROR] Failed to download {dataset}: {e}")
            raise

    console.print("\n[green]Download complete![/green]")
    console.print(f"[INFO] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
