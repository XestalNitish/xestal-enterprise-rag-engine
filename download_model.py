"""
download_model.py — XHToken/Spark-X2.5-4B-GGUF Downloader & Model Manager
========================================================================
Downloads the free open-weight Spark-X2.5-4B GGUF model for local CPU/GPU execution.

Usage in other Python files:
----------------------------
    from download_model import download_spark_model, get_model_path, is_model_downloaded

    # 1. Check if model is downloaded:
    if not is_model_downloaded():
        model_path = download_spark_model()
    else:
        model_path = get_model_path()

    print(f"Model ready at: {model_path}")
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ID = "XHToken/Spark-X2.5-4B-GGUF"

AVAILABLE_QUANTS = {
    "q4_k_m": "Spark-X2.5-1.7B-Q4_K_M.gguf",  # ~2.5 GB (Recommended)
    "q8_0":   "Spark-X2.5-4B-Q8_0.gguf",    # ~4.5 GB
    "full":   "Spark-X2.5-4B.gguf",        # ~8.0 GB
}

DEFAULT_QUANT = "q4_k_m"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "Spark-X2.5-1.7B-GGUF"


def get_model_path(quant: str = DEFAULT_QUANT, custom_dir: Optional[str] = None) -> Path:
    """Return the absolute path to the local GGUF file."""
    model_dir = Path(custom_dir) if custom_dir else DEFAULT_MODEL_DIR
    filename = AVAILABLE_QUANTS.get(quant.lower(), AVAILABLE_QUANTS[DEFAULT_QUANT])
    return model_dir / filename


def is_model_downloaded(quant: str = DEFAULT_QUANT, custom_dir: Optional[str] = None) -> bool:
    """Check if the model file exists and is non-empty."""
    path = get_model_path(quant, custom_dir)
    return path.exists() and path.stat().st_size > 100_000_000  # at least 100MB


def download_spark_model(
    quant: str = DEFAULT_QUANT,
    save_dir: Optional[str] = None,
    force_redownload: bool = False
) -> Path:
    """
    Download the GGUF model from Hugging Face with resume capability.

    Args:
        quant: 'q4_k_m' (default), 'q8_0', or 'full'.
        save_dir: directory to store the model.
        force_redownload: re-download even if file exists.

    Returns:
        Path to the downloaded model.
    """
    from huggingface_hub import hf_hub_download

    filename = AVAILABLE_QUANTS.get(quant.lower(), AVAILABLE_QUANTS[DEFAULT_QUANT])
    dest_dir = Path(save_dir) if save_dir else DEFAULT_MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / filename

    if not force_redownload and is_model_downloaded(quant, str(dest_dir)):
        size_gb = target_path.stat().st_size / (1024 ** 3)
        print(f"✅ Model already exists locally: {target_path} ({size_gb:.2f} GB)")
        return target_path

    print(f"📥 Starting download of '{filename}' from repo '{REPO_ID}'...")
    print(f"📁 Destination directory: {dest_dir}")
    print("⏳ This may take a few minutes depending on your internet connection...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        final_path = Path(downloaded_path)
        size_gb = final_path.stat().st_size / (1024 ** 3)
        print(f"\n🎉 Download Complete! Path: {final_path} ({size_gb:.2f} GB)")
        return final_path
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download XHToken/Spark-X2.5-4B-GGUF model")
    parser.add_argument("--quant", default="q4_k_m", choices=list(AVAILABLE_QUANTS.keys()), help="Quantization level")
    parser.add_argument("--check", action="store_true", help="Check if already downloaded")
    parser.add_argument("--dir", default=None, help="Custom save directory")
    args = parser.parse_args()

    if args.check:
        downloaded = is_model_downloaded(args.quant, args.dir)
        path = get_model_path(args.quant, args.dir)
        print(f"Downloaded: {downloaded} -> {path}")
    else:
        path = download_spark_model(quant=args.quant, save_dir=args.dir)
