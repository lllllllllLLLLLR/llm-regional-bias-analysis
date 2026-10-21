#!/usr/bin/env python3
"""
Clone a model from Hugging Face to the local 'models' folder.

Usage:
  python clone_model.py llama
  python clone_model.py mistral
  python clone_model.py qwen

Llama requires accepting the license on Hugging Face and: huggingface-cli login
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download

MODELS_DIR = Path.cwd() / "models"

MODELS = {
    "llama": {
        "name": "Llama 2 7B",
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "local_dir": "llama-2-7b",
    },
    "mistral": {
        "name": "Mistral 7B",
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "local_dir": "mistral-7b",
    },
    "qwen": {
        "name": "Qwen 2 7B",
        "repo_id": "Qwen/Qwen2-7B",
        "local_dir": "qwen-2-7b",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Clone a model from Hugging Face (llama, mistral, or qwen) to models/"
    )
    parser.add_argument(
        "model",
        choices=["llama", "mistral", "qwen"],
        help="Model to clone: llama, mistral, or qwen",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Log in to Hugging Face (required for Llama)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Folder to save model (default: {MODELS_DIR})",
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    local_dir = models_dir / cfg["local_dir"]

    if args.login:
        login()

    print(f"Cloning {cfg['name']} from {cfg['repo_id']}...")
    print(f"Target: {local_dir.absolute()}\n")
    try:
        path = snapshot_download(
            repo_id=cfg["repo_id"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Done. Saved to {path}")
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
