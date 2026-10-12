"""One-time download of ELF's pre-tokenized OpenWebText (T5) into a local dir.

Usage:
    uv run python -m scripts.download_data [--out data/openwebtext-t5]
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download openwebtext-t5 to a local dir")
    parser.add_argument("--repo", default="embedded-language-flows/openwebtext-t5")
    parser.add_argument("--out", default="data/openwebtext-t5")
    args = parser.parse_args()

    path = snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=args.out)
    print(f"downloaded {args.repo} -> {path}")


if __name__ == "__main__":
    main()