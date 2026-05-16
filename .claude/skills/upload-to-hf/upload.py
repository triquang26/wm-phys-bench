#!/usr/bin/env python3
"""Upload a file or folder to a Hugging Face dataset repo and print a shareable URL.

Usage:
    python upload.py <path> [--repo REPO_ID] [--in-repo PATH_IN_REPO]
                            [--private] [--message MSG] [--bucket]

Examples:
    # Single video -> default repo, same filename at repo root
    python upload.py results/run.mp4

    # Folder -> default repo, placed under runs/<folder-name>/
    python upload.py results/

    # Custom repo and target path
    python upload.py output.mp4 --repo myname/my-videos --in-repo demos/output.mp4

    # Upload to an HF Bucket (xet) instead of a Dataset repo
    python upload.py results/run.mp4 --bucket
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_TOKEN = "REDACTED_HF_TOKEN"
DEFAULT_REPO = "twanghcmut/feepe-uploads"
DEFAULT_BUCKET = "twanghcmut/claude-codecode"


def get_token() -> str:
    return os.environ.get("HF_TOKEN") or DEFAULT_TOKEN


def view_url(repo_id: str, path_in_repo: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path_in_repo}"


def tree_url(repo_id: str, path_in_repo: str = "", revision: str = "main") -> str:
    base = f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"
    return f"{base}/{path_in_repo}".rstrip("/")


def ensure_repo(api, repo_id: str, private: bool, token: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )


def upload_file_to_dataset(api, path: Path, repo_id: str, in_repo: str,
                           message: str, token: str) -> str:
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=message,
    )
    return view_url(repo_id, in_repo)


def upload_folder_to_dataset(api, path: Path, repo_id: str, in_repo: str,
                             message: str, token: str) -> str:
    api.upload_folder(
        folder_path=str(path),
        path_in_repo=in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=message,
    )
    return tree_url(repo_id, in_repo)


def upload_to_bucket(path: Path, bucket_id: str, in_repo: str, token: str) -> str:
    """Upload using `hf sync` to an HF Bucket."""
    import subprocess
    src = str(path) if path.is_dir() else str(path.parent)
    # `hf sync` syncs a *folder* — for single files we sync the parent and
    # limit by file pattern via --include.
    extra = []
    if path.is_file():
        extra = ["--include", path.name]
        dest = f"hf://buckets/{bucket_id}/{in_repo.rsplit('/', 1)[0] or ''}".rstrip("/")
    else:
        dest = f"hf://buckets/{bucket_id}/{in_repo}".rstrip("/")
    cmd = ["hf", "sync", src, dest, "--token", token, *extra]
    print("Running:", " ".join(cmd[:-2]), "--token ***")
    subprocess.run(cmd, check=True)
    return f"https://huggingface.co/buckets/{bucket_id}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Local file or folder to upload")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"Dataset repo id (default: {DEFAULT_REPO})")
    p.add_argument("--in-repo", default=None, help="Destination path inside the repo")
    p.add_argument("--private", action="store_true", help="Create the repo as private if it doesn't exist")
    p.add_argument("--message", default=None, help="Commit message")
    p.add_argument("--bucket", action="store_true", help="Upload to an HF Bucket via `hf sync` instead of a Dataset repo")
    p.add_argument("--bucket-id", default=DEFAULT_BUCKET, help=f"Bucket id when --bucket (default: {DEFAULT_BUCKET})")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: path does not exist: {path}", file=sys.stderr)
        return 2

    token = get_token()
    ts = time.strftime("%Y%m%d-%H%M%S")
    default_in_repo = path.name if path.is_file() else f"runs/{path.name}-{ts}"
    in_repo = args.in_repo or default_in_repo
    message = args.message or f"Upload {path.name} ({ts})"

    if args.bucket:
        url = upload_to_bucket(path, args.bucket_id, in_repo, token)
        print("\nUploaded to bucket. Browse here:")
        print(url)
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    ensure_repo(api, args.repo, args.private, token)

    if path.is_file():
        url = upload_file_to_dataset(api, path, args.repo, in_repo, message, token)
        print("\nDirect file URL (open on phone):")
    else:
        url = upload_folder_to_dataset(api, path, args.repo, in_repo, message, token)
        print("\nFolder tree URL (browse on phone):")
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
