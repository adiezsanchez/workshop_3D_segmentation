"""Download and extract workshop raw data using the committed presigned URL."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_URL_PATH = REPO_ROOT / "data_url.json"
TEMP_DIR = REPO_ROOT / "temp_zip_parts"


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def ensure_data_url_matches_remote() -> None:
    """Fail if local data_url.json differs from the tracked upstream version."""
    inside = _run_git("rev-parse", "--is-inside-work-tree", check=False)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise RuntimeError("Not inside a git repository.")

    fetch = _run_git("fetch", check=False)
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout or "git fetch failed").strip()
        raise RuntimeError(f"Could not fetch from remote: {detail}")

    upstream = _run_git(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False
    )
    if upstream.returncode != 0:
        branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        remote_ref = f"origin/{branch}"
        has_remote = _run_git("rev-parse", "--verify", remote_ref, check=False)
        if has_remote.returncode != 0:
            raise RuntimeError(
                "No upstream branch configured and "
                f"'{remote_ref}' does not exist. Run `git pull` / set upstream first."
            )
        upstream_ref = remote_ref
    else:
        upstream_ref = upstream.stdout.strip()

    rel_path = DATA_URL_PATH.relative_to(REPO_ROOT).as_posix()
    diff = _run_git("diff", "--quiet", upstream_ref, "--", rel_path, check=False)
    if diff.returncode != 0:
        raise RuntimeError(
            f"Local {rel_path} does not match {upstream_ref}. "
            "Run `git pull` to sync the committed presigned URL, then retry."
        )

    print(f"Local {rel_path} matches {upstream_ref}.")


def load_presigned_url() -> str:
    if not DATA_URL_PATH.exists():
        raise RuntimeError(f"Missing {DATA_URL_PATH.name}. Run `git pull` first.")

    payload = json.loads(DATA_URL_PATH.read_text(encoding="utf-8"))
    url = payload.get("url") or ""
    if not url:
        raise RuntimeError(
            f"{DATA_URL_PATH.name} has an empty url. "
            "Ask the repository owner to run `pixi run generate_link`."
        )
    return url


def download_and_extract_file(
    url: str, filename: Path, retries: int = 5, backoff_factor: float = 1.0
) -> None:
    """Download a file with retry logic, print MD5, extract, then remove the zip."""
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, stream=True, timeout=(10, 60))
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {filename.name}",
            )
            md5_hash = hashlib.md5()

            with open(filename, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                    md5_hash.update(data)
            progress_bar.close()

            file_md5 = md5_hash.hexdigest()
            print(f"MD5 hash of {filename}: {file_md5}")

            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(REPO_ROOT)
            print(f"{filename} extracted.")

            os.remove(filename)
            return
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as exc:
            wait = backoff_factor * (2**attempt)
            print(f"Download error: {exc}, retrying in {wait} seconds...")
            time.sleep(wait)
            attempt += 1

    raise RuntimeError(f"Failed to download and extract after {retries} attempts.")


def main() -> int:
    try:
        ensure_data_url_matches_remote()
        url = load_presigned_url()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    part_filename = TEMP_DIR / "part_1.zip"

    try:
        download_and_extract_file(url, part_filename)
        if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
            TEMP_DIR.rmdir()
    except Exception as exc:  # noqa: BLE001 - surface CLI errors
        print(str(exc), file=sys.stderr)
        return 1

    print("All files downloaded, extracted, and cleaned up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
