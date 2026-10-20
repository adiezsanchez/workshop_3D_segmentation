"""Generate an S3 presigned URL for workshop data, store it, and push to git.

Requires AWS credentials via the ``mhread`` profile (repository owner only).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_URL_PATH = REPO_ROOT / "data_url.json"

REPOSITORY_NAME = "workshop_YAP_nuclear_3D_orgs"
BUCKET_NAME = "mhbucket"
FOLDER_PATH = "CMIC/raw_data/"
EXPIRATION = 7200
AWS_PROFILE = "mhread"
COMMIT_MESSAGE = "chore: refresh S3 presigned data URL"


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def generate_presigned_url() -> tuple[str, str, str]:
    """Return (url, etag, object_key) for the matching workshop zip."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3_client = session.client("s3")

    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_PATH)
    for obj in response.get("Contents", []):
        object_key = obj["Key"]
        file_name = object_key.split("/")[-1]
        if object_key.endswith(".zip") and REPOSITORY_NAME in object_key:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": object_key},
                ExpiresIn=EXPIRATION,
            )
            etag = obj["ETag"]
            print(f"Pre-signed URL for {file_name}: {url}")
            print(f"S3 Etag for {file_name}: {etag}")
            return url, etag, object_key

    raise RuntimeError(
        f"No .zip object containing '{REPOSITORY_NAME}' found under "
        f"s3://{BUCKET_NAME}/{FOLDER_PATH}"
    )


def write_data_url(url: str, etag: str, object_key: str) -> bool:
    """Write data_url.json. Return True if contents changed."""
    payload = {
        "repository_name": REPOSITORY_NAME,
        "object_key": object_key,
        "url": url,
        "etag": etag,
        "expires_in": EXPIRATION,
    }
    previous = None
    if DATA_URL_PATH.exists():
        previous = json.loads(DATA_URL_PATH.read_text(encoding="utf-8"))

    DATA_URL_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return previous != payload


def commit_and_push_if_changed(changed: bool) -> None:
    if not changed:
        print("data_url.json unchanged; skipping commit and push.")
        return

    _run_git("add", str(DATA_URL_PATH.relative_to(REPO_ROOT)))
    status = _run_git("status", "--porcelain", "--", str(DATA_URL_PATH.relative_to(REPO_ROOT)))
    if not status.stdout.strip():
        print("data_url.json unchanged in git index; skipping commit and push.")
        return

    _run_git("commit", "-m", COMMIT_MESSAGE)

    upstream = _run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False)
    if upstream.returncode != 0:
        raise RuntimeError(
            "No upstream branch configured. Set one with "
            "`git push -u origin <branch>` before running generate_link."
        )

    _run_git("push")
    print(f"Committed and pushed {DATA_URL_PATH.name} to {upstream.stdout.strip()}.")


def main() -> int:
    try:
        url, etag, object_key = generate_presigned_url()
    except NoCredentialsError:
        print(
            "Credentials not available. Configure the AWS profile "
            f"'{AWS_PROFILE}' and retry.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - surface owner-facing CLI errors
        print(f"Failed to generate presigned URL: {exc}", file=sys.stderr)
        return 1

    changed = write_data_url(url, etag, object_key)
    try:
        commit_and_push_if_changed(changed)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        print(f"Git operation failed: {detail}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
