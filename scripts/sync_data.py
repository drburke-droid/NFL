#!/usr/bin/env python3
"""Move the big, un-versioned data files between PCs via a GitHub Release.

The odds DB (1.1 GB) and the derived .pkl caches are deliberately not in git --
32 pushed LFS versions exhausted the 10 GB LFS quota. Release assets are the
supported way to ship large binaries on GitHub: 2 GB per file, and they do NOT
count against the LFS quota.

    python scripts/sync_data.py push     # work PC -> GitHub  (uploads ~216 MB)
    python scripts/sync_data.py pull      # GitHub -> home PC
    python scripts/sync_data.py status    # what's local vs. what's published

Auth: set GH_TOKEN, or drop a personal access token (repo scope) into
config/gh_token.txt (gitignored). Only `push` needs a token -- `pull` works
unauthenticated as long as the repo is public, which is what draft night wants.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG = "data-latest"

DB_PATH = REPO_ROOT / "db" / "nfl_odds.db"
DB_ASSET = "nfl_odds.db.gz"
CACHE_ASSET = "derived_caches.tar.gz"
MANIFEST_ASSET = "manifest.json"

# Derived caches that used to live in git. Regenerable from the DB, but only on a
# machine that HAS the DB -- so we ship them rather than stranding the home PC.
CACHE_GLOBS = ["outputs/*.pkl", "outputs/models/*.pkl"]

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"


# --------------------------------------------------------------------------- util


def repo_slug():
    """owner/repo parsed from the origin remote."""
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    slug = url.split("github.com")[-1].lstrip(":/")
    return slug[:-4] if slug.endswith(".git") else slug


def token(required=True):
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        f = REPO_ROOT / "config" / "gh_token.txt"
        if f.exists():
            tok = f.read_text(encoding="utf-8").strip()
    if not tok and required:
        sys.exit(
            "No token. Set GH_TOKEN, or write a PAT (repo scope) to "
            "config/gh_token.txt\n"
            "  https://github.com/settings/tokens"
        )
    return tok


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def human(n):
    return f"{n / 1048576:.0f} MB"


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """S3 rejects a request carrying both our token and its own signed params."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.headers = {
                k: v for k, v in new.headers.items() if k.lower() != "authorization"
            }
        return new


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


def api(path, tok=None, method="GET", body=None, full_url=None, accept=None):
    url = full_url or f"{API}/repos/{repo_slug()}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", accept or "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if data:
        req.add_header("Content-Type", "application/json")
    with _opener.open(req) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def get_release(tok=None):
    try:
        return api(f"/releases/tags/{TAG}", tok)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


# --------------------------------------------------------------------------- push


def upload(rel, name, path, tok):
    for asset in rel.get("assets", []):
        if asset["name"] == name:
            print(f"  replacing existing {name}")
            api(full_url=asset["url"], tok=tok, method="DELETE")
    size = path.stat().st_size
    print(f"  uploading {name} ({human(size)}) ...")
    url = f"{UPLOADS}/repos/{repo_slug()}/releases/{rel['id']}/assets?name={name}"
    with open(path, "rb") as fh:
        req = urllib.request.Request(url, data=fh, method="POST")
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(size))
        req.add_header("Accept", "application/vnd.github+json")
        with _opener.open(req) as resp:
            resp.read()
    print(f"  done {name}")


def cmd_push(args):
    tok = token()
    if not DB_PATH.exists():
        sys.exit(f"Missing {DB_PATH} -- nothing to push.")

    tmp = Path(tempfile.mkdtemp(prefix="nflsync-"))
    try:
        manifest = {"db": {}, "caches": []}

        gz = tmp / DB_ASSET
        print(f"compressing {DB_PATH.name} ({human(DB_PATH.stat().st_size)}) ...")
        with open(DB_PATH, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        manifest["db"] = {
            "raw_sha256": sha256(DB_PATH),
            "raw_bytes": DB_PATH.stat().st_size,
            "gz_bytes": gz.stat().st_size,
        }
        print(f"  -> {human(gz.stat().st_size)} compressed")

        tar = tmp / CACHE_ASSET
        members = sorted(
            p for g in CACHE_GLOBS for p in REPO_ROOT.glob(g) if p.is_file()
        )
        with tarfile.open(tar, "w:gz") as tf:
            for p in members:
                rel_name = p.relative_to(REPO_ROOT).as_posix()
                tf.add(p, arcname=rel_name)
                manifest["caches"].append(
                    {"path": rel_name, "sha256": sha256(p), "bytes": p.stat().st_size}
                )
        print(f"bundled {len(members)} derived caches -> {human(tar.stat().st_size)}")

        man = tmp / MANIFEST_ASSET
        man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        rel = get_release(tok)
        if rel is None:
            print(f"creating release {TAG} ...")
            rel = api(
                "/releases", tok, method="POST",
                body={
                    "tag_name": TAG,
                    "name": "Data bundle (odds DB + derived caches)",
                    "body": (
                        "Large data files kept out of git to protect the LFS quota.\n"
                        "Fetch with `python scripts/sync_data.py pull`."
                    ),
                },
            )

        upload(rel, DB_ASSET, gz, tok)
        upload(rel, CACHE_ASSET, tar, tok)
        upload(rel, MANIFEST_ASSET, man, tok)
        print(f"\nPublished to https://github.com/{repo_slug()}/releases/tag/{TAG}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- pull


def download(rel, name, dest, tok):
    asset = next((a for a in rel.get("assets", []) if a["name"] == name), None)
    if asset is None:
        return None
    print(f"  downloading {name} ({human(asset['size'])}) ...")
    try:
        api_url, accept = asset["url"], "application/octet-stream"
        req = urllib.request.Request(api_url)
        req.add_header("Accept", accept)
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with _opener.open(req) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 22)
    except urllib.error.HTTPError:
        # public-repo fallback: unauthenticated CDN URL
        with _opener.open(asset["browser_download_url"]) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 22)
    return dest


def cmd_pull(args):
    tok = token(required=False)
    rel = get_release(tok)
    if rel is None:
        sys.exit(f"No release tagged {TAG}. Run `sync_data.py push` on the other PC first.")

    tmp = Path(tempfile.mkdtemp(prefix="nflsync-"))
    try:
        manifest = {}
        man = download(rel, MANIFEST_ASSET, tmp / MANIFEST_ASSET, tok)
        if man:
            manifest = json.loads(man.read_text(encoding="utf-8"))

        want = manifest.get("db", {}).get("raw_sha256")
        if DB_PATH.exists() and want and not args.force:
            if sha256(DB_PATH) == want:
                print(f"{DB_PATH.name} already matches the published copy -- skipping.")
            else:
                sys.exit(
                    f"{DB_PATH} exists and DIFFERS from the published copy.\n"
                    "Re-run with --force to overwrite it, or push your local one instead."
                )
        else:
            gz = download(rel, DB_ASSET, tmp / DB_ASSET, tok)
            if gz:
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                staged = tmp / "nfl_odds.db"
                print("  decompressing ...")
                with gzip.open(gz, "rb") as src, open(staged, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 22)
                if want and sha256(staged) != want:
                    sys.exit("Checksum mismatch after decompress -- aborting, DB untouched.")
                shutil.move(str(staged), str(DB_PATH))
                print(f"  wrote {DB_PATH} ({human(DB_PATH.stat().st_size)})")

        tar = download(rel, CACHE_ASSET, tmp / CACHE_ASSET, tok)
        if tar:
            with tarfile.open(tar, "r:gz") as tf:
                safe = [
                    m for m in tf.getmembers()
                    if not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts
                ]
                tf.extractall(REPO_ROOT, members=safe)
            print(f"  restored {len(safe)} derived caches")

        print("\nPull complete.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------- status


def cmd_status(args):
    tok = token(required=False)
    print(f"repo: {repo_slug()}")
    if DB_PATH.exists():
        print(f"local DB: {human(DB_PATH.stat().st_size)}  {DB_PATH}")
    else:
        print(f"local DB: MISSING ({DB_PATH})")
    caches = [p for g in CACHE_GLOBS for p in REPO_ROOT.glob(g) if p.is_file()]
    print(f"local derived caches: {len(caches)} files, "
          f"{human(sum(p.stat().st_size for p in caches))}")

    rel = get_release(tok)
    if rel is None:
        print(f"published: no release tagged {TAG}")
        return
    print(f"published release {TAG} (updated {rel.get('published_at')}):")
    for a in rel.get("assets", []):
        print(f"  {a['name']:<24} {human(a['size']):>8}  "
              f"{a.get('download_count', 0)} downloads")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push", help="compress + upload the DB and caches").set_defaults(fn=cmd_push)
    p = sub.add_parser("pull", help="download the DB and caches")
    p.add_argument("--force", action="store_true",
                   help="overwrite a local DB that differs from the published one")
    p.set_defaults(fn=cmd_pull)
    sub.add_parser("status", help="compare local files against the release").set_defaults(fn=cmd_status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
