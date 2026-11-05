#!/usr/bin/env python3
"""Download OGBench datasets to scratch.

Datasets are NOT fetched via `ogbench.make_env_and_datasets`: its default directory is
`~/.ogbench/data` (home quota is 93 GB on Stellar), and the auto-download would fire
inside a Slurm job, where outbound internet is not guaranteed. Everything is pulled
here, up front, from a login or vis node, and training always gets an explicit path.

Sharded 100M datasets live in a subdirectory as `<stem>-{000..099}.npz`, each shard a
self-contained 1M-transition file with its own `-val` sibling.

Usage:
    python src/scripts/download_datasets.py                  # the planned set
    python src/scripts/download_datasets.py --dry-run
    python src/scripts/download_datasets.py --deferred       # the big optional extras
    python src/scripts/download_datasets.py --dataset cube-double-play-100m-v0 --shards 10-19
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from offline.config import (  # noqa: E402
    DATA_DIR, DATASET_URL, DATASETS, DOWNLOAD_DEFERRED, DOWNLOAD_PLAN, DownloadItem,
)

CHUNK = 1 << 22  # 4 MiB
#: Tolerance when comparing a finished file against the registry's approximate size.
SIZE_TOLERANCE = 0.15


def _fmt(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except (urllib.error.URLError, OSError):
        return None


def download_one(url: str, dest: str, *, force: bool = False) -> tuple[str, int]:
    """Returns (status, bytes). Status is 'skip', 'ok' or 'resize'."""
    expected = remote_size(url)
    if os.path.exists(dest) and not force:
        have = os.path.getsize(dest)
        if expected is None or have == expected:
            return "skip", have
        # A partial file from an interrupted run: re-fetch.
        print(f"    size mismatch (have {_fmt(have)}, remote {_fmt(expected)}); re-fetching")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if expected:
                pct = 100 * got / expected
                print(f"\r    {_fmt(got)} / {_fmt(expected)}  ({pct:5.1f}%)", end="", flush=True)
    print("\r" + " " * 48 + "\r", end="")

    if expected is not None and got != expected:
        os.remove(tmp)
        raise IOError(f"{url}: got {got} bytes, expected {expected}")
    os.replace(tmp, dest)
    dt = time.time() - t0
    rate = got / dt / 1e6 if dt > 0 else 0
    print(f"    {os.path.basename(dest)}  {_fmt(got)}  in {dt:.0f}s  ({rate:.0f} MB/s)")
    return "ok", got


def plan_files(items: tuple[DownloadItem, ...]) -> list[tuple[str, str, str]]:
    """Expand download items into (url, dest, label) triples."""
    out: list[tuple[str, str, str]] = []
    for item in items:
        spec = DATASETS[item.dataset]
        for shard in item.shard_list():
            for rel in spec.rel_paths(shard):
                out.append((f"{DATASET_URL}/{rel}", os.path.join(DATA_DIR, rel), rel))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    ap.add_argument("--deferred", action="store_true",
                    help="fetch DOWNLOAD_DEFERRED instead of DOWNLOAD_PLAN (large)")
    ap.add_argument("--dataset", help="override: a single dataset key from the registry")
    ap.add_argument("--shards", help="override: inclusive shard range, e.g. 10-19")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()

    if args.dataset:
        shards = None
        if args.shards:
            lo, hi = (int(x) for x in args.shards.split("-"))
            shards = (lo, hi)
        items = (DownloadItem(args.dataset, shards, "cli override"),)
    else:
        items = DOWNLOAD_DEFERRED if args.deferred else DOWNLOAD_PLAN

    files = plan_files(items)
    missing = [f for f in files if not os.path.exists(f[1])]

    print(f"target   {args.data_dir}")
    print(f"planned  {len(files)} files, {len(missing)} missing")
    free = shutil.disk_usage(os.path.dirname(args.data_dir.rstrip("/")) or "/").free
    print(f"free     {_fmt(free)} (filesystem; your scratch QUOTA is 1000 GB -- check with `checkquota`)")

    if args.dry_run:
        for url, dest, label in files:
            print(f"  {'have' if os.path.exists(dest) else 'need'}  {label}")
        return 0

    n_ok = n_skip = 0
    total = 0
    t0 = time.time()
    for i, (url, dest, label) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {label}")
        try:
            status, got = download_one(url, dest, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
            return 1
        total += got
        if status == "skip":
            n_skip += 1
            print("    already present")
        else:
            n_ok += 1

    dt = time.time() - t0
    print(f"\ndone: {n_ok} downloaded, {n_skip} already present, "
          f"{_fmt(total)} on disk, {dt:.0f}s elapsed")

    # Cross-check the registry's declared sizes against reality.
    print("\nsize check vs registry:")
    for item in items:
        spec = DATASETS[item.dataset]
        for shard in item.shard_list():
            train_rel, _ = spec.rel_paths(shard)
            p = os.path.join(args.data_dir, train_rel)
            if not os.path.exists(p) or not spec.train_bytes:
                continue
            actual = os.path.getsize(p)
            drift = abs(actual - spec.train_bytes) / spec.train_bytes
            if drift > SIZE_TOLERANCE:
                print(f"  DRIFT {train_rel}: registry {_fmt(spec.train_bytes)}, "
                      f"actual {_fmt(actual)} -- update config.py")
    print("  ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
